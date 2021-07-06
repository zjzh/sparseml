import argparse
import os
import json
import itertools
import os

import threading

import transformers
from filelock import FileLock
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

def load_collection(filename):
    collection = {}
    with open(filename, 'r') as f:
        for l in f:
            l = l.strip().split('\t')
            try:
                collection[int(l[0])] = l[1]
            except:
                print(l)
    return collection

def chunked(it, size):
    it = iter(it)
    while True:
        p = tuple(itertools.islice(it, size))
        if not p:
            break
        yield p

def write_chunk(chunk, generated_answers,args):
    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--collection_file",
        default="data/collection.tsv",
        type=str,
        help="The msmarco passage collection file",
    )
    parser.add_argument(
        "--augmented_file",
        default="data/collection_augmented.json",
        type=str,
        help="Name for expanded document"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="Doc2Query predictions",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=32,
        help="length of document queries",
    )
    parser.add_argument(
        '--no_cuda',
        action="store_true",
        help="Use this to not use cuda")
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='batch_size for generation'
    )
    parser.add_argument(
        '--top_k',
        type=int,
        default=10,
        help='The number of highest probability vocabulary tokens to keep for top-k-filtering'
    )
    parser.add_argument(
        '--num_beams',
        type=int,
        default=3,
        help='Beam Size for beam search'
    )
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=3,
        help="number of queries to generate per passage",
    )
    args = parser.parse_args()

    print("Loading collection")
    collection = load_collection(args.collection_file)
    print("Collection loaded")
    device='cuda'
    if args.no_cuda:
        device='cpu'

    if os.path.exists(args.augmented_file):
        os.remove(args.augmented_file) #remove predictions if exists

    print("Loading model")
    config = AutoConfig.from_pretrained(args.model_name_or_path,)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
    model.to(device)
    model.resize_token_embeddings(len(tokenizer))
    print("Model Loaded")

    print("Augmenting passages")
    batches = 0
    with open(args.augmented_file, 'a') as w:
        for chunk in chunked(collection.items(),args.batch_size):
            d_chunk = dict(chunk)
            collection_keys = list(d_chunk.keys())
            if batches % 100 == 0:
                print("{} passage processed".format(batches*args.batch_size))
            input_ids = tokenizer(list(dict(chunk).values()), return_tensors='pt', padding=True, truncation=True)['input_ids'].to(device)
            outputs = model.generate(
                input_ids=input_ids,
                max_length=args.max_length,
                do_sample=True,
                top_k=args.top_k,
                num_beams=args.num_beams,
                num_return_sequences=args.num_return_sequences)
            #write_thread = threading.Thread(target=function_that_downloads, name="writer", args=some_args)
            #write_thread.start()
            for i in range(len(collection_keys)):
                query_augment = ''   
                doc_id = collection_keys[i]
                for j in range(args.num_return_sequences):
                    query_augment += ' '
                    query_augment += tokenizer.decode(outputs[i+j], skip_special_tokens=True)
                output_dict = {'id': doc_id, 'original_input': collection[doc_id], 'input': collection[doc_id] + query_augment}
                w.write(json.dumps(output_dict) + '\n')  
            batches += 1
     
if __name__ == "__main__":
    main()

