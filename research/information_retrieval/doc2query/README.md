# Doc2Query Compressed

Author: @spacemanidol

Doc2query introduced a simple and direct method to integrate neural information retrieval in context of tradition keyword search. Instead of introducing a neural ranking engine at query time neural methods are moved to index generation time. 
A sequence to sequence is trained with the input being passages(short context windows) and the target being the relevant query. Since the MSMARCO coprus features over 500,000 relevant passages methods like T5 can be leveraged. Unfortunatley, without compression existing T5 takes the index generation from 10 minutes(16 threads on a 14 core Intel(R) Xeon(R) Gold 5120 CPU @ 2.20GHz) to PLACEHOLDER > using 4 16 GB V100

## Results

| Method        | Sparsity | MRR @10 MSMARCO Dev | Latency(s) per 1000 queries | Index Generation (HH:MM:SS)|
|---------------|----------|---------------------|-----------------------------|----------------------------|
|BM25(Anserini) |0         |0.1874               |79.85                        |00:10:16                    |
|Doc2Query      |0         |                     |                             |                            |
|Doc2Query Prune|90        |                     |                             |                            |

#### Results by sparsity measured by ROUGEL


## Baseline
Download the data
```sh
cd data
wget https://msmarco.blob.core.windows.net/msmarcoranking/collectionandqueries.tar.gz
tar -xzvf collectionandqueries.tar.gz
rm collectionandqueries.tar.gz
cat queries.dev.tsv queries.train.tsv queries.eval.tsv > queries.tsv
```

To format the collections file, build simple index, run on msmarco dev set and evaluate which should produce output
```
mkdir data/base_collection
python src/convert_doc_collection_to_json.py --collection_path data/collection.tsv --output_path data/base_collection/collection
python src/make_doc2query_data.py --collection_file data/collection.tsv --query_file data/queries.tsv --train_qrel_file data/qrels.train.tsv --dev_qrel_file data/qrels.dev.tsv --output_file_prefix data/doc_query_
head -n 5500 data/doc_query_dev.json > data/doc_query_dev_small.json
python -m pyserini.index -collection JsonCollection -generator DefaultLuceneDocumentGenerator \
 -threads 16 -input data/base_collection \
 -index dataindexes/msmarco-passage-baseline -storePositions -storeDocvectors -storeRaw
python -m pyserini.search --topics data/queries.dev.small.tsv \
 --index data/indexes/msmarco-passage-baseline \
 --output runs/run.msmarco-passage.bm25baseline.tsv \
 --bm25 --output-format msmarco --hits 1000 --k1 0.82 --b 0.68
python src/msmarco_passage_eval.py data/qrels.dev.small.tsv runs/run.msmarco-passage.bm25baseline.tsv
#####################
MRR @10: 0.18741227770955543
QueriesRanked: 6980
#####################
```

### Doc2Query

#### Training (Dense)
Format the data for training and train a T5 Model for 5 epoch. Training will take about 5 hours on a server with 4 V100(16gb).

```sh
mkdir models
python src/run_doc2query.py --model_name_or_path t5-base --do_train --do_eval --evaluation_strategy epochs --source_prefix "summarize: " --output_dir models/doc2query_baseline --overwrite_output_dir --per_device_train_batch_size=1 2 --per_device_eval_batch_size=4 --cache_dir cache/ --save_strategy epoch --seed 42 --recipe recipes/noprune.yaml --num_train_epochs 10 --eval_accumulation_steps 10
```
#### Training (Sparse) 90% pruned encoder + decoder

```sh
python src/run_doc2query.py --model_name_or_path t5-base --do_train --do_eval --evaluation_strategy epoch --source_prefix "summarize: " --output_dir 90sparse-distill --distill_teacher doc2query_baseline/  --overwrite_output_dir --per_device_train_batch_size=12 --per_device_eval_batch_size=4 --cache_dir cache/ --save_strategy epoch --seed 42 --recipe recipes/90sparseencode-then-decode.yaml --distill_hardness 0.5  --num_train_epochs 10 --eval_accumulation_steps 10
```

#### Prediction and Index Generation
Predition happens best when the index file is sharded. Each GPU process produces can augment about 1000 passages a minute which means with one server with 4 V100 GPUs augmented collection creation will take about 36 hours. 
as a re Started 4:04pm
```sh
mkdir data/base_doc2query_collection
cp data/collection.tsv data/base_doc2query_collection
split -n 8 data/collection.tsv data/base_doc2query_collection/
CUDA_VISIBLE_DEVICES=0 python src/augment_collection.py --collection_file data/base_doc2query_collection/xaa --model_name_or_path models/doc2query_baseline/ --augmented_file data/base_doc2query_collection/xaa.json
```

### Prediction and Index Generation with ONNX Run Time
TBD
### Prediction and Index Generation with DeepSparse
TBD