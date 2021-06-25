# Doc2Query Compressed

Author: @spacemanidol

Doc2query introduced a simple and direct method to integrate neural information retrieval in context of tradition keyword search. Instead of introducing a neural ranking engine at query time neural methods are moved to index generation time. 
A sequence to sequence is trained with the input being passages(short context windows) and the target being the relevant query. Since the MSMARCO coprus features over 500,000 relevant passages methods like T5 can be leveraged. Unfortunatley, without compression existing T5 takes the index generation from 10 minutes(16 threads on a 14 core Intel(R) Xeon(R) Gold 5120 CPU @ 2.20GHz) to > using 4 16 GB V100
## Results

| Method       | Sparsity | MRR @10 MSMARCO Dev | Latency(s) per 1000 queries | Index Generation (HH:MM:SS)|Citation        |
|--------------|----------|---------------------|-----------------------------|----------------------------|----------------|
|BM25(Anserini)|0         |0.1874               |79.85                        |00:10:16                    |                |
|Doc2Query     |0         |


### Baseline
Download the data
```sh
cd data
wget https://msmarco.blob.core.windows.net/msmarcoranking/collectionandqueries.tar.gz
tar -xzvf collectionandqueries.tar.gz
rm collectionandqueries.tar.gz
cat queries.dev.tsv queries.train.tsv queries.eval.tsv > queries.tsv
```

To format the collections file, build simple index, run on msmarco dev set and evaluate which should produce outpu
```
mkdir data/base_collection
python src/convert_doc_collection_to_json.py --collection_path data/collection.tsv --output_path data/base_collection/collection
python -m pyserini.index -collection JsonCollection -generator DefaultLuceneDocumentGenerator \
 -threads 16 -input data/base_collection \
 -index indexes/msmarco-passage-baseline -storePositions -storeDocvectors -storeRaw
python -m pyserini.search --topics data/queries.dev.small.tsv \
 --index indexes/msmarco-passage-baseline \
 --output runs/run.msmarco-passage.bm25baseline.tsv \
 --bm25 --output-format msmarco --hits 1000 --k1 0.82 --b 0.68
python src/msmarco_passage_eval.py data/qrels.dev.small.tsv runs/run.msmarco-passage.bm25baseline.tsv
#####################
MRR @10: 0.18741227770955543
QueriesRanked: 6980
#####################
```

### Doc2Query

Format the data for training and train a T5 Model for 5 epoch. Training will take about 5 hours on a server with 4 V100(16gb) and generating predicted queries will take about 15 min

```sh
python src/make_doc2query_data.py --collection_file data/collection.tsv --query_file data/queries.tsv --train_qrel_file data/qrels.train.tsv --dev_qrel_file data/qrels.dev.tsv --output_file_prefix data/doc_query_
head -n 1000 data/doc_query_dev.json > data/doc_query_dev_small.json
python src/convert_doc_collection_to_json.py --collection_path data/collection.tsv --output_path data/doc_query_to_predict.json
python src/run_doc2query.py --model_name_or_path t5-base --do_train --do_eval --evaluation_strategy steps --eval_steps 2400 --source_prefix "summarize: " --output_dir doc2query_baseline --overwrite_output_dir --per_device_train_batch_size=16 --per_device_eval_batch_size=16 --cache_dir cache/ --save_strategy epoch --seed 42 --recipe recipes/t5-base-24layers-prune0.md
python run_doc2query.py --model_name_or_path doc2query_baseline  --source_prefix "summarize: "  --do_predict --overwrite_output_dir --per_device_eval_batch_size=100 --text_column input --summary_column target
```
