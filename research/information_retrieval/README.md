<!--
Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Compressing Neural Methods for Information Retrieval
Author: @spacemanidol

Neural Methods for information retrieval have shown tremendous promise. Leveraging language models like BERT and T5 single stage and multi stage systems have exploded and in some cases are able to outperform traditional sparse search(BM25) by 2-3x. 
Despite the improvement in quality neural methods prove trippy to use in production. Large model size makes index generation difficult and expensive and requires large GPU clusters. In this folder we explore how compression methods like structured pruning, unstructured pruning and distilliation can be used with the Neural Magic framework to bridge the gap from research to production. 

We experiment with compressing and optimizing sparse search by doing query prediction and expansion with a T5 model and dense retireval using BERT based bi-encoders. The goal of these experiments is to determine if neural models can be made deployable for any type of workload without using GPUs

### Doc2Query
Fill out when project is done

### DPR

### Elastic Search Implementation 

## Results


