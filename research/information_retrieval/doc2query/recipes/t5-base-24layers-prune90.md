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

---
# General Variables
num_epochs: &num_epochs 5

# Pruning Hyperparameters
init_sparsity: &init_sparsity 0.00
final_sparsity: &final_sparsity 0.90
pruning_start_epoch: &pruning_start_epoch 1
pruning_end_epoch: &pruning_end_epoch 4
update_frequency: &pruning_update_frequency 0.01


# Modifiers
training_modifiers:
  - !EpochRangeModifier
    end_epoch: *num_epochs
    start_epoch: 0.0

pruning_modifiers:
  - !GMPruningModifier
    params:
      - re:*.block.*.layer.*.SelfAttention.q.weight
      - re:*.block.*.layer.*.SelfAttention.k.weight
      - re:*.block.*.layer.*.SelfAttention.v.weight
      - re:*.block.*.layer.*.SelfAttention.o.weight
      - re:*.block.*.layer.*.DenseReluDense.wi.weight
      - re:*.block.*.layer.*.DenseReluDense.wo.weight

    start_epoch: *pruning_start_epoch
    end_epoch: *pruning_end_epoch
    init_sparsity: *init_sparsity
    final_sparsity: *final_sparsity
    inter_func: cubic
    update_frequency: *pruning_update_frequency
    leave_enabled: True
    mask_type: unstructured
    log_types: __ALL__
---

# BERT Model with Pruned Encoder Layers

This recipe defines a pruning strategy to sparsify all encoder and decoder layers of a T5 model at 90% sparsity. 
Training was done using one V100 GPUusing a training batch size of 16 with the
## Weights and Biases

- [TBD]())

## Training

To set up the training environment, use the requirements.txt in this repo to install dependencies. Steps for downloading and preparing data are also described in this repo's README.
Adjust the training command below with your setup for GPU device, checkpoint saving frequency, and logging options.

*training command*
```
python run_doc2query.py \
  --distill_teacher MODELS_DIR/teacher \
  --model_name_or_path bert-base-uncased \
  --dataset_name squad \
  --do_train \
  --fp16 \
  --do_eval \
  --evaluation_strategy epoch \
  --per_device_train_batch_size 16 \
  --learning_rate 5e-5 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir MODELS_DIR/sparse80 \
  --cache_dir cache \
  --preprocessing_num_workers 6 \
  --seed 42 \
  --num_train_epochs 30 \
  --distill_hardness 1.0 \
  --distill_temperature 2.0 \
  --recipe recipes/t5-base-24layers-prune90.md \
  --onnx_export_path MODELS_DIR/sparse80/onnx \
  --report_to wandb
```