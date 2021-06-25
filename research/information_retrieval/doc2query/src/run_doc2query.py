# neuralmagic: no copyright
# flake8: noqa
# fmt: off
# isort: skip_file
#!/usr/bin/env python
# coding=utf-8
# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Example script for integrating spaseml with the transformers library to perform model distillation and pruning on GLUE tasks. 
This script is addopted from hugging face's implementation for the GLUEDataset. 
Hugging Face's original implementation is regularly updated and can be found at https://github.com/huggingface/transformers/blob/master/examples/pytorch/text-classification/run_glue.py
This script will:
- Load transformer based models
- Load a sparseml training and pruning optimizer
- Train on Target GLUE Task
- Evaluate on GLUE
- Export model to onnx.
##########
Command help:
usage: run_glue.py [-h] \
    [--teacher_model_name_or_path] \
    [--student_model_name_or_path] \
    [--task_name] \
    [--temperature] \
    [--distill_hardness] \
    [--dataset_name]  \
    [--num_train_epochs] \
    [--do_train] \
    [--do_eval] \
    [--per_device_train_batch_size] \
    [--per_device_eval_batch_size] \
    [--learning_rate]\
    [--output_dir] \
    [--overwrite_output_dir] \
    [--cache_dir]\
    [--preprocessing_num_workers] \
    [--seed] \
    [--nm_prune_config] \
    [--do_onnx_export] \
    [--onnx_export_path] \
    [--layers_to_keep] \

Train, prune, and evaluate a transformer base question answering model on squad. 
    -h, --help            show this help message and exit
    --teacher_model_name_or_path    The name or path of model which will be used for distilation.
                                    Note, this model needs to be trained for QA task already.
    --student_model_name_or_path    The path to the transformers model you wish to train
                                    or the name of the pretrained language model you wish
                                    to use. ex: bert-base-uncased.
    --task_name                     The name of the GLUE task which the model with train and evalute on. 
    --temperature                   Hyperparameter which controls model distilation 
    --distill_hardness              Hyperparameter which controls how much of the loss comes from teacher vs training labels
    --dataset_name                  The name of which dataset you want to use to train or
                                    your model. ex: squad for using SQuAD.
    --num_train_epochs              Paramater to control how many training epochs you wish
                                    your model to train.
    --do_train                      Boolean denoting if the model should be trained
                                    or not. Default is false.
    --do_eval                       Boolean denoting if the model should be evaluated
                                    or not. Default is false.
    --per_device_train_batch_size   Size of each training batch based on samples per GPU. 
                                    24 will fit in a 11gb GPU, 32 in a 16gb.
    --per_device_eval_batch_size    Size of each training batch based on samples per GPU. 
                                    24 will fit in a 11gb GPU, 32 in a 16gb.
    --learning_rate                 Learning rate initial float value. ex: 3e-5.
    --output_dir                    Path which model checkpoints and paths should be saved.
    --overwrite_output_dir          Boolean to define if the 
    --cache_dir                     Directiory which cached transformer files(datasets, models
                                    , tokenizers) are saved for fast loading. 
    --preprocessing_num_workers     The amount of cpu workers which are used to process datasets
    --seed                          Int which determines what random seed is for training/shuffling
    --nm_prune_config               Path to the neural magic prune configuration file. examples can
                                    be found in prune_config_files but are customized for bert-base-uncased. 
    --do_onnx_export                Boolean denoting if the model should be exported to onnx
    --onnx_export_path              Path where onnx model path will be exported. ex: onnx-export
    --layers_to_keep                Number of layers to keep from original model. Layers are dropped before training
    --max_seq_length                Int for the max sequence length to be parsed for glue tasks ex: 128 tokens.

##########
Example command for training a 95% sparse BERT SQUAD model for 1 epoch without distilation on the Quora Duplicate Question Task:
python examples/transformers/run_glue.py \
    --teacher_model_name_or_path NONE
    --student_model_name_or_path bert-base-uncased \
    --task_name QQP
    --dataset_name squad \
    --num_train_epochs 1 \
    --do_train \
    --do_eval \
    --per_device_train_batch_size 12 \
    --per_device_eval_batch_size 12 \
    --learning_rate 3e-5 \
    --max_seq_length 128 \
    --doc_stride 128 \
    --output_dir 95sparsity1epoch/ \
    --overwrite_output_dir \
    --cache_dir cache \
    --preprocessing_num_workers 8 \
    --seed 42 \
    --nm_prune_config prune_config_files/95sparsity1epoch.yaml \
    --do_onnx_export \
    --onnx_export_path 95sparsity1epoch/ \
    --distill_hardness 1.0 \
    --temperature 2.0 \
    --layers_to_keep 12 \
"""
import logging
import os
import json
import sys
from dataclasses import dataclass, field
from typing import Optional
import random
import math
import numpy

import pdb #Remove after done tweaking
import nltk
import wandb
import numpy
import torch
import torch.nn.functional as F
from datasets import load_dataset, load_metric

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


from transformers.optimization import (
    Adafactor,
    AdamW,
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,    
    get_polynomial_decay_schedule_with_warmup,
)

from transformers.file_utils import is_offline_mode
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version

from sparseml_utils import SparseMLSeq2SeqTrainer, export_model

logger = logging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    distill_teacher: Optional[str] = field(
        default=None, metadata={"help": "Path to a pretrained Doc2Query Model to be used for distillation"}
    )
    distill_temperature: Optional[float] = field(
        default=2.0, metadata={"help": "Temperature applied to teacher softmax for distillation."}
    )
    distill_hardness: Optional[float] = field(
        default=1.0, metadata={"help": "Proportion of loss coming from teacher model."}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    recipe: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a SparseML sparsification recipe, see https://github.com/neuralmagic/sparseml "
                  "for more information"},
    )
    onnx_export_path: Optional[str] = field(
        default=None, metadata={"help": "The filename and path which will be where onnx model is outputed"}
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(
        default="data/doc_query_train.json", metadata={"help": "The input training data file (a jsonlines or csv file)."}
    )
    validation_file: Optional[str] = field(
        default='data/doc_query_dev_small.json',
        metadata={
            "help": "An optional input evaluation data file to evaluate the metrics (rouge) on "
            "(a jsonlines or csv file)."
        },
    )
    test_file: Optional[str] = field(
        default='data/doc_query_to_predict.json',
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge) on " "(a jsonlines or csv file)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=256,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=32,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=32,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
            "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
            "during ``evaluate`` and ``predict``."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    num_beams: Optional[int] = field(
        default=1,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
            "which is used during ``evaluate`` and ``predict``."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default=None, metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )
    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    if data_args.source_prefix is None and model_args.model_name_or_path in [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
    ]:
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is the expected, e.g. with "
            "`--source_prefix 'summarize: ' `"
        )

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    set_seed(training_args.seed)

    if data_args.dataset_name is not None: #Need to upload datasets
        datasets = load_dataset(data_args.dataset_name, data_args.dataset_config_name, cache_dir=model_args.cache_dir)
    else:
        data_files = {}
        if data_args.train_file is not None and training_args.do_train:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None and training_args.do_eval:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if data_args.test_file is not None and training_args.do_predict:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
        datasets = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir)

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([numpy.prod(p.size()) for p in model_parameters])
    logger.info("Model has %s parameters", params)
    model.resize_token_embeddings(len(tokenizer))

    teacher_model = None
    if model_args.distill_teacher is not None:
        teacher_model = AutoModelForSeq2SeqLM.from_pretrained(
            model_args.distill_teacher,
            from_tf=bool(".ckpt" in model_args.distill_teacher),
            config=config,
            cache_dir=model_args.cache_dir,
        )
        teacher_model_parameters = filter(lambda p: p.requires_grad, teacher_model.parameters())
        params = sum([numpy.prod(p.size()) for p in teacher_model_parameters])
        logger.info("Teacher Model has %s parameters", params)
        teacher_model.resize_token_embeddings(len(tokenizer))
    
    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

    if training_args.do_train:
        column_names = datasets["train"].column_names
    elif training_args.do_eval:
        column_names = datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = datasets["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )

    def preprocess_function(examples):
        inputs = examples['input']
        targets = examples['target']
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    if training_args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        if "validation" not in datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = datasets["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
        eval_dataset = eval_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    def preprocess_predict_function(examples):
        inputs = examples['input']
        targets = examples['target']
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)
        model_inputs["doc_ids"] =  examples['target']
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    prediction_file = None
    if training_args.do_predict:
        prediction_file = os.path.join(training_args.output_dir, "predictions.txt")
        max_target_length = data_args.val_max_target_length
        if "test" not in datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = datasets["test"]
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))
        predict_dataset = predict_dataset.map(
            preprocess_predict_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    # Metric
    metric = load_metric("rouge")
    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]
        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        labels = numpy.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        if isinstance(preds, tuple):
            preds = preds[0]
        tpreds = torch.from_numpy(preds)
        _, words  = torch.topk(tpreds, 1, 2) #beam size 1
        words = words.squeeze(-1)
        preds = []
        logger.info("Post Processing Predictions")
        for i in range(words.shape[0]):
            eos_index = numpy.argwhere(words[i]==1).squeeze()
            if len(eos_index.shape) > 0 and len(eos_index) > 0:
                eos_index = numpy.argwhere(words[i]==1).squeeze()[0]
            else:
                eos_index = data_args.val_max_target_length
            preds.append(words[i][:int(eos_index) + 1])
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        logger.info("***********************\nROUGE METRICS :{}\n***********************\n".format(result))
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        prediction_lens = [numpy.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = numpy.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        #result['predictions'] = decoded_preds
        return result

    trainer = SparseMLSeq2SeqTrainer(
        data_args.recipe,
        teacher=teacher_model,
        prediction_file=prediction_file,
        distill_hardness=model_args.distill_hardness,
        distill_temperature=model_args.distill_temperature,
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(
            max_length=data_args.val_max_target_length, num_beams=data_args.num_beams, metric_key_prefix="eval"
        )
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        results = trainer.predict(
            predict_dataset,
            metric_key_prefix="predict",
            max_length=data_args.val_max_target_length,
            num_beams=data_args.num_beams,
        )
        prediction_size = len(results[2]['predict_predictions'])
        with open(os.path.join(training_args.output_dir, "predictions.txt"), "a") as writer:
            for i in range(prediction_size):
                writer.write("{}\n".format(json.dumps({"doc_id":predict_dataset[i]['doc_ids'], "prediction":results[2]['predict_predictions'][i]})))

    if training_args.push_to_hub:
        kwargs = {"finetuned_from": model_args.model_name_or_path, "tags": "summarization"}
        if data_args.dataset_name is not None:
            kwargs["dataset_tags"] = data_args.dataset_name
            if data_args.dataset_config_name is not None:
                kwargs["dataset_args"] = data_args.dataset_config_name
                kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
            else:
                kwargs["dataset"] = data_args.dataset_name

        trainer.push_to_hub(**kwargs)
    
    if data_args.onnx_export_path:
        logger.info("*** Export to ONNX ***")
        eval_dataloader = trainer.get_eval_dataloader(eval_dataset)
        export_model(model, eval_dataloader, data_args.onnx_export_path)
    return results

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()

if __name__ == "__main__":
    main()
