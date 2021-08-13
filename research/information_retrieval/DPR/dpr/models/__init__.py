#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

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

import importlib

"""
 'Router'-like set of methods for component initialization with lazy imports 
"""


def init_hf_bert_biencoder(args, **kwargs):
    if importlib.util.find_spec("transformers") is None:
        raise RuntimeError('Please install transformers lib')
    from .hf_models import get_bert_biencoder_components
    return get_bert_biencoder_components(args, **kwargs)


def init_hf_bert_reader(args, **kwargs):
    if importlib.util.find_spec("transformers") is None:
        raise RuntimeError('Please install transformers lib')
    from .hf_models import get_bert_reader_components
    return get_bert_reader_components(args, **kwargs)


def init_pytext_bert_biencoder(args, **kwargs):
    if importlib.util.find_spec("pytext") is None:
        raise RuntimeError('Please install pytext lib')
    from .pytext_models import get_bert_biencoder_components
    return get_bert_biencoder_components(args, **kwargs)


def init_fairseq_roberta_biencoder(args, **kwargs):
    if importlib.util.find_spec("fairseq") is None:
        raise RuntimeError('Please install fairseq lib')
    from .fairseq_models import get_roberta_biencoder_components
    return get_roberta_biencoder_components(args, **kwargs)


def init_hf_bert_tenzorizer(args, **kwargs):
    if importlib.util.find_spec("transformers") is None:
        raise RuntimeError('Please install transformers lib')
    from .hf_models import get_bert_tensorizer
    return get_bert_tensorizer(args)


def init_hf_roberta_tenzorizer(args, **kwargs):
    if importlib.util.find_spec("transformers") is None:
        raise RuntimeError('Please install transformers lib')
    from .hf_models import get_roberta_tensorizer
    return get_roberta_tensorizer(args)


BIENCODER_INITIALIZERS = {
    'hf_bert': init_hf_bert_biencoder,
    'pytext_bert': init_pytext_bert_biencoder,
    'fairseq_roberta': init_fairseq_roberta_biencoder,
}

READER_INITIALIZERS = {
    'hf_bert': init_hf_bert_reader,
}

TENSORIZER_INITIALIZERS = {
    'hf_bert': init_hf_bert_tenzorizer,
    'hf_roberta': init_hf_roberta_tenzorizer,
    'pytext_bert': init_hf_bert_tenzorizer,  # using HF's code as of now
    'fairseq_roberta': init_hf_roberta_tenzorizer,  # using HF's code as of now
}


def init_comp(initializers_dict, type, args, **kwargs):
    if type in initializers_dict:
        return initializers_dict[type](args, **kwargs)
    else:
        raise RuntimeError('unsupported model type: {}'.format(type))


def init_biencoder_components(encoder_type: str, args, **kwargs):
    return init_comp(BIENCODER_INITIALIZERS, encoder_type, args, **kwargs)


def init_reader_components(encoder_type: str, args, **kwargs):
    return init_comp(READER_INITIALIZERS, encoder_type, args, **kwargs)


def init_tenzorizer(encoder_type: str, args, **kwargs):
    return init_comp(TENSORIZER_INITIALIZERS, encoder_type, args, **kwargs)
