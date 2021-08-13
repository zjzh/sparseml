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

import logging

import hydra
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class BiencoderDatasetsCfg(object):
    def __init__(self, cfg: DictConfig):
        datasets = cfg.datasets
        self.train_datasets_names = cfg.train_datasets
        logger.info("train_datasets: %s", self.train_datasets_names)
        if self.train_datasets_names:
            self.train_datasets = [
                hydra.utils.instantiate(datasets[ds_name])
                for ds_name in self.train_datasets_names
            ]
        else:
            self.train_datasets = []
        if cfg.dev_datasets:
            self.dev_datasets_names = cfg.dev_datasets
            logger.info("dev_datasets: %s", self.dev_datasets_names)
            self.dev_datasets = [
                hydra.utils.instantiate(datasets[ds_name])
                for ds_name in self.dev_datasets_names
            ]
        self.sampling_rates = cfg.train_sampling_rates
