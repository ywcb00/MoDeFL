from model.LearningStrategy import LearningType
from model.ModelUpdateMarket import ModelUpdateStrategy
from tffdataset.DatasetUtils import DatasetID
from tffdataset.FedDataset import PartitioningScheme
from tffmodel.types.SparseGradient import SparsificationType

import json
import logging
import os

class ConfigurationUtils:
    DEFAULT_CONFIG = {
        "seed": 13,

        "dataset_id": DatasetID.Mnist,

        "part_scheme": PartitioningScheme.ROUND_ROBIN,
        "num_workers": 4, # i.e., number of actors in DFL

        "addr_file": "./resources/actor_addresses.txt",
        "adj_file": "./resources/actor_adjacency.txt",

        "num_threads_server": os.cpu_count(),

        "learning_type": LearningType.DFLv3,
        "model_update_strategy": ModelUpdateStrategy.ONE_FROM_ALL,
        "model_update_strat_percentage": 0.5,
        "model_update_strat_amount": 2,
        "model_update_strat_timeout": 3,

        "sparsification_type": SparsificationType.LAYERWISE_TOPK,
        "sparse_k": 100,
        "sparse_perc": 0.2,

        "num_fed_epochs": 5,
        "num_epochs": 1, # TODO: FIXME: this number corresponds to the local training rounds at the moment

        "tensorboard_logging": False,
        "performance_logging": True,
        "log_dir": "./log",
        "log_level": logging.DEBUG,
    }

    OPTIONAL_CONFIGS = ["lr", "lr_server", "lr_client"]

    CLI_OPTIONS = [*[ck + "=" for ck in DEFAULT_CONFIG.keys()],
        *[oc + "=" for oc in OPTIONAL_CONFIGS]]

    @classmethod
    def loadConfig(self_class, config_path):
        with open(config_path) as cf:
            config_dict = json.loads(cf.read())
        assert (not "adj_file" in config_dict.keys() and not "addr_file" in config_dict.keys()
            ), "Specifying the address file and adjacency file through the config file is not working."
        return config_dict

    @classmethod
    def parseCLIOption(self_class, config, opt, arg):
        if(opt == "-p"):
            opt = "--port"
        config[opt.strip('-')] = arg
        return config

    @classmethod
    def convertConfigTypes(self_class, config):
        def convertEnum(value, enum_class):
            if(isinstance(value, str)):
                value = enum_class((int(value)))
            elif(isinstance(value, int)):
                value = enum_class(value)
            elif(isinstance(value, enum_class)):
                pass # value has already the correct type
            else:
                raise RuntimeError(f'Cannot convert type {type(value)} to enum {enum_class.__name__}.')
            return value
        config["dataset_id"] = convertEnum(config["dataset_id"], DatasetID)
        config["part_scheme"] = convertEnum(config["part_scheme"], PartitioningScheme)
        config["learning_type"] = convertEnum(config["learning_type"], LearningType)
        config["model_update_strategy"] = convertEnum(config["model_update_strategy"], ModelUpdateStrategy)
        config["sparsification_type"] = convertEnum(config["sparsification_type"], SparsificationType)

        def convertBool(value):
            if(isinstance(value, str)):
                value = value.lower().capitalize() in ("True", "1", "T")
            elif(isinstance(value, int)):
                value = value != 0
            elif(isinstance(value, bool)):
                pass # value has already bool type
            else:
                raise RuntimeError(f'Cannot convert type {type(value)} to bool.')
            return value
        bool_type_configs = ["tensorboard_logging", "performance_logging"]
        for btc in bool_type_configs:
            if(btc in config.keys()):
                config[btc] = convertBool(config[btc])

        def convertInt(value):
            if(isinstance(value, str)):
                value = int(value)
            elif(isinstance(value, int)):
                pass # value has already int type
            else:
                raise RuntimeError(f'Cannot convert type {type(value)} to int.')
            return value
        int_type_configs = ["seed", "num_workers", "num_threads_server",
            "model_update_strat_amount", "num_fed_epochs", "num_epochs", "log_level"]
        for itc in int_type_configs:
            if(itc in config.keys()):
                config[itc] = convertInt(config[itc])

        def convertFloat(value):
            if(isinstance(value, str) or isinstance(value, int)):
                value = float(value)
            elif(isinstance(value, float)):
                pass # value has already float type
            else:
                raise RuntimeError(f'Cannot convert type {type(value)} to float.')
            return value
        float_type_configs = ["model_update_strat_percentage", "model_update_strat_timeout",
            "lr", "lr_server", "lr_client"]
        for ftc in float_type_configs:
            if(ftc in config.keys()):
                config[ftc] = convertFloat(config[ftc])

        return config
