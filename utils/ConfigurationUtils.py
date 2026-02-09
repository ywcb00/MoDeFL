from model.LearningStrategy import LearningType
from model.ModelUpdateMarket import SynchronizationStrategy
from network.PartialDeviceParticipation import PartialDeviceParticipationStrategy
from tffdataset.DatasetUtils import DatasetID
from tffdataset.FedDataset import PartitioningScheme
from network.Compression import CompressionType
from network.NetworkUtils import NetworkServiceType
from utils.PartitioningUtils import ModelPartitioningStrategy

import json
import logging
import os

# define possible configuration options, convert the options to the correct format, and provide default configurations
class ConfigurationUtils:
    DEFAULT_CONFIG = {
        "seed": 13,

        "dataset_id": DatasetID.Mnist,

        "partitioning_scheme": PartitioningScheme.ROUND_ROBIN,
        "partitioning_alpha": 2.5, # argument for Dirichlet partitioning
        "model_partitioning_strategy": ModelPartitioningStrategy.LAYERWISE,

        # the number of workers is initialized by the initiator based on the address file
        # "num_workers": 4, # i.e., number of actors in DFL
        "num_fed_epochs": 5,
        "num_local_epochs": 1,

        "addr_file": "./addr.txt",
        "adj_file": "./adj.txt",

        "num_threads_server": os.cpu_count(),

        "learning_type": LearningType.DFLv1,

        "networkservice_type": NetworkServiceType.GRPC,

        "sync_strategy": SynchronizationStrategy.ONE_FROM_EACH,
        "sync_strat_percentage": 0.5,
        "sync_strat_amount": 2,
        "sync_strat_timeout": 3,
        "sync_strat_allowempty": False,

        "compression_type": CompressionType.NoneType,
        "compression_k": 100,
        "compression_percentage": 0.2,
        "compression_precision": 8,

        "pdp_strategy": PartialDeviceParticipationStrategy.NoneStrategy,
        "pdp_k": 2,

        "log_tensorboard_flag": False,
        "log_performance_flag": True,
        "log_communication_flag": True,
        "log_dir": "./log",
        "log_level": logging.DEBUG,
    }

    OPTIONAL_CONFIGS = ["lr", "lr_global"]

    CLI_OPTIONS = [*[ck + "=" for ck in DEFAULT_CONFIG.keys()],
        *[oc + "=" for oc in OPTIONAL_CONFIGS]]

    # load a configuration from the specifed json file
    @classmethod
    def loadConfig(self_class, config_path):
        with open(config_path) as cf:
            config_dict = json.loads(cf.read())
        return config_dict

    # translate the option name from CLI to the name used in the configuration dictionary
    @classmethod
    def parseCLIOption(self_class, config, opt, arg):
        if(opt == "-p"):
            opt = "--port"
        config[opt.strip('-')] = arg
        return config

    # convert the configuration options to the proper data types
    @classmethod
    def convertConfigTypes(self_class, config):
        # convert enum options
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
        config["partitioning_scheme"] = convertEnum(config["partitioning_scheme"], PartitioningScheme)
        config["model_partitioning_strategy"] = convertEnum(config["model_partitioning_strategy"], ModelPartitioningStrategy)
        config["learning_type"] = convertEnum(config["learning_type"], LearningType)
        config["networkservice_type"] = convertEnum(config["networkservice_type"], NetworkServiceType)
        config["sync_strategy"] = convertEnum(config["sync_strategy"], SynchronizationStrategy)
        config["compression_type"] = convertEnum(config["compression_type"], CompressionType)
        config["pdp_strategy"] = convertEnum(config["pdp_strategy"],
            PartialDeviceParticipationStrategy)

        # convert boolean options
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
        bool_type_configs = ["sync_strat_allowempty", "log_tensorboard_flag",
            "log_performance_flag", "log_communication_flag"]
        for btc in bool_type_configs:
            if(btc in config.keys()):
                config[btc] = convertBool(config[btc])

        # convert integer options
        def convertInt(value):
            if(isinstance(value, str)):
                value = int(value)
            elif(isinstance(value, int)):
                pass # value has already int type
            else:
                raise RuntimeError(f'Cannot convert type {type(value)} to int.')
            return value
        int_type_configs = ["seed", "num_threads_server",
            "num_fed_epochs", "num_local_epochs", "sync_strat_amount",
            "compression_k", "compression_precision", "pdp_k", "log_level"]
        for itc in int_type_configs:
            if(itc in config.keys()):
                config[itc] = convertInt(config[itc])

        # convert float options
        def convertFloat(value):
            if(isinstance(value, str) or isinstance(value, int)):
                value = float(value)
            elif(isinstance(value, float)):
                pass # value has already float type
            else:
                raise RuntimeError(f'Cannot convert type {type(value)} to float.')
            return value
        float_type_configs = ["partitioning_alpha",
            "sync_strat_percentage", "sync_strat_timeout",
            "compression_percentage", "lr", "lr_global"]
        for ftc in float_type_configs:
            if(ftc in config.keys()):
                config[ftc] = convertFloat(config[ftc])

        return config
