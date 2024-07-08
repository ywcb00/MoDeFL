from model.DFLv1Model import DFLv1Model
from tffdataset.DatasetUtils import DatasetID, getDataset
from tffdataset.FedDataset import FedDataset, PartitioningScheme
from tffmodel.FedCoreModel import FedCoreModel
from tffmodel.FedApiModel import FedApiModel
from tffmodel.KerasModel import KerasModel
from tffmodel.ModelUtils import ModelUtils

import getopt
import logging
import sys
import tensorflow as tf

config = {
    "seed": 13,

    "force_load": False,

    "dataset_id": DatasetID.Mnist,

    "train_response_path": "./data/train_response.csv",
    "val_response_path": "./data/val_response.csv",
    "test_response_path": "./data/test_response.csv",

    "flooded_classes": tf.constant([
            1, # Building-flooded
            3, # Road-flooded
            5, # Water
            # 8, # Pool
        ], dtype=tf.uint32),
    "flooded_threshold": 1/4,

    "part_scheme": PartitioningScheme.ROUND_ROBIN,
    "num_workers": 4,

    "num_train_rounds": 10,

    "log_dir": "./log/training",
    "log_level": logging.DEBUG,
}

def trainLocalKeras(dataset, config):
    # ===== Local Training =====
    # create and fit the local keras model
    keras_model = KerasModel(config)
    keras_model.fit(dataset)

    # evaluate the model
    evaluation_metrics = keras_model.evaluate(dataset.val)
    return evaluation_metrics

def trainFedApi(dataset, fed_dataset, config):
    # ===== Federated Training =====
    # create and fit the federated model with tff api
    fed_api_model = FedApiModel(config)
    fed_api_model.fit(fed_dataset)

    # evaluate the model
    evaluation_metrics = fed_api_model.evaluate(fed_dataset.val)
    # evaluation_metrics = fed_api_model.evaluateCentralized(dataset.val)
    return evaluation_metrics

def trainFedCore(dataset, fed_dataset, config):
    # ===== Federated Training =====
    # create and fit the federated model with tff core
    fed_core_model = FedCoreModel(config)
    fed_core_model.fit(fed_dataset)

    # evaluate the model
    evaluation_metrics = fed_core_model.evaluate(fed_dataset.val)
    # evaluation_metrics = fed_core_model.evaluateCentralized(dataset.val)
    return evaluation_metrics

def trainDFLv1(dataset, fed_dataset, config):
    dflv1_model = DFLv1Model(config)
    dflv1_model.fit(fed_dataset)

def main(argv):
    logger = logging.getLogger("main.py")
    logger.setLevel(config["log_level"])

    try:
        opts, args = getopt.getopt(argv[1:], "hl", ["help", "forceload"])
    except getopt.GetoptError:
        print("Wrong usage.")
        print("Usage:", argv[0], "[--forceload]")
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print("Usage:", argv[0], "[--forceload]")
            sys.exit()
        elif opt in ("-l", "--forceload"):
            config["force_load"] = True

    # obtain the dataset (either load or compute the response labels)
    dataset = getDataset(config)
    dataset.load()

    # construct data partitions for federated execution
    fed_dataset = FedDataset(config)
    fed_dataset.construct(dataset)
    fed_dataset.batch()

    dataset.batch()

    # evaluations = dict()
    # evaluations["keras"] = trainLocalKeras(dataset, config)
    # evaluations["fedapi"] = trainFedApi(dataset, fed_dataset, config)
    # evaluations["fedcore"] = trainFedCore(dataset, fed_dataset, config)

    trainDFLv1(dataset, fed_dataset, config)

    # logger.info(ModelUtils.printEvaluations(evaluations, config))


if __name__ == '__main__':
    main(sys.argv)
