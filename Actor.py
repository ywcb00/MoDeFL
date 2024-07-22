from model.DFLv1Strategy import DFLv1Strategy
from model.LearningStrategy import LearningType
from model.SerializationUtils import SerializationUtils
from network.InitializationService import InitializationService
from tffdataset.DatasetUtils import DatasetID, getDataset
from tffdataset.DirectDataset import DirectDataset
from tffdataset.FedDataset import FedDataset, PartitioningScheme
from tffmodel.KerasModel import KerasModel

import logging

class Actor:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("Actor")
        self.logger.setLevel(config["log_level"])

    def initialize(self):
        def initializeIdentityCallback(addr):
            self.config["address"] = addr

            self.logger.debug(f'Initialized own identity as {addr}.')

        def initializeDatasetCallback(dataset_id, part_scheme_id, part_index, seed):
            self.config["dataset_id"] = DatasetID(dataset_id)
            self.config["part_scheme"] = PartitioningScheme(part_scheme_id)
            self.config["part_index"] = part_index
            self.config["seed"] = seed

            self.dataset = getDataset(self.config)
            self.dataset.load()

            # TODO: distinguish between loading an entire dataset and partitioning it
            #       or directly loading a single partition
            self.fed_dataset = FedDataset(self.config)
            self.fed_dataset.construct(self.dataset)
            self.fed_dataset.batch()

            self.dataset = DirectDataset(self.dataset.batch_size, self.dataset.element_spec,
                self.fed_dataset.train[self.config["part_index"]],
                self.fed_dataset.val[self.config["part_index"]],
                self.fed_dataset.test[self.config["part_index"]],
                self.config)

            self.logger.debug(f'Using partition {self.config["part_index"]} of '
                + f'dataset {self.config["dataset_id"].name}.')

        def initializeModelCallback(serialized_model):
            # TODO: support arbitrary serialized models from the initiator
            self.keras_model = KerasModel(self.config)
            self.keras_model.initModel(self.fed_dataset.train[0])
            self.logger.debug("Initialized the model.")

        def initializeModelWeightsCallback(weights_serialized):
            # deserialize and reshape the retrieved weights
            init_weights = SerializationUtils.deserializeModelWeights(
                weights_serialized, self.keras_model.getWeights())
            self.keras_model.setWeights(init_weights)
            self.logger.debug("Initialized the model weights.")

        def initializeLearningStrategyCallback(learning_type_id):
            self.config["learning_type"] = LearningType(learning_type_id)
            self.logger.debug(f'Using learning strategy {self.config["learning_type"].name}')

        def registerNeighborsCallback(neighbors_ip_and_port):
            self.config["neighbors"] = neighbors_ip_and_port
            self.logger.debug(f'Registered {len(self.config["neighbors"])} neighbors')

        callbacks = {"InitIdentity": initializeIdentityCallback,
            "InitDataset": initializeDatasetCallback,
            "InitModel": initializeModelCallback,
            "InitModelWeights": initializeModelWeightsCallback,
            "InitLearningStrategy": initializeLearningStrategyCallback,
            "RegisterNeighbors": registerNeighborsCallback}

        init_service = InitializationService(self.config)
        init_service.waitForInitialization(callbacks)

        self.logger.info("Starting with the learning procedure")

        self.model = DFLv1Strategy(self.config, self.keras_model)
        self.model.performTraining(self.dataset)
