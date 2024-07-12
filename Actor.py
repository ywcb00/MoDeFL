
from model.LearningStrategy import LearningType
from model.SerializationUtils import SerializationUtils
from network.InitializationService import InitializationService
from tffdataset.DatasetUtils import DatasetID, getDataset
from tffdataset.FedDataset import FedDataset, PartitioningScheme
from tffmodel.KerasModel import KerasModel

import logging

class Actor:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("Actor")
        self.logger.setLevel(config["log_level"])

    def initialize(self):
        def initializeDatasetCallback(dataset_id, part_scheme_id, part_seed):
            self.config["dataset_id"] = DatasetID(dataset_id)
            self.config["part_scheme"] = PartitioningScheme(part_scheme_id)
            self.config["seed"] = part_seed

            self.dataset = getDataset(self.config)
            self.dataset.load()

            self.fed_dataset = FedDataset(self.config)
            self.fed_dataset.construct(self.dataset)
            self.fed_dataset.batch()

            self.logger.debug(f'Using dataset {self.config["dataset_id"].name}.')

        def initializeModelCallback(serialized_model):
            # TODO: support arbitrary serialized models from the initiator
            self.model = KerasModel(self.config)
            self.model.initModel(self.fed_dataset.train[0])
            self.logger.debug("Initialized the model.")

        def initializeModelWeightsCallback(weights_serialized):
            # deserialize and reshape the retrieved weights
            init_weights = SerializationUtils.deserializeModelWeights(
                weights_serialized, self.model.getWeights())
            self.model.setWeights(init_weights)
            self.logger.debug("Initialized the model weights.")

        def initializeLearningStrategyCallback(learning_type_id):
            self.config["learning_type"] = LearningType(learning_type_id)
            self.logger.debug(f'Using learning strategy {self.config["learning_type"].name}')

        callbacks = {"InitDataset": initializeDatasetCallback,
            "InitModel": initializeModelCallback,
            "InitModelWeights": initializeModelWeightsCallback,
            "InitLearningStrategy": initializeLearningStrategyCallback}

        init_service = InitializationService(self.config)
        init_service.waitForInitialization(callbacks)

        self.logger.info("Starting with the learning procedure")
