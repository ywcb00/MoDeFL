from model.DFLv1Strategy import DFLv1Strategy
from model.LearningStrategy import LearningStrategy, LearningType
from model.ModelUpdateMarket import SynchronizationStrategy
from model.SerializationUtils import SerializationUtils
from network.InitializationService import InitializationService
from tffdataset.DatasetUtils import DatasetID, getDataset
from tffdataset.DirectDataset import DirectDataset
from tffdataset.FedDataset import FedDataset, PartitioningScheme
from tffmodel.KerasModel import KerasModel

import logging
import tensorflow as tf

class Actor:
    def __init__(self, config):
        self.config = config
        self.config["log_dir"] = f'{self.config["log_dir"]}/{self.config["port"]}'
        self.logger = logging.getLogger("Actor")
        self.logger.setLevel(config["log_level"])

    def initialize(self):
        def initializeIdentityCallback(addr, actor_idx, num_actors):
            self.config["address"] = addr
            self.config["actor_idx"] = actor_idx
            self.config["num_workers"] = num_actors

            self.logger.debug(f'Initialized own identity as {addr} with idx {actor_idx}/{num_actors}.')

        def initializeDatasetCallback(dataset_id, partitioning_scheme_id, part_index, seed):
            self.config["dataset_id"] = DatasetID(dataset_id)
            self.config["partitioning_scheme"] = PartitioningScheme(partitioning_scheme_id)
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

        def initializeModelCallback(model_config_serialized, optimizer_config_serialized):
            model, optimizer = SerializationUtils.deserializeModel(
                model_config_serialized, optimizer_config_serialized)
            self.keras_model = KerasModel.fromExistingModel(model, optimizer, self.config)

            self.logger.debug("Initialized the model.")

        def initializeModelWeightsCallback(request):
            # deserialize and reshape the retrieved weights
            init_weights = SerializationUtils.deserializeParameters(
                request.parameters, sparse=request.sparse)
            self.keras_model.setWeights(init_weights)
            self.logger.debug("Initialized the model weights.")

        def initializeLearningStrategyCallback(learning_type_id,
            synchronization_strategy_id, synchronization_strat_percentage,
            synchronization_strat_amount, synchronization_strat_timeout):
            self.config["learning_type"] = LearningType(learning_type_id)
            self.config["synchronization_strategy"] = SynchronizationStrategy(synchronization_strategy_id)
            self.config["synchronization_strat_percentage"] = synchronization_strat_percentage
            self.config["synchronization_strat_amount"] = synchronization_strat_amount
            self.config["synchronization_strat_timeout"] = synchronization_strat_timeout

            self.logger.debug(f'Using learning strategy {self.config["learning_type"].name} ' +
                f'and model update strategy {self.config["synchronization_strategy"].name}')

        def registerNeighborsCallback(neighbors_net_id):
            self.config["neighbors"] = list(neighbors_net_id.keys())
            self.config["neighbor_idx"] = list(neighbors_net_id.values())
            self.logger.debug(f'Registered {len(self.config["neighbors"])} neighbors.')

        callbacks = {"InitIdentity": initializeIdentityCallback,
            "InitDataset": initializeDatasetCallback,
            "InitModel": initializeModelCallback,
            "InitModelWeights": initializeModelWeightsCallback,
            "InitLearningStrategy": initializeLearningStrategyCallback,
            "RegisterNeighbors": registerNeighborsCallback}

        init_service = InitializationService(self.config)
        init_service.waitForInitialization(callbacks)

    def train(self):
        self.logger.info("Starting with the learning procedure")

        self.strategy = LearningStrategy.getStrategy(self.config, self.keras_model, self.dataset)
        self.strategy.performTraining()

    def run(self):
        self.initialize()
        self.train()
