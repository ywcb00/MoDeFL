from model.DFLv1Strategy import DFLv1Strategy
from model.LearningStrategy import LearningStrategy, LearningType
from model.ModelUpdateMarket import SynchronizationStrategy
from network.Compression import CompressionType
from network.PartialDeviceParticipation import PartialDeviceParticipationStrategy
from model.SerializationUtils import SerializationUtils
from network.InitializationService import InitializationService
from tffdataset.DatasetUtils import DatasetID, getDataset
from tffdataset.DirectDataset import DirectDataset
from tffdataset.FedDataset import FedDataset, PartitioningScheme
from tffmodel.KerasModel import KerasModel

import logging
import tensorflow as tf

# represent a device and hold the main process in DFL with all related properties and functionalities
class Actor:
    def __init__(self, config):
        self.config = config
        self.config["log_dir"] = f'{self.config["log_dir"]}/{self.config["port"]}'
        self.logger = logging.getLogger("Actor")
        self.logger.setLevel(config["log_level"])

    # set the seed of all libraries used
    def setSeed(self):
        # random.seed(seed)
        # np.random.seed(seed)
        # tf.random.set_seed(seed)
        tf.keras.utils.set_random_seed(self.config["seed"])

    # start the initialization service, perform the initizalizations on request,
    #   and block until the start of the learning phase
    def initialize(self):
        def initializeIdentityCallback(addr, actor_idx, num_actors, seed):
            self.config["address"] = addr
            self.config["actor_idx"] = actor_idx
            self.config["num_workers"] = num_actors
            self.config["seed"] = seed

            self.logger.debug(f'Initialized own identity as {addr} with idx {actor_idx}/{num_actors}.')

        def initializeDatasetCallback(dataset_id, partitioning_scheme_id, partition_index,
            dataset_seed, partition_dirichlet_alpha):
            self.config["dataset_id"] = DatasetID(dataset_id)
            self.config["partitioning_scheme"] = PartitioningScheme(partitioning_scheme_id)
            self.config["partition_index"] = partition_index
            self.config["partitioning_alpha"] = partition_dirichlet_alpha

            self.dataset = getDataset(self.config)
            self.dataset.load(seed=dataset_seed)

            # TODO: distinguish between loading an entire dataset and partitioning it
            #       or directly loading a single partition
            self.fed_dataset = FedDataset(self.config)
            self.fed_dataset.construct(self.dataset, seed=dataset_seed)
            self.fed_dataset.batch()

            self.dataset = DirectDataset(self.dataset.batch_size, self.dataset.element_spec,
                self.fed_dataset.train[self.config["partition_index"]],
                self.fed_dataset.val[self.config["partition_index"]],
                self.fed_dataset.test[self.config["partition_index"]],
                self.config)

            self.logger.debug(f'Using partition {self.config["partition_index"]} of '
                + f'dataset {self.config["dataset_id"].name}.')

        def initializeModelCallback(model_config_serialized, optimizer_config_serialized):
            model, optimizer = SerializationUtils.deserializeModel(
                model_config_serialized, optimizer_config_serialized)
            self.keras_model = KerasModel.fromExistingModel(model, optimizer, self.config)

            self.logger.debug("Initialized the model.")

        def initializeModelParametersCallback(request):
            # deserialize and reshape the retrieved weights
            init_weights = SerializationUtils.deserializeParameters(
                request.parameters, sparse=request.sparse)
            self.keras_model.setWeights(init_weights)
            self.logger.debug("Initialized the model weights.")

        def initializeStrategyCallback(
            num_fed_epochs, num_local_epochs,
            learning_type_id, lr_local, lr_global,
            synchronization_strat_id, synchronization_strat_percentage,
            synchronization_strat_amount, synchronization_strat_timeout,
            synchronization_strat_allowempty,
            compression_strat_id, compression_strat_k,
            compression_strat_percentage, compression_strat_precision,
            pdp_strat_id, pdp_strat_k):
            self.config["num_fed_epochs"] = num_fed_epochs
            self.config["num_local_epochs"] = num_local_epochs
            self.config["learning_type"] = LearningType(learning_type_id)
            self.config["lr"] = lr_local if lr_local != 0 else None
            self.config["lr_global"] = lr_global if lr_global != 0 else None
            self.config["sync_strategy"] = SynchronizationStrategy(synchronization_strat_id)
            self.config["sync_strat_percentage"] = synchronization_strat_percentage
            self.config["sync_strat_amount"] = synchronization_strat_amount
            self.config["sync_strat_timeout"] = synchronization_strat_timeout
            self.config["sync_strat_allowempty"] = synchronization_strat_allowempty
            self.config["compression_type"] = CompressionType(compression_strat_id)
            self.config["compression_k"] = compression_strat_k
            self.config["compression_percentage"] = compression_strat_percentage
            self.config["compression_precision"] = compression_strat_precision
            self.config["pdp_strategy"] = PartialDeviceParticipationStrategy(pdp_strat_id)
            self.config["pdp_k"] = pdp_strat_k

            self.logger.debug(f'Using learning strategy {self.config["learning_type"].name}, ' +
                f'model update strategy {self.config["sync_strategy"].name}, ' +
                f'compression type {self.config["compression_type"].name}, ' +
                f'partial device participation strategy {self.config["pdp_strategy"].name}, ' +
                f'local learning rate {self.config["lr"] if self.config["lr"] != 0 else "DEFAULT"}, ' +
                f'and global learning rate {self.config["lr_global"] if self.config["lr_global"] != 0 else "DEFAULT"}.')

        def registerNeighborsCallback(neighbors_net_id):
            self.config["neighbors"] = list(neighbors_net_id.keys())
            self.config["neighbor_idx"] = list(neighbors_net_id.values())
            self.logger.debug(f'Registered {len(self.config["neighbors"])} neighbors.')

        callbacks = {"InitIdentity": initializeIdentityCallback,
            "InitDataset": initializeDatasetCallback,
            "InitModel": initializeModelCallback,
            "InitModelParameters": initializeModelParametersCallback,
            "InitStrategy": initializeStrategyCallback,
            "RegisterNeighbors": registerNeighborsCallback}

        init_service = InitializationService(self.config)
        init_service.waitForInitialization(callbacks)

        # set the seed for the random generators
        self.setSeed()

    # perform the training phase (and the evaluation phase)
    def train(self):
        self.logger.info("Starting with the learning procedure")

        self.strategy = LearningStrategy.getStrategy(self.config, self.keras_model, self.dataset)
        self.strategy.performTraining()

    # run the actor
    def run(self):
        self.initialize()
        self.train()
