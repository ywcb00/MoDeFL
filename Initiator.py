
from model.LearningStrategy import LearningType
from model.SerializationUtils import SerializationUtils
from network.NetworkUtils import NetworkUtils
import network.protos.Initialization_pb2 as Initialization_pb2
import network.protos.Initialization_pb2_grpc as Initialization_pb2_grpc
import network.protos.ModelUpdate_pb2 as ModelUpdate_pb2
from tffdataset.DatasetUtils import getDatasetElementSpec
from tffmodel.KerasModel import KerasModel
from tffmodel.ModelBuilderUtils import getFedLearningRateSchedules, getModelBuilder
from tffmodel.types.Weights import Weights

import asyncio
import grpc
import itertools
import logging
import numpy as np
import tensorflow as tf

class Initiator:
    actor_idx = itertools.count()

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("Initiator")
        self.logger.setLevel(config["log_level"])

    async def initializeActor(self, addr, actor_idx, num_actors, model_config_serialized,
        optimizer_config_serialized, init_weights_serialized):
        self.logger.debug(f'Connecting to {addr}')
        async with grpc.aio.insecure_channel(addr) as channel:
            stub = Initialization_pb2_grpc.InitializeStub(channel)

            await stub.InitIdentity(Initialization_pb2.Identity(
                net_id=ModelUpdate_pb2.NetworkIdentity(
                    ip_and_port=addr, actor_idx=actor_idx),
                num_workers=num_actors))

            await stub.InitDataset(Initialization_pb2.Dataset(
                dataset_id=self.config["dataset_id"].value,
                partition_scheme_id=self.config["part_scheme"].value,
                partition_index=next(self.actor_idx),
                seed=self.config["seed"]))

            await stub.InitModel(Initialization_pb2.Model(
                model_config=model_config_serialized, optimizer_config=optimizer_config_serialized))

            await stub.InitModelWeights(ModelUpdate_pb2.ModelWeights(
                weights=init_weights_serialized))

            await stub.InitLearningStrategy(Initialization_pb2.LearningStrategy(
                learning_type_id=self.config["learning_type"].value,
                model_update_spec=Initialization_pb2.ModelUpdateSpec(
                    synchronization_strategy_id=self.config["synchronization_strategy"].value,
                    synchronization_strat_percentage=self.config.setdefault("synchronization_strat_percentage", 0.0),
                    synchronization_strat_amount=self.config.setdefault("synchronization_strat_amount", 0),
                    synchronization_strat_timeout=self.config.setdefault("synchronization_strat_timeout", 0.0))))
        self.logger.debug(f'Initialized {addr}')
        return

    async def registerNeighbors(self, addr, identities):
        self.logger.debug(f'Connecting to {addr}')
        async with grpc.aio.insecure_channel(addr) as channel:
            stub = Initialization_pb2_grpc.InitializeStub(channel)
            await stub.RegisterNeighbors(
                Initialization_pb2.NeighborSpec(net_id=[ModelUpdate_pb2.NetworkIdentity(
                    ip_and_port=addr, actor_idx=aidx) for aidx, addr in identities.items()]))
        self.logger.debug(f'Registered neighbors of {addr}')
        return

    async def startActorLearning(self, addr):
        self.logger.debug(f'Connecting to {addr}')
        async with grpc.aio.insecure_channel(addr) as channel:
            stub = Initialization_pb2_grpc.InitializeStub(channel)
            try:
                await stub.StartLearning(ModelUpdate_pb2.Ack())
            except grpc.aio._call.AioRpcError:
                # connection gets lost during the call because we stop the initiation server on the actor
                pass
        self.logger.debug(f'Started learning on {addr}')

    async def initialize(self, addresses, adj_mat):
        model = KerasModel.createKerasModelElementSpec(
            getDatasetElementSpec(self.config), self.config)
        _, local_lr = getFedLearningRateSchedules(self.config)
        actor_optimizer = tf.keras.optimizers.SGD(learning_rate=local_lr)
        model_config_serialized, optimizer_config_serialized = SerializationUtils.serializeModel(
            model, actor_optimizer)

        init_weights = Weights(model.get_weights())
        init_weights_serialized = SerializationUtils.serializeModelWeights(init_weights)

        tasks = []
        for actor_idx, addr in enumerate(addresses):
            tasks.append(asyncio.create_task(self.initializeActor(addr, actor_idx, len(addresses),
                model_config_serialized, optimizer_config_serialized, init_weights_serialized)))
            neighbor_identities = NetworkUtils.getNeighborIdentities(addr, addresses, adj_mat)
            assert (not self.config["learning_type"] in
                        [LearningType.DFLv1, LearningType.DFLv4, LearningType.DFLv5, LearningType.DFLv6] or
                    len(neighbor_identities)+1 == self.config["num_workers"]
                ), "DFLv1 requires a fully connected actor network."
            tasks.append(asyncio.create_task(self.registerNeighbors(addr, neighbor_identities)))
        await asyncio.wait(tasks, return_when=asyncio.ALL_COMPLETED)

    async def startLearning(self, addresses):
        tasks = []
        for addr in addresses:
            tasks.append(asyncio.create_task(self.startActorLearning(addr)))
        await asyncio.wait(tasks, return_when=asyncio.ALL_COMPLETED)

    def initiate(self):
        actor_addresses = [addr.strip() for addr in open(self.config["addr_file"])]
        self.config["num_workers"] = len(actor_addresses)
        actor_adjacency = np.fromfile(self.config["adj_file"], dtype=int, sep=" ")
        actor_adjacency = np.reshape(actor_adjacency,
            newshape=(self.config["num_workers"], self.config["num_workers"]))
        asyncio.run(self.initialize(actor_addresses, actor_adjacency))
        asyncio.run(self.startLearning(actor_addresses))
        self.logger.info("Initiation completed.")
