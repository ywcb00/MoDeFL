
from model.SerializationUtils import SerializationUtils
from network.NetworkUtils import NetworkUtils
import network.protos.Initialization_pb2 as Initialization_pb2
import network.protos.Initialization_pb2_grpc as Initialization_pb2_grpc
import network.protos.ModelUpdate_pb2 as ModelUpdate_pb2
from tffdataset.DatasetUtils import getDatasetElementSpec
from tffmodel.KerasModel import KerasModel

import asyncio
import grpc
import itertools
import logging
import numpy as np

class Initiator:
    actor_idx = itertools.count()

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("Initiator")
        self.logger.setLevel(config["log_level"])

    async def initializeActor(self, addr):
        self.logger.debug(f'Connecting to {addr}')
        async with grpc.aio.insecure_channel(addr) as channel:
            stub = Initialization_pb2_grpc.InitializeStub(channel)

            await stub.InitIdentity(Initialization_pb2.Identity(ip_and_port=addr))

            await stub.InitDataset(Initialization_pb2.Dataset(
                dataset_id=self.config["dataset_id"].value,
                partition_scheme_id=self.config["part_scheme"].value,
                partition_index=next(self.actor_idx),
                seed=self.config["seed"]))

            await stub.InitModel(Initialization_pb2.Model(model=None))

            init_weights = KerasModel.createKerasModelElementSpec(
                getDatasetElementSpec(self.config), self.config).get_weights()
            init_weights_serialized = SerializationUtils.serializeModelWeights(init_weights)
            await stub.InitModelWeights(ModelUpdate_pb2.ModelWeights(
                layer_weights=init_weights_serialized, ip_and_port=addr))

            await stub.InitLearningStrategy(Initialization_pb2.LearningStrategy(
                learning_type_id=self.config["learning_type"].value))
        self.logger.debug(f'Initialized {addr}')
        return

    async def registerNeighbors(self, addr, addresses):
        addresses = [a for a in addresses if a != addr]
        self.logger.debug(f'Connecting to {addr}')
        async with grpc.aio.insecure_channel(addr) as channel:
            stub = Initialization_pb2_grpc.InitializeStub(channel)
            await stub.RegisterNeighbors(Initialization_pb2.NeighborSpec(ip_and_port=addresses))
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
        tasks = []
        for addr in addresses:
            tasks.append(asyncio.create_task(self.initializeActor(addr)))
            neighbor_addresses = NetworkUtils.getNeighborAddresses(addr, addresses, adj_mat)
            tasks.append(asyncio.create_task(self.registerNeighbors(addr, neighbor_addresses)))
        for t in tasks:
            await t

    async def startLearning(self, addresses):
        tasks = []
        for addr in addresses:
            tasks.append(asyncio.create_task(self.startActorLearning(addr)))
        for t in tasks:
            await t

    def initiate(self):
        actor_addresses = [addr.strip() for addr in open(self.config["address_file"])]
        actor_adjacency = np.fromfile(self.config["adjacency_file"], dtype=int, sep=" ")
        actor_adjacency = np.reshape(actor_adjacency,
            newshape=(len(actor_addresses), len(actor_addresses)))
        asyncio.run(self.initialize(actor_addresses, actor_adjacency))
        asyncio.run(self.startLearning(actor_addresses))
        self.logger.info("Initiation completed.")
