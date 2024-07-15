
from model.SerializationUtils import SerializationUtils
import network.protos.Initialization_pb2 as Initialization_pb2
import network.protos.Initialization_pb2_grpc as Initialization_pb2_grpc
import network.protos.ModelUpdate_pb2 as ModelUpdate_pb2
from tffdataset.DatasetUtils import getDatasetElementSpec
from tffmodel.KerasModel import KerasModel

import asyncio
import grpc
import logging

class Initiator:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("Initiator")
        self.logger.setLevel(config["log_level"])

    async def initializeActor(self, addr):
        self.logger.debug(f'Connecting to {addr}')
        async with grpc.aio.insecure_channel(addr) as channel:
            stub = Initialization_pb2_grpc.InitializeStub(channel)

            await stub.InitDataset(Initialization_pb2.Dataset(
                dataset_id=self.config["dataset_id"].value,
                partition_scheme_id=self.config["part_scheme"].value,
                partition_seed=self.config["seed"]))

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

    async def initialize(self):
        address_file = open(self.config["address_file"])
        tasks = []
        for addr in address_file:
            tasks.append(asyncio.create_task(self.initializeActor(addr)))
        for t in tasks:
            await t

    async def startLearning(self):
        address_file = open(self.config["address_file"])
        tasks = []
        for addr in address_file:
            tasks.append(asyncio.create_task(self.startActorLearning(addr)))
        for t in tasks:
            await t

    def initiate(self):
        asyncio.run(self.initialize())
        asyncio.run(self.startLearning())
        self.logger.info("Initiation completed.")
