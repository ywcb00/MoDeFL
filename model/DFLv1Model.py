from model.IDFLModel import IDFLModel
from model.SerializationUtils import SerializationUtils
from network.ModelUpdateService import ModelUpdateService
import network.protos.ModelUpdate_pb2 as ModelUpdate_pb2
import network.protos.ModelUpdate_pb2_grpc as ModelUpdate_pb2_grpc
from tffmodel.ModelUtils import ModelUtils

import asyncio
import grpc
import logging

# TODO: rename to DFLv1Strategy
class DFLv1Model(IDFLModel):
    def __init__(self, config, keras_model):
        super().__init__(config, keras_model)
        self.logger = logging.getLogger("model/DFLv1Model")
        self.logger.setLevel(config["log_level"])

    def startServer(self):
        def transferModelUpdateCallback(weights_serialized, address):
            weights = SerializationUtils.deserializeModelWeights(
                weights_serialized, self.keras_model.getWeights())
            self.model_update_market.put(weights, address)

        callbacks = {"TransferModelUpdate": transferModelUpdateCallback}

        self.model_update_service = ModelUpdateService(self.config)
        self.model_update_service.startServer(callbacks)

    def stopServer(self):
        self.model_update_service.stopServer()

    def fitLocal(self, dataset):
        self.logger.info("Fitting local model.")
        # TODO: change number of epochs for fit (to 1?)
        self.keras_model.fit(dataset)

    async def broadcastTo(self, weights_serialized, address):
        async with grpc.aio.insecure_channel(address) as channel:
            stub = ModelUpdate_pb2_grpc.ModelUpdateStub(channel)
            await stub.TransferModelUpdate(ModelUpdate_pb2.ModelWeights(
                layer_weights=weights_serialized,
                ip_and_port=self.config["address"]))

    async def broadcastToNeighbors(self, weights_serialized):
        tasks = []
        for addr in self.config["neighbors"]:
            tasks.append(asyncio.create_task(self.broadcastTo(weights_serialized, addr)))
        for t in tasks:
            await t

    def broadcast(self):
        weights = self.keras_model.getWeights()
        weights_serialized = SerializationUtils.serializeModelWeights(weights)

        asyncio.run(self.broadcastToNeighbors(weights_serialized))

    def aggregate(self):
        model_updates = self.model_update_market.getFromAll()
        avg_model_update = ModelUtils.averageModelWeights(list(model_updates.values()))
        self.keras_model.setWeights(avg_model_update)

    def performTraining(self, dataset):
        self.startServer()

        for counter in range(10):
            self.fitLocal(dataset)
            self.broadcast()
            self.aggregate()

        self.stopServer()
