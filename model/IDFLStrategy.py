from model.ModelUpdateMarket import ModelUpdateMarket
import network.protos.ModelUpdate_pb2 as ModelUpdate_pb2
import network.protos.ModelUpdate_pb2_grpc as ModelUpdate_pb2_grpc

from abc import ABC, abstractmethod
import asyncio
import grpc

class IDFLStrategy(ABC):
    def __init__(self, config, keras_model, dataset):
        self.config = config
        self.keras_model = keras_model
        self.model_update_market = ModelUpdateMarket(self.config)
        self.dataset = dataset

    def startServer(self):
        pass

    def stopServer(self):
        pass

    def fitLocal(self):
        pass

    async def broadcastWeightsTo(self, weights_serialized, address):
        async with grpc.aio.insecure_channel(address) as channel:
            stub = ModelUpdate_pb2_grpc.ModelUpdateStub(channel)
            await stub.TransferModelUpdate(ModelUpdate_pb2.ModelWeights(
                layer_weights=weights_serialized,
                ip_and_port=self.config["address"]))

    async def broadcastWeightsToNeighbors(self, weights_serialized):
        tasks = []
        for addr in self.config["neighbors"]:
            tasks.append(asyncio.create_task(self.broadcastWeightsTo(weights_serialized, addr)))
        await asyncio.wait(tasks, return_when=asyncio.ALL_COMPLETED)

    def broadcast(self):
        pass

    def aggregate(self):
        pass

    # obtain evaluation metrics from our own model evaluated on the neighbors' evaluation data
    async def evaluateWeightsNeighbor(self, weights_serialized, address):
        async with grpc.aio.insecure_channel(address) as channel:
            stub = ModelUpdate_pb2_grpc.ModelUpdateStub(channel)
            eval_metrics = await stub.EvaluateModel(ModelUpdate_pb2.ModelWeights(
                layer_weights=weights_serialized,
                ip_and_port=self.config["address"]))
        return eval_metrics.metrics

    async def evaluateWeightsAllNeighbors(self, weights_serialized):
        tasks = []
        for addr in self.config["neighbors"]:
            tasks.append(asyncio.create_task(self.evaluateWeightsNeighbor(weights_serialized, addr)))
        eval_metrics = []
        for t in tasks:
            response = await t
            eval_metrics.append(dict([(elem.key, elem.value) for elem in response]))
        return eval_metrics

    def evaluate(self):
        pass

    def performTraining(self):
        pass
