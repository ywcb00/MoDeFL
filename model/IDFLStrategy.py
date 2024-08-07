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

    @abstractmethod
    def startServer(self):
        pass

    @abstractmethod
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

    async def broadcastWeightsAndGradientTo(self, weights_serialized, gradient_serialized, address):
        async with grpc.aio.insecure_channel(address) as channel:
            stub = ModelUpdate_pb2_grpc.ModelUpdateStub(channel)
            await stub.TransferModelUpdate(ModelUpdate_pb2.ModelWeights(
                layer_weights=weights_serialized,
                gradient=ModelUpdate_pb2.ModelGradient(gradient=gradient_serialized),
                ip_and_port=self.config["address"]))

    async def broadcastWeightsAndGradientsToNeighbors(self, weights_serialized, gradients_serialized):
        tasks = []
        for addr in self.config["neighbors"]:
            tasks.append(asyncio.create_task(self.broadcastWeightsAndGradientTo(
                weights_serialized, gradients_serialized[addr], addr)))
        await asyncio.wait(tasks, return_when=asyncio.ALL_COMPLETED)

    @abstractmethod
    def broadcast(self):
        pass

    @abstractmethod
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

    @abstractmethod
    def evaluate(self):
        pass

    def registerTerminationPermission(self, address):
        self.termination_permission[address] = True
        if(all(self.termination_permission.values())):
            self.model_update_service.stopServer()

    async def signalTerminationPermissionTo(self, address):
        async with grpc.aio.insecure_channel(address) as channel:
            stub = ModelUpdate_pb2_grpc.ModelUpdateStub(channel)
            await stub.AllowTermination(ModelUpdate_pb2.NetworkID(
                ip_and_port=self.config["address"]))

    async def signalTerminationPermission(self):
        tasks = []
        for addr in self.config["neighbors"]:
            tasks.append(asyncio.create_task(self.signalTerminationPermissionTo(addr)))
        await asyncio.wait(tasks, return_when=asyncio.ALL_COMPLETED)

    @abstractmethod
    def stop(self):
        pass

    def performTraining(self):
        self.startServer()

        # TODO: think about the number of epochs for learning (perhaps termination based on local training loss?)
        for epoch in range(5):
            self.logger.debug(f'Federated epoch #{epoch}')
            self.fitLocal()
            self.broadcast()
            self.aggregate()

        eval_avg = self.evaluate()
        self.logger.info(f'Evaluation with neighbors resulted in an average of {eval_avg}')

        self.stop()
