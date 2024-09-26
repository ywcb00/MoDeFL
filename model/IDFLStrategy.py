from model.ModelUpdateMarket import ModelUpdateMarket
from model.SerializationUtils import SerializationUtils
import network.protos.ModelUpdate_pb2 as ModelUpdate_pb2
import network.protos.ModelUpdate_pb2_grpc as ModelUpdate_pb2_grpc
from tffmodel.KerasModel import KerasModel
from utils.PerformanceLogger import PerformanceLogger

from abc import ABC, abstractmethod
import asyncio
import grpc
import numpy as np

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

    async def broadcastWeightsTo(self, weights_serialized, address, aggregation_weight=0):
        async with grpc.aio.insecure_channel(address) as channel:
            stub = ModelUpdate_pb2_grpc.ModelUpdateStub(channel)
            await stub.TransferModelUpdate(ModelUpdate_pb2.ModelUpdateMessage(
                weights=ModelUpdate_pb2.ModelWeights(weights=weights_serialized),
                aggregation_weight=aggregation_weight,
                identity=ModelUpdate_pb2.NetworkIdentity(ip_and_port=self.config["address"])))

    async def broadcastWeightsToNeighbors(self, weights_serialized, aggregation_weight=0):
        tasks = []
        for addr in self.config["neighbors"]:
            tasks.append(asyncio.create_task(self.broadcastWeightsTo(weights_serialized,
                addr, aggregation_weight)))
        await asyncio.wait(tasks, return_when=asyncio.ALL_COMPLETED)

    async def broadcastWeightPartitions(self, weights_partitioned_serialized,
        aggregation_weight=0):
        tasks = list()
        for addr, weights_serialized in weights_partitioned_serialized.items():
            tasks.append(asyncio.create_task(self.broadcastWeightsTo(weights_serialized,
                addr, aggregation_weight)))
        await asyncio.wait(tasks, return_when=asyncio.ALL_COMPLETED)

    async def broadcastWeightsAndGradientTo(self, weights_serialized,
        gradient_serialized, address, aggregation_weight=0):
        async with grpc.aio.insecure_channel(address) as channel:
            stub = ModelUpdate_pb2_grpc.ModelUpdateStub(channel)
            await stub.TransferModelUpdate(ModelUpdate_pb2.ModelUpdateMessage(
                weights=ModelUpdate_pb2.ModelWeights(weights=weights_serialized),
                aggregation_weight=aggregation_weight,
                gradient=ModelUpdate_pb2.ModelGradient(gradient=gradient_serialized),
                identity=ModelUpdate_pb2.NetworkIdentity(ip_and_port=self.config["address"])))

    async def broadcastWeightsAndGradientsToNeighbors(self, weights_serialized,
        gradients_serialized, aggregation_weight=0):
        tasks = []
        for addr in self.config["neighbors"]:
            tasks.append(asyncio.create_task(self.broadcastWeightsAndGradientTo(
                weights_serialized, gradients_serialized[addr], addr, aggregation_weight)))
        await asyncio.wait(tasks, return_when=asyncio.ALL_COMPLETED)

    async def broadcastGradientTo(self, gradient_serialized, address, aggregation_weight=0):
        async with grpc.aio.insecure_channel(address) as channel:
            stub = ModelUpdate_pb2_grpc.ModelUpdateStub(channel)
            await stub.TransferModelUpdate(ModelUpdate_pb2.ModelUpdateMessage(
                aggregation_weight=aggregation_weight,
                gradient=ModelUpdate_pb2.ModelGradient(gradient=gradient_serialized),
                identity=ModelUpdate_pb2.NetworkIdentity(ip_and_port=self.config["address"])))

    async def broadcastGradientToNeighbors(self, gradient_serialized, aggregation_weight=0):
        tasks = []
        for addr in self.config["neighbors"]:
            tasks.append(asyncio.create_task(self.broadcastGradientTo(
                gradient_serialized, addr, aggregation_weight)))
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
            eval_metrics = await stub.EvaluateModel(
                ModelUpdate_pb2.ModelWeights(weights=weights_serialized))
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

    def evaluateNeighbors(self):
        weights = self.keras_model.getWeights()
        weights_serialized = SerializationUtils.serializeModelWeights(weights)

        eval_metrics = asyncio.run(self.evaluateWeightsAllNeighbors(weights_serialized))
        eval_metrics.append(self.evaluate())
        eval_avg = dict([(key, np.mean([em[key] for em in eval_metrics]))
            for key in eval_metrics[0].keys()])
        return eval_avg

    def evaluateWeights(self, weights):
        eval_model = self.keras_model.clone()
        eval_model.setWeights(weights)
        eval_metrics = KerasModel.evaluateKerasModel(
            eval_model.getModel(), self.dataset.val)
        return eval_metrics

    def evaluate(self):
        eval_metrics = KerasModel.evaluateKerasModel(
            self.keras_model.getModel(), self.dataset.val)
        return eval_metrics

    def registerTerminationPermission(self, address):
        self.termination_permission[address] = True
        if(all(self.termination_permission.values())):
            self.model_update_service.stopServer()

    async def signalTerminationPermissionTo(self, address):
        async with grpc.aio.insecure_channel(address) as channel:
            stub = ModelUpdate_pb2_grpc.ModelUpdateStub(channel)
            await stub.AllowTermination(ModelUpdate_pb2.NetworkIdentity(
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
        for epoch in range(int(self.config["num_fed_epochs"])):
            self.logger.debug(f'Federated epoch #{epoch}')

            train_metrics = self.fitLocal()
            if(self.config['performance_logging'] and train_metrics):
                metric_keys = list(train_metrics.keys())
                if(isinstance(list(train_metrics.values())[0], list)):
                    # log the result of multiple local epochs in different rows
                    for metric_values in zip(*train_metrics.values()):
                        PerformanceLogger.log(f'{self.config["log_dir"]}/local/train', dict(zip(metric_keys, metric_values)))
                else:
                    PerformanceLogger.log(f'{self.config["log_dir"]}/local/train', dict(zip(metric_keys, list(train_metrics.values()))))

            eval_metrics = self.evaluate()
            if(self.config['performance_logging']):
                PerformanceLogger.log(f'{self.config["log_dir"]}/local/eval', eval_metrics)

            self.broadcast()

            self.aggregate()

            eval_avg = self.evaluateNeighbors()
            if(self.config['performance_logging']):
                PerformanceLogger.log(f'{self.config["log_dir"]}/neighbors/eval', eval_avg)

        eval_avg = self.evaluateNeighbors()
        self.logger.info(f'Evaluation with neighbors resulted in an average of {eval_avg}')

        self.stop()
        if(self.config['performance_logging']):
            PerformanceLogger.write()
