from model.ModelUpdateMarket import ModelUpdateMarket
from model.SerializationUtils import SerializationUtils
from network.Compression import Compression
from network.PartialDeviceParticipation import PartialDeviceParticipation
import network.protos.ModelUpdate_pb2 as ModelUpdate_pb2
import network.protos.ModelUpdate_pb2_grpc as ModelUpdate_pb2_grpc
from tffmodel.KerasModel import KerasModel
from utils.CommunicationLogger import CommunicationLogger
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

    # define the required callbacks for the service and start the model update service
    @abstractmethod
    def startServer(self):
        pass

    # train the local model on the local data
    @abstractmethod
    def fitLocal(self):
        pass

    # construct the model update message and broadcast it to the specified address
    async def broadcastParametersTo(self, address, weights_serialized=None, weights_sparse=None,
        gradient_serialized=None, gradient_sparse=None, aggregation_weight=0):
        async with grpc.aio.insecure_channel(address) as channel:
            weights_msg = ModelUpdate_pb2.ModelParameters(sparse=weights_sparse,
                parameters=weights_serialized) if weights_serialized else None
            gradient_msg = ModelUpdate_pb2.ModelParameters(sparse=gradient_sparse,
                parameters=gradient_serialized) if gradient_serialized else None
            stub = ModelUpdate_pb2_grpc.ModelUpdateStub(channel)
            await stub.TransferModelUpdate(ModelUpdate_pb2.ModelUpdateMessage(
                update=ModelUpdate_pb2.ModelParameterUpdate(
                    weights=weights_msg,
                    gradient=gradient_msg,
                    aggregation_weight=aggregation_weight),
                identity=ModelUpdate_pb2.NetworkIdentity(ip_and_port=self.config["address"])))

    # broadcast the model update to the neighboring actors
    async def broadcastParametersToNeighbors(self, weights=None, gradient=None, aggregation_weight=0):
        # neighbors are selected by partial device participation strategy
        selected_neighbors = PartialDeviceParticipation.getNeighbors(self.config)

        weights_serialized = None
        weights_sparse = None
        gradient_serialized = None
        gradient_sparse = None
        if(weights):
            # apply compression by the specified compression method
            weights = Compression.compress(weights, self.config)
            weights_serialized = weights.serialize()
            weights_sparse = weights.is_sparse
            if(self.config["log_communication_flag"]):
                CommunicationLogger.logMultiple(self.config["address"], selected_neighbors,
                    {"size": weights.getSize(), "dtype": weights.getDTypeName()})
        if(gradient):
            # apply compression by the specified compression method
            gradient = Compression.compress(gradient, self.config)
            gradient_serialized = gradient.serialize()
            gradient_sparse = gradient.is_sparse
            if(self.config["log_communication_flag"]):
                CommunicationLogger.logMultiple(self.config["address"], selected_neighbors,
                    {"size": gradient.getSize(), "dtype": gradient.getDTypeName()})

        tasks = []
        self.logger.debug(f'Broadcasting updates to {len(selected_neighbors)} neighboring actors.')
        for addr in self.config["neighbors"]:
            if addr in selected_neighbors:
                tasks.append(asyncio.create_task(self.broadcastParametersTo(addr,
                    weights_serialized=weights_serialized, weights_sparse=weights_sparse,
                    gradient_serialized=gradient_serialized, gradient_sparse=gradient_sparse,
                    aggregation_weight=aggregation_weight)))
            else: # send an empty model update message to excluded neighbors
                tasks.append(asyncio.create_task(self.broadcastParametersTo(addr)))
        await asyncio.wait(tasks, return_when=asyncio.ALL_COMPLETED)

    # broadcast the partitions of a partitioned model to the respective actors
    async def broadcastWeightPartitions(self, weights_partitioned,
        aggregation_weight=0):
        # apply compression by the specified compression method
        weights_partitioned = {addr: Compression.compress(weights, self.config)
            for addr, weights in weights_partitioned.items()}

        weights_partitioned_serialized = {addr: weights.serialize()
            for addr, weights in weights_partitioned.items()}
        weights_partitioned_sparse = {addr: weights.is_sparse
            for addr, weights in weights_partitioned.items()}

        selected_neighbors = PartialDeviceParticipation.getNeighbors(self.config)

        if(self.config["log_communication_flag"]):
            for addr, weights in weights_partitioned.items():
                if addr in selected_neighbors:
                    CommunicationLogger.log(self.config["address"], addr,
                        {"size": weights.getSize(), "dtype": weights.getDTypeName()})

        tasks = list()
        for addr, weights_serialized in weights_partitioned_serialized.items():
            if addr in selected_neighbors:
                tasks.append(asyncio.create_task(self.broadcastParametersTo(addr,
                    weights_serialized=weights_serialized,
                    weights_sparse=weights_partitioned_sparse[addr],
                    aggregation_weight=aggregation_weight)))
            else: # send an empty model update message to excluded neighbors
                tasks.append(asyncio.create_task(self.broadcastParametersTo(addr)))
        await asyncio.wait(tasks, return_when=asyncio.ALL_COMPLETED)

    # broadcast weights and individual gradients to the neighboring actors
    async def broadcastWeightsAndGradientsToNeighbors(self, weights,
        gradient_dict, aggregation_weight=0):
        # apply compression by the specified compression method
        weights = Compression.compress(weights, self.config)
        gradient_dict = {addr: Compression.compress(grad, self.config)
            for addr, grad in gradient_dict.items()}

        weights_serialized = weights.serialize()
        weights_sparse = weights.is_sparse
        gradient_serialized_dict = dict(
            [(addr, SerializationUtils.serializeParameters(grad)) for addr, grad in gradient_dict.items()])
        gradient_sparse_dict = dict(
            [(addr, grad.is_sparse) for addr, grad in gradient_dict.items()])

        selected_neighbors = PartialDeviceParticipation.getNeighbors(self.config)

        if(self.config["log_communication_flag"]):
            CommunicationLogger.logMultiple(self.config["address"], selected_neighbors,
                {"size": weights.getSize(), "dtype": weights.getDTypeName()})
            for addr, grad in gradient_dict.items():
                if addr in selected_neighbors:
                    CommunicationLogger.log(self.config["address"], addr,
                        {"size": grad.getSize(), "dtype": grad.getDTypeName()})

        tasks = []
        self.logger.debug(f'Broadcasting updates to {len(selected_neighbors)} neighboring actors.')
        for addr in self.config["neighbors"]:
            if addr in selected_neighbors:
                tasks.append(asyncio.create_task(self.broadcastParametersTo(addr,
                    weights_serialized=weights_serialized,
                    weights_sparse=weights_sparse,
                    gradient_serialized=gradient_serialized_dict[addr],
                    gradient_sparse=gradient_sparse_dict[addr],
                    aggregation_weight=aggregation_weight)))
            else: # send an empty model update message to excluded neighbors
                tasks.append(asyncio.create_task(self.broadcastParametersTo(addr)))
        await asyncio.wait(tasks, return_when=asyncio.ALL_COMPLETED)

    # model update exchange step of the actor
    @abstractmethod
    def broadcast(self):
        pass

    # aggregation step of the actor
    @abstractmethod
    def aggregate(self):
        pass

    # obtain evaluation metrics from our own model evaluated on the neighbors' evaluation data
    async def evaluateWeightsNeighbor(self, weights_serialized, weights_sparse, address):
        async with grpc.aio.insecure_channel(address) as channel:
            stub = ModelUpdate_pb2_grpc.ModelUpdateStub(channel)
            eval_metrics = await stub.EvaluateModel(
                ModelUpdate_pb2.ModelParameters(sparse=weights_sparse, parameters=weights_serialized))
        return eval_metrics.metrics

    async def evaluateWeightsAllNeighbors(self, weights):
        weights_serialized = weights.serialize()
        weights_sparse = weights.is_sparse
        tasks = []
        for addr in self.config["neighbors"]:
            tasks.append(asyncio.create_task(self.evaluateWeightsNeighbor(
                weights_serialized, weights_sparse, addr)))
        eval_metrics = []
        for t in tasks:
            response = await t
            eval_metrics.append(dict([(elem.key, elem.value) for elem in response]))
        return eval_metrics

    def evaluateNeighbors(self):
        weights = self.keras_model.getWeights()

        eval_metrics = asyncio.run(self.evaluateWeightsAllNeighbors(weights))
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

    # register the termination permission of a neighboring actor
    def registerTerminationPermission(self, address):
        self.termination_permission[address] = True
        if(all(self.termination_permission.values())):
            self.model_update_service.stopServer()

    # notify the specified neighboring actor that we are ready to terminate
    async def signalTerminationPermissionTo(self, address):
        async with grpc.aio.insecure_channel(address) as channel:
            stub = ModelUpdate_pb2_grpc.ModelUpdateStub(channel)
            await stub.AllowTermination(ModelUpdate_pb2.NetworkIdentity(
                ip_and_port=self.config["address"]))

    # notify all neighboring actors that we are ready to terminate
    async def signalTerminationPermission(self):
        tasks = []
        for addr in self.config["neighbors"]:
            tasks.append(asyncio.create_task(self.signalTerminationPermissionTo(addr)))
        await asyncio.wait(tasks, return_when=asyncio.ALL_COMPLETED)

    @abstractmethod
    def stop(self):
        pass

    # general training loop
    def performTraining(self):
        self.startServer()

        # TODO: think about the number of epochs for learning (perhaps termination based on local training loss?)
        for epoch in range(int(self.config["num_fed_epochs"])):
            self.logger.debug(f'Federated epoch #{epoch}')

            # train the local model
            train_metrics = self.fitLocal()
            if(self.config['log_performance_flag'] and train_metrics):
                metric_keys = list(train_metrics.keys())
                if(isinstance(list(train_metrics.values())[0], list)):
                    # log the result of multiple local epochs in different rows
                    for metric_values in zip(*train_metrics.values()):
                        PerformanceLogger.log(f'{self.config["log_dir"]}/local/train', dict(zip(metric_keys, metric_values)))
                else:
                    PerformanceLogger.log(f'{self.config["log_dir"]}/local/train', dict(zip(metric_keys, list(train_metrics.values()))))

            # evaluate the local model
            eval_metrics = self.evaluate()
            if(self.config['log_performance_flag']):
                PerformanceLogger.log(f'{self.config["log_dir"]}/local/eval', eval_metrics)

            # send model update to neighboring actors
            self.broadcast()

            # aggregate model updates received from neighboring actors and update local model
            self.aggregate()

            # evaluate the local model on the neighboring actors
            eval_avg = self.evaluateNeighbors()
            if(self.config['log_performance_flag']):
                PerformanceLogger.log(f'{self.config["log_dir"]}/neighbors/eval', eval_avg)

        eval_avg = self.evaluateNeighbors()
        self.logger.info(f'Evaluation with neighbors resulted in an average of {eval_avg}')

        self.stop()
        if(self.config['log_performance_flag']):
            PerformanceLogger.write()
        if(self.config['log_communication_flag']):
            CommunicationLogger.write(f'{self.config["log_dir"]}/network/communication')
