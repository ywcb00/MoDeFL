from model.AggregationUtils import AggregationUtils
from model.DFLv1Strategy import DFLv1Strategy
from model.ModelUpdateMarket import ModelUpdateMarket
from model.SerializationUtils import SerializationUtils
from network.ModelUpdateService import ModelUpdateService
from tffmodel.KerasModel import KerasModel
from tffmodel.types.Weights import Weights
from utils.PartitioningUtils import PartitioningUtils

import asyncio
import logging

GLOBAL_PARTITION_FLAG = -1

class DFLv8Strategy(DFLv1Strategy):
    def __init__(self, config, keras_model, dataset):
        super().__init__(config, keras_model, dataset)
        self.global_weight_partition = PartitioningUtils.getParameterPartition(
            keras_model.getWeights(), config["actor_idx"], config["num_workers"])
        self.model_partition_market = ModelUpdateMarket(self.config)
        self.logger = logging.getLogger("model/DFLv8Strategy")
        self.logger.setLevel(config["log_level"])

    def startServer(self):
        def transferModelUpdateCallback(weights_serialized, aggregation_weight, _, address):
            weights = SerializationUtils.deserializeModelWeights(weights_serialized)
            if(aggregation_weight == GLOBAL_PARTITION_FLAG):
                self.model_partition_market.put(weights, address)
            else:
                self.model_update_market.put((weights, aggregation_weight), address)

        def evaluateModelCallback(weights_serialized):
            weights = SerializationUtils.deserializeModelWeights(weights_serialized)
            eval_metrics = self.evaluateWeights(weights)
            return eval_metrics

        self.termination_permission = dict(
            [(addr, False) for addr in self.config["neighbors"]])
        self.termination_permission[self.config["address"]] = False
        def allowTerminationCallback(address):
            self.registerTerminationPermission(address)

        callbacks = {"TransferModelUpdate": transferModelUpdateCallback,
            "EvaluateModel": evaluateModelCallback,
            "AllowTermination": allowTerminationCallback}

        self.model_update_service = ModelUpdateService(self.config)
        self.model_update_service.startServer(callbacks)

    def broadcast(self):
        current_weights = self.keras_model.getWeights()
        model_delta = current_weights - self.previous_weights

        model_delta_partitioned = PartitioningUtils.partitionModelParameters(model_delta, self.config)
        model_delta_partitioned = {addr: SerializationUtils.serializeModelWeights(weights)
            for addr, weights in model_delta_partitioned.items()}

        asyncio.run(self.broadcastWeightPartitions(model_delta_partitioned,
            self.dataset.train.cardinality().numpy()))

    def aggregateWeightPartitions(self):
        current_model_delta = PartitioningUtils.getParameterPartition(
            (self.keras_model.getWeights() - self.previous_weights),
            self.config["actor_idx"], self.config["num_workers"])
        model_deltas_and_weight = self.model_update_market.get()
        model_deltas, aggregation_weights = zip(*list(model_deltas_and_weight.values()))
        model_deltas = [current_model_delta, *model_deltas]
        aggregation_weights = [self.dataset.train.cardinality().numpy(), *aggregation_weights]
        avg_model_deltas = AggregationUtils.averageModelWeights(model_deltas, aggregation_weights)
        self.global_weight_partition = self.global_weight_partition + avg_model_deltas

    def broadcastGlobalWeightPartition(self):
        global_partition_serialized = SerializationUtils.serializeModelWeights(
            self.global_weight_partition)
        asyncio.run(self.broadcastWeightsToNeighbors(global_partition_serialized,
            aggregation_weight=GLOBAL_PARTITION_FLAG))

    def setLocalWeights(self):
        new_weights = Weights.getZero(self.keras_model.getWeights())
        actor_idx_lookup_dict = dict(zip(self.config["neighbors"], self.config["neighbor_idx"]))
        model_partitions = self.model_partition_market.getOneFromAll()
        # re-construct weight matrix from partitions
        # local actor's partition
        for counter, part_idx in enumerate(PartitioningUtils.getPartitionIndices(
            new_weights.getLength(), self.config["actor_idx"], self.config["num_workers"])):
            new_weights.setLayer(part_idx, self.global_weight_partition[counter])
        # other actors' partitions
        for addr, weight_partition in model_partitions.items():
            for counter, part_idx in enumerate(PartitioningUtils.getPartitionIndices(
                new_weights.getLength(), actor_idx_lookup_dict[addr], self.config["num_workers"])):
                new_weights.setLayer(part_idx, weight_partition[counter])
        self.keras_model.setWeights(new_weights)

    def aggregate(self):
        self.aggregateWeightPartitions()
        self.broadcastGlobalWeightPartition()
        self.setLocalWeights()
