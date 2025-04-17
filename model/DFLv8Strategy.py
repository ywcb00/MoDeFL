from model.AggregationUtils import AggregationUtils
from model.DFLv1Strategy import DFLv1Strategy
from model.ModelUpdateMarket import ModelUpdateMarket
from model.SerializationUtils import SerializationUtils
from network.ModelUpdateService import ModelUpdateService
from tffmodel.KerasModel import KerasModel
from tffmodel.types.HeterogeneousDenseArray import HeterogeneousDenseArray
from utils.PartitioningUtils import PartitioningUtils
from utils.CommunicationLogger import CommunicationLogger

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
        def transferModelUpdateCallback(update, address):
            if(update.aggregation_weight == GLOBAL_PARTITION_FLAG):
                self.model_partition_market.putUpdate(update, address)
            else:
                self.model_update_market.putUpdate(update, address)

        def evaluateModelCallback(request):
            weights = SerializationUtils.deserializeParameters(
                request.parameters, sparse=request.sparse)
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

        if(self.config["log_communication_flag"]):
            for addr, weights in model_delta_partitioned.items():
                CommunicationLogger.log(self.config["address"], addr,
                    {"size": weights.getSize(), "dtype": weights.getDTypeName()})

        asyncio.run(self.broadcastWeightPartitions(model_delta_partitioned,
            self.dataset.train.cardinality().numpy()))

    def aggregateWeightPartitions(self):
        current_model_delta = PartitioningUtils.getParameterPartition(
            (self.keras_model.getWeights() - self.previous_weights),
            self.config["actor_idx"], self.config["num_workers"])
        received_model_update_vals = self.model_update_market.get().values()
        model_deltas = [rmu["weights"] for rmu in received_model_update_vals]
        aggregation_weights = [rmu["aggregation_weight"] for rmu in received_model_update_vals]
        model_deltas = [current_model_delta, *model_deltas]
        aggregation_weights = [self.dataset.train.cardinality().numpy(), *aggregation_weights]
        avg_model_deltas = AggregationUtils.averageModelWeights(model_deltas, aggregation_weights)
        self.global_weight_partition = self.global_weight_partition + avg_model_deltas

    def broadcastGlobalWeightPartition(self):
        if(self.config["log_communication_flag"]):
            CommunicationLogger.logMultiple(self.config["address"], self.config["neighbors"],
                {"size": self.global_weight_partition.getSize(), "dtype": self.global_weight_partition.getDTypeName()})

        asyncio.run(self.broadcastWeightsToNeighbors(self.global_weight_partition,
            aggregation_weight=GLOBAL_PARTITION_FLAG))

    def setLocalWeights(self):
        actor_idx_lookup_dict = dict(zip(self.config["neighbors"], self.config["neighbor_idx"]))
        received_model_partitions = self.model_partition_market.getOneFromAll()

        # re-construct weight matrix from partitions
        partition_dict = {actor_idx_lookup_dict[addr]: elem["weights"] for addr, elem in received_model_partitions.items()}
        partition_dict[self.config["actor_idx"]] = self.global_weight_partition
        flattened_parameters = PartitioningUtils.joinParameterPartitions(partition_dict)
        new_weights = HeterogeneousDenseArray.fromFlattened(flattened_parameters, self.keras_model.getWeights())

        self.keras_model.setWeights(new_weights)

    def aggregate(self):
        self.aggregateWeightPartitions()
        self.broadcastGlobalWeightPartition()
        self.setLocalWeights()
