from model.AggregationUtils import AggregationUtils
from model.IDFLStrategy import IDFLStrategy
from model.SerializationUtils import SerializationUtils
from network.Compression import Compression
from network.ModelUpdateService import ModelUpdateService
from tffmodel.KerasModel import KerasModel

import asyncio
import logging

# FedAvg w/ Gradient Compression
class DFLv7Strategy(IDFLStrategy):
    def __init__(self, config, keras_model, dataset):
        super().__init__(config, keras_model, dataset)
        self.logger = logging.getLogger("model/DFLv7Strategy")
        self.logger.setLevel(config["log_level"])

    def startServer(self):
        # callback for receiving a model update from an actor
        def transferModelUpdateCallback(update, address):
            self.model_update_market.putUpdate(update, address)

        # callback for getting an evaluation request form an actor
        def evaluateModelCallback(request):
            weights = SerializationUtils.deserializeParameters(
                request.parameters, sparse=request.sparse)
            eval_metrics = self.evaluateWeights(weights)
            return eval_metrics

        # dictionary to track which neighboring actor has sent its last model update
        self.termination_permission = dict(
            [(addr, False) for addr in self.config["neighbors"]])
        self.termination_permission[self.config["address"]] = False
        # callback for registering a termination permission from an actor
        def allowTerminationCallback(address):
            self.registerTerminationPermission(address)

        callbacks = {"TransferModelUpdate": transferModelUpdateCallback,
            "EvaluateModel": evaluateModelCallback,
            "AllowTermination": allowTerminationCallback}

        self.model_update_service = ModelUpdateService(self.config)
        self.model_update_service.startServer(callbacks)

    def fitLocal(self):
        self.logger.info(f'Fitting local model for {self.config["num_local_epochs"]} local epochs.')

        train_metrics = None
        self.previous_weights = self.keras_model.getWeights()

        self.computed_gradient, train_metrics = self.keras_model.fitGradient(self.dataset)

        return train_metrics

    def broadcast(self):
        asyncio.run(self.broadcastParametersToNeighbors(gradient=self.computed_gradient,
            aggregation_weight=self.dataset.train.cardinality().numpy()))

    def aggregate(self):
        received_model_update_vals = self.model_update_market.get().values()
        model_gradients = [rmu["gradient"] for rmu in received_model_update_vals]
        aggregation_weights = [rmu["aggregation_weight"] for rmu in received_model_update_vals]
        model_gradients = [Compression.compressDecompress(
            self.computed_gradient, self.config), *model_gradients]
        aggregation_weights = [self.dataset.train.cardinality().numpy(), *aggregation_weights]
        avg_model_gradient = AggregationUtils.averageModelParameters(model_gradients, aggregation_weights)
        new_weights = self.previous_weights - (avg_model_gradient * self.config["lr_global"])
        self.keras_model.setWeights(new_weights)

    # notify the neighbors about the completion and wait until this actor can terminate safely
    def stop(self):
        self.registerTerminationPermission(self.config["address"])
        asyncio.run(self.signalTerminationPermission())
        self.model_update_service.waitForTermination()
