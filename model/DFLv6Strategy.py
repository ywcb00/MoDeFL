from model.AggregationUtils import AggregationUtils
from model.IDFLStrategy import IDFLStrategy
from model.SerializationUtils import SerializationUtils
from network.ModelUpdateService import ModelUpdateService
from tffmodel.KerasModel import KerasModel

import asyncio
import logging

# FedAvg using gradients
class DFLv6Strategy(IDFLStrategy):
    def __init__(self, config, keras_model, dataset):
        super().__init__(config, keras_model, dataset)
        self.logger = logging.getLogger("model/DFLv6Strategy")
        self.logger.setLevel(config["log_level"])

    def startServer(self):
        def transferModelUpdateCallback(_weights_serialized, aggregation_weight,
            gradient_serialized, address):
            gradient = SerializationUtils.deserializeGradient(gradient_serialized)
            self.model_update_market.put((gradient, aggregation_weight), address)

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

    def fitLocal(self):
        self.logger.info(f'Fitting local model for {self.config["num_epochs"]} local epochs.')

        self.previous_weights = self.keras_model.getWeights()

        self.computed_gradient, train_metrics = self.keras_model.fitGradient(self.dataset)

        return train_metrics

    def broadcast(self):
        gradient_serialized = SerializationUtils.serializeGradient(self.computed_gradient)
        asyncio.run(self.broadcastGradientToNeighbors(gradient_serialized,
            self.dataset.train.cardinality().numpy()))

    def aggregate(self):
        model_gradients_and_weight = self.model_update_market.get()
        model_gradients, aggregation_weights = zip(*list(model_gradients_and_weight.values()))
        model_gradients = [self.computed_gradient, *model_gradients]
        aggregation_weights = [self.dataset.train.cardinality().numpy(), *aggregation_weights]
        avg_model_gradient = AggregationUtils.averageModelWeights(model_gradients, aggregation_weights)
        new_weights = self.previous_weights - (avg_model_gradient * self.config["lr_server"])
        self.keras_model.setWeights(new_weights)

    def stop(self):
        self.registerTerminationPermission(self.config["address"])
        asyncio.run(self.signalTerminationPermission())
        self.model_update_service.waitForTermination()
