from model.AggregationUtils import AggregationUtils
from model.IDFLStrategy import IDFLStrategy
from model.SerializationUtils import SerializationUtils
from network.ModelUpdateService import ModelUpdateService
from tffmodel.KerasModel import KerasModel

import asyncio
import logging

class DFLv1Strategy(IDFLStrategy):
    def __init__(self, config, keras_model, dataset):
        super().__init__(config, keras_model, dataset)
        self.logger = logging.getLogger("model/DFLv1Strategy")
        self.logger.setLevel(config["log_level"])

    def startServer(self):
        def transferModelUpdateCallback(weights_serialized, _, address):
            weights = SerializationUtils.deserializeModelWeights(
                weights_serialized, self.keras_model.getWeights())
            self.model_update_market.put(weights, address)

        def evaluateModelCallback(weights_serialized):
            weights = SerializationUtils.deserializeModelWeights(
                weights_serialized, self.keras_model.getWeights())
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
        self.logger.info("Fitting local model.")
        self.previous_weights = self.keras_model.getWeights()
        # TODO: change number of epochs for fit (to 1?)
        fit_history = self.keras_model.fit(self.dataset)
        return fit_history

    def broadcast(self):
        current_weights = self.keras_model.getWeights()
        model_delta = current_weights - self.previous_weights
        model_delta_serialized = SerializationUtils.serializeModelWeights(model_delta)

        asyncio.run(self.broadcastWeightsToNeighbors(model_delta_serialized))

    def aggregate(self):
        current_weights = self.keras_model.getWeights()
        model_deltas = self.model_update_market.getOneFromAll()
        avg_model_deltas = AggregationUtils.averageModelWeights(list(model_deltas.values()))
        new_weights = current_weights + avg_model_deltas
        self.keras_model.setWeights(new_weights)

    def stop(self):
        self.registerTerminationPermission(self.config["address"])
        asyncio.run(self.signalTerminationPermission())
        self.model_update_service.waitForTermination()
