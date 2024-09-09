from model.AggregationUtils import AggregationUtils
from model.IDFLStrategy import IDFLStrategy
from model.SerializationUtils import SerializationUtils
from network.ModelUpdateService import ModelUpdateService
from tffmodel.KerasModel import KerasModel

import asyncio
import logging
import numpy as np

class DFLv5Strategy(IDFLStrategy):
    def __init__(self, config, keras_model, dataset):
        super().__init__(config, keras_model, dataset)
        self.logger = logging.getLogger("model/DFLv5Strategy")
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
        self.logger.info("Fitting local model.")

        train_metrics = None

        self.computed_gradient = self.keras_model.computeGradient(self.dataset)

        # NOTE: metrics are about the local gradient applied to the current model weights
        if(self.config['performance_logging']):
            eval_model = self.keras_model.clone()
            eval_model.setWeights(self.keras_model.getWeights() -
                (self.computed_gradient * self.config["lr_client"]))
            train_metrics = KerasModel.evaluateKerasModel(
                eval_model.getModel(), self.dataset.train)

        return train_metrics

    def broadcast(self):
        gradient_serialized = SerializationUtils.serializeGradient(self.computed_gradient)
        asyncio.run(self.broadcastGradientToNeighbors(gradient_serialized,
            self.dataset.train.cardinality().numpy()))

    def aggregate(self):
        current_weights = self.keras_model.getWeights()
        model_gradients_and_weight = self.model_update_market.get()
        model_gradients, aggregation_weights = zip(*list(model_gradients_and_weight.values()))
        model_gradients = [self.computed_gradient, *model_gradients]
        aggregation_weights = [self.dataset.train.cardinality().numpy(), *aggregation_weights]

        # TODO: set the hyperparameters a_values (or a_vectors) and tau_eff
        a_values = np.ones(len(model_gradients))
        tau_eff = 1

        new_weights = AggregationUtils.fedNova(current_weights, model_gradients,
            aggregation_weights, tau_eff, self.config["lr_client"], a_values)
        self.keras_model.setWeights(new_weights)

    def stop(self):
        self.registerTerminationPermission(self.config["address"])
        asyncio.run(self.signalTerminationPermission())
        self.model_update_service.waitForTermination()
