from model.AggregationUtils import AggregationUtils
from model.IDFLStrategy import IDFLStrategy
from model.SerializationUtils import SerializationUtils
from network.ModelUpdateService import ModelUpdateService
from tffmodel.KerasModel import KerasModel
from utils.CommunicationLogger import CommunicationLogger

import asyncio
import logging
import numpy as np

class DFLv5Strategy(IDFLStrategy):
    def __init__(self, config, keras_model, dataset):
        super().__init__(config, keras_model, dataset)
        self.logger = logging.getLogger("model/DFLv5Strategy")
        self.logger.setLevel(config["log_level"])

    def startServer(self):
        def transferModelUpdateCallback(update, address):
            self.model_update_market.putUpdate(update, address)

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
        self.logger.info(f'Fitting local model for {self.config["num_local_epochs"]} local epochs.')

        self.previous_weights = self.keras_model.getWeights()
        self.computed_gradient, train_metrics = self.keras_model.fitGradient(self.dataset)

        return train_metrics

    def broadcast(self):
        gradient_serialized = SerializationUtils.serializeGradient(self.computed_gradient)

        if(self.config["log_communication_flag"]):
            CommunicationLogger.logMultiple(self.config["address"], self.config["neighbors"],
                {"size": self.computed_gradient.getSize(), "dtype": self.computed_gradient.getDTypeName()})

        asyncio.run(self.broadcastGradientToNeighbors(gradient_serialized,
            self.dataset.train.cardinality().numpy()))

    def aggregate(self):
        received_model_update_vals = self.model_update_market.get().values()
        model_gradients = [rmu["gradient"] for rmu in received_model_update_vals]
        aggregation_weights = [rmu["aggregation_weight"] for rmu in received_model_update_vals]
        model_gradients = [self.computed_gradient, *model_gradients]
        aggregation_weights = [self.dataset.train.cardinality().numpy(), *aggregation_weights]

        # TODO: set the hyperparameters a_values (or a_vectors) and tau_eff
        a_values = np.ones(len(model_gradients))
        tau_eff = 1

        new_weights = AggregationUtils.fedNova(self.previous_weights, model_gradients,
            aggregation_weights, tau_eff, self.config["lr_server"], a_values)
        self.keras_model.setWeights(new_weights)

    def stop(self):
        self.registerTerminationPermission(self.config["address"])
        asyncio.run(self.signalTerminationPermission())
        self.model_update_service.waitForTermination()
