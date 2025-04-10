from model.AggregationUtils import AggregationUtils
from model.IDFLStrategy import IDFLStrategy
from model.SerializationUtils import SerializationUtils
from network.ModelUpdateService import ModelUpdateService
from tffmodel.KerasModel import KerasModel
from utils.CommunicationLogger import CommunicationLogger

import asyncio
import logging

class DFLv2Strategy(IDFLStrategy):
    def __init__(self, config, keras_model, dataset):
        super().__init__(config, keras_model, dataset)
        self.logger = logging.getLogger("model/DFLv2Strategy")
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
        fit_history = self.keras_model.fit(self.dataset)
        train_metrics = fit_history.history
        return train_metrics

    def broadcast(self):
        weights = self.keras_model.getWeights()
        weights_serialized = SerializationUtils.serializeModelWeights(weights)

        if(self.config["log_communication_flag"]):
            CommunicationLogger.logMultiple(self.config["address"], self.config["neighbors"],
                {"size": weights.getSize(), "dtype": weights.getDTypeName()})

        asyncio.run(self.broadcastWeightsToNeighbors(weights_serialized))

    def aggregate(self):
        # TODO: set the hyperparameters eps_t and alph_t (i.e., consensus step-size and mixing weights)
        eps_t = 1 / len(self.config["neighbors"])
        alph_t = dict([(actor_addr, 1) for actor_addr in self.config["neighbors"]])
        current_weights = self.keras_model.getWeights()
        received_model_updates = self.model_update_market.get()
        received_model_weights = {key: val["weights"] for key, val in received_model_updates.items()}
        new_weights = AggregationUtils.consensusbasedFedAvg(
            current_weights, received_model_weights, eps_t, alph_t)
        self.keras_model.setWeights(new_weights)

    def stop(self):
        self.registerTerminationPermission(self.config["address"])
        asyncio.run(self.signalTerminationPermission())
        self.model_update_service.waitForTermination()
