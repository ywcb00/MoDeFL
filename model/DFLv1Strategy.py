from model.AggregationUtils import AggregationUtils
from model.IDFLStrategy import IDFLStrategy
from model.SerializationUtils import SerializationUtils
from network.ModelUpdateService import ModelUpdateService
from network.PartialDeviceParticipation import PartialDeviceParticipation
from tffmodel.KerasModel import KerasModel
from utils.CommunicationLogger import CommunicationLogger

import asyncio
import logging

class DFLv1Strategy(IDFLStrategy):
    def __init__(self, config, keras_model, dataset):
        super().__init__(config, keras_model, dataset)
        self.logger = logging.getLogger("model/DFLv1Strategy")
        self.logger.setLevel(config["log_level"])

    def startServer(self):
        def transferModelUpdateCallback(update, address):
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

    def fitLocal(self):
        self.logger.info(f'Fitting local model for {self.config["num_local_epochs"]} local epochs.')
        self.previous_weights = self.keras_model.getWeights()
        fit_history = self.keras_model.fit(self.dataset)
        train_metrics = fit_history.history
        return train_metrics

    def broadcast(self):
        current_weights = self.keras_model.getWeights()
        model_delta = current_weights - self.previous_weights

        selected_neighbors = PartialDeviceParticipation.getNeighbors(self.config)

        if(self.config["log_communication_flag"]):
            CommunicationLogger.logMultiple(self.config["address"], selected_neighbors,
                {"size": model_delta.getSize(), "dtype": model_delta.getDTypeName()})

        asyncio.run(self.broadcastWeightsToNeighbors(model_delta,
            self.dataset.train.cardinality().numpy(),
            selected_neighbors=selected_neighbors))

    def aggregate(self):
        current_model_delta = self.keras_model.getWeights() - self.previous_weights
        received_model_updates_vals = self.model_update_market.get().values()
        model_deltas = [rmu["weights"] for rmu in received_model_updates_vals]
        aggregation_weights = [rmu["aggregation_weight"] for rmu in received_model_updates_vals]
        # model_deltas, aggregation_weights = zip(*list(model_deltas_and_weight.values()))
        model_deltas = [current_model_delta, *model_deltas]
        aggregation_weights = [self.dataset.train.cardinality().numpy(), *aggregation_weights]
        avg_model_deltas = AggregationUtils.averageModelWeights(model_deltas, aggregation_weights)
        new_weights = self.previous_weights + avg_model_deltas
        self.keras_model.setWeights(new_weights)

    def stop(self):
        self.registerTerminationPermission(self.config["address"])
        asyncio.run(self.signalTerminationPermission())
        self.model_update_service.waitForTermination()
