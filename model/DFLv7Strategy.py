from model.AggregationUtils import AggregationUtils
from model.IDFLStrategy import IDFLStrategy
from model.SerializationUtils import SerializationUtils
from network.ModelUpdateService import ModelUpdateService
from tffmodel.KerasModel import KerasModel
from tffmodel.types.SparseGradient import SparseGradient

import asyncio
import logging

# FedAvg w/ layer-wise top-k Gradient Sparsification
class DFLv7Strategy(IDFLStrategy):
    # TODO: set hyperparameter K_SPARSITY
    K_SPARSITY = 100

    def __init__(self, config, keras_model, dataset):
        super().__init__(config, keras_model, dataset)
        self.logger = logging.getLogger("model/DFLv7Strategy")
        self.logger.setLevel(config["log_level"])

    def startServer(self):
        def transferModelUpdateCallback(_weights_serialized, aggregation_weight,
            sparse_gradient_serialized, address):
            sparse_gradient = SerializationUtils.deserializeSparseGradient(sparse_gradient_serialized)
            self.model_update_market.put((sparse_gradient, aggregation_weight), address)

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

        # TODO: compute and return metrics from fitting and return for the performance logger
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
        sparse_gradient = SparseGradient.sparsifyLayerwiseTopK(
            self.computed_gradient, self.K_SPARSITY)
        sparse_gradient_serialized = SerializationUtils.serializeSparseGradient(sparse_gradient)
        asyncio.run(self.broadcastGradientToNeighbors(sparse_gradient_serialized,
            self.dataset.train.cardinality().numpy()))

    def aggregate(self):
        current_model_weights = self.keras_model.getWeights()
        model_gradients_and_weight = self.model_update_market.get()
        model_gradients, aggregation_weights = zip(*list(model_gradients_and_weight.values()))
        model_gradients = [SparseGradient.sparsifyLayerwiseTopK(
            self.computed_gradient, self.K_SPARSITY), *model_gradients]
        aggregation_weights = [self.dataset.train.cardinality().numpy(), *aggregation_weights]
        avg_model_gradient = AggregationUtils.averageModelWeights(model_gradients, aggregation_weights)
        new_weights = current_model_weights - (avg_model_gradient * self.config["lr_client"])
        self.keras_model.setWeights(new_weights)

    def stop(self):
        self.registerTerminationPermission(self.config["address"])
        asyncio.run(self.signalTerminationPermission())
        self.model_update_service.waitForTermination()
