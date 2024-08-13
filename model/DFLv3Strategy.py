from model.AggregationUtils import AggregationUtils
from model.IDFLStrategy import IDFLStrategy
from model.SerializationUtils import SerializationUtils
from network.ModelUpdateService import ModelUpdateService
from tffmodel.Gradient import Gradient

import asyncio
import logging

# MEWMA to predict the next gradients based on currently computed gradients and
#   previously predicted gradients
class MultivariateExponentiallyWeightedMovingAverage:
    def __init__(self, neighbors, shape_gradient, a_ma):
        self.a_ma = a_ma
        self.predictions = dict(
            [(addr, Gradient.getZero(shape_gradient)) for addr in neighbors])

    def get(self):
        return self.predictions

    def predict(self, computed_gradients):
        pred_grad = lambda cg, pp: (cg * self.a_ma) + (pp * (1-self.a_ma))
        new_predictions = dict(
            [(addr, pred_grad(cg, self.predictions[addr])) for addr, cg in computed_gradients.items()])
        self.predictions = new_predictions
        return new_predictions

class DFLv3Strategy(IDFLStrategy):
    def __init__(self, config, keras_model, dataset):
        super().__init__(config, keras_model, dataset)
        self.model_parameters = keras_model.getWeights()
        # shape_gradient = keras_model.computeGradient(dataset)
        shape_gradient = self.model_parameters # gradient and weights have the same shape, only used to determine shape
        # TODO: set the hyperparameter a_ma (i.e., moving average magnitude)
        self.mewma = MultivariateExponentiallyWeightedMovingAverage(
            neighbors=config["neighbors"], shape_gradient=shape_gradient, a_ma=0.99)
        self.logger = logging.getLogger("model/DFLv3Strategy")
        self.logger.setLevel(config["log_level"])

    def startServer(self):
        def transferModelUpdateCallback(weights_serialized, gradient_serialized, address):
            weights = SerializationUtils.deserializeModelWeights(
                weights_serialized, self.keras_model.getWeights())
            gradient = SerializationUtils.deserializeGradient(
                gradient_serialized, next(iter(self.mewma.get().values())))
            self.model_update_market.put((weights, gradient), address)

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
        # TODO: change number of epochs for fit (to 1?)
        fit_history = self.keras_model.fit(self.dataset)
        return fit_history

    def broadcast(self):
        model_parameters_serialized = SerializationUtils.serializeModelWeights(self.model_parameters)
        gradient_predictions_serialized = dict(
            [(addr, SerializationUtils.serializeGradient(pg)) for addr, pg in self.mewma.get().items()])
        asyncio.run(self.broadcastWeightsAndGradientsToNeighbors(
            model_parameters_serialized, gradient_predictions_serialized))

    def computeGradients(self, received_model_updates):
        comp_grad_model = self.keras_model.clone()
        computed_gradients = dict()
        for addr, (mp, _) in received_model_updates.items():
            comp_grad_model.setWeights(mp)
            computed_gradients[addr] = comp_grad_model.computeGradient(self.dataset)
        return computed_gradients

    def aggregate(self):
        # TODO: set the hyperparameters eps_t, alph_t, mu_t, and beta_t (i.e., consensus step-size and mixing weights)
        eps_t = 1
        alph_t = dict([(actor_addr, 1 / len(self.config["neighbors"])) for actor_addr in self.config["neighbors"]])
        mu_t = 1
        beta_t = dict([(actor_addr, 1 / 15) for actor_addr in self.config["neighbors"]])
        current_weights = self.keras_model.getWeights()
        received_model_updates = self.model_update_market.getOneFromAll()
        computed_gradients = self.computeGradients(received_model_updates)
        self.model_parameters, adjusted_model_parameters = AggregationUtils.consensusbasedFedAvgWithGradExchange(
            current_weights, received_model_updates, eps_t, alph_t, mu_t, beta_t)
        self.mewma.predict(computed_gradients)
        self.keras_model.setWeights(adjusted_model_parameters)

    def stop(self):
        self.registerTerminationPermission(self.config["address"])
        asyncio.run(self.signalTerminationPermission())
        self.model_update_service.waitForTermination()
