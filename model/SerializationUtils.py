from tffmodel.Gradient import Gradient
from tffmodel.Weights import Weights

import numpy as np
import pickle
import tensorflow as tf

class SerializationUtils:
    @classmethod
    def serializeModelWeights(self_class, weights):
        # NOTE: we transfer the weights as float32 albeit they could be float64
        # TODO: find a more efficient way to serialize the weights object
        serialized_weights = [np.float32(layer_weights).tobytes() for layer_weights in weights.get()]
        return serialized_weights

    @classmethod
    # NOTE: param source weights is used to identify the respective types and shapes
    def deserializeModelWeights(self_class, weights_serialized, shape_weights):
        weights = [np.frombuffer(layer_weights, dtype=np.float32)
                .reshape(shape_weights[idx].shape)
            for idx, layer_weights in enumerate(weights_serialized)]
        return Weights(weights)

    @classmethod
    def serializeGradient(self_class, gradient):
        # NOTE: we transfer the gradient as float32 albeit it is float64
        # TODO: find a more efficient way to serialize the weights object
        serialized_gradient = [np.float32(layer_gradient).tobytes() for layer_gradient in gradient.get()]
        return serialized_gradient

    @classmethod
    def deserializeGradient(self_class, gradient_serialized, shape_gradient):
        gradient = [np.frombuffer(layer_gradient, dtype=np.float32)
                .reshape(shape_gradient[idx].shape)
            for idx, layer_gradient in enumerate(gradient_serialized)]
        return Gradient(gradient)

    @classmethod
    def serializeModel(self_class, model, optimizer):
        model_config = model.get_config()
        serialized_model_config = pickle.dumps(model_config, protocol=4)
        optimizer_config = tf.keras.optimizers.serialize(optimizer)
        serialized_optimizer_config = pickle.dumps(optimizer_config, protocol=4)
        return serialized_model_config, serialized_optimizer_config

    @classmethod
    def deserializeModel(self_class, serialized_model_config, serialized_optimizer_config):
        model_config = pickle.loads(serialized_model_config)
        model = tf.keras.Sequential.from_config(model_config)
        optimizer_config = pickle.loads(serialized_optimizer_config)
        optimizer = tf.keras.optimizers.deserialize(optimizer_config)
        return model, optimizer
