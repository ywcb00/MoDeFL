from tffmodel.types.Gradient import Gradient
from tffmodel.types.SparseGradient import SparseGradient
from tffmodel.types.Weights import Weights

import numpy as np
import pickle
import tensorflow as tf

class SerializationUtils:
    @classmethod
    def serializeModelWeights(self_class, weights):
        return weights.serialize()

    @classmethod
    def deserializeModelWeights(self_class, weights_serialized):
        if(not weights_serialized):
            return None
        return Weights.deserialize(weights_serialized)

    @classmethod
    def serializeGradient(self_class, gradient):
        return gradient.serialize()

    @classmethod
    def deserializeGradient(self_class, gradient_serialized):
        if(not gradient_serialized):
            return None
        return Gradient.deserialize(gradient_serialized)

    @classmethod
    def serializeSparseGradient(self_class, sparse_gradient):
        return sparse_gradient.serialize()

    @classmethod
    def deserializeSparseGradient(self_class, sparse_gradient_serialized):
        if(not sparse_gradient_serialized):
            return None
        return SparseGradient.deserialize(sparse_gradient_serialized)

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
