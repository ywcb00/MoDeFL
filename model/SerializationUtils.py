from tffmodel.Weights import Weights

import numpy as np
import pickle
import tensorflow as tf

class SerializationUtils:
    @classmethod
    def serializeModelWeights(self_class, weights):
        # TODO: find a more efficient way to serialize the weights object
        serialized_weights = [layer_weights.tobytes() for layer_weights in weights._weights]
        return serialized_weights

    @classmethod
    # NOTE: param source weights is used to identify the respective types and shapes
    def deserializeModelWeights(self_class, weights_serialized, source_weights):
        weights = [np.frombuffer(layer_weights, dtype=source_weights[idx].dtype.name)
                .reshape(source_weights[idx].shape)
            for idx, layer_weights in enumerate(weights_serialized)]
        return Weights(weights)

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
