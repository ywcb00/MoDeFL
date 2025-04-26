from network.Compression import Compression
from tffmodel.types.HeterogeneousDenseArray import HeterogeneousDenseArray
from tffmodel.types.HeterogeneousSparseArray import HeterogeneousSparseArray

import numpy as np
import pickle
import tensorflow as tf

class SerializationUtils:
    # serialize parameters of type HeterogeneousArray into an array of byte-strings
    @classmethod
    def serializeParameters(self_class, parameters):
        return parameters.serialize()

    # deserialize parameters from a byte-string back into a HeterogeneousArray
    @classmethod
    def deserializeParameters(self_class, serialized_parameters, sparse=False):
        if(not serialized_parameters):
            return None
        if(sparse):
            data = HeterogeneousSparseArray.deserialize(serialized_parameters)
            data = Compression.decompress(data) # decompress the data if compressed
            return data
        else:
            data = HeterogeneousDenseArray.deserialize(serialized_parameters)
            data = Compression.decompress(data) # decompress the data if compressed
            return data

    # serialize the model architecture and the optimizer configuration of a keras model
    @classmethod
    def serializeModel(self_class, model, optimizer):
        model_config = model.get_config()
        serialized_model_config = pickle.dumps(model_config, protocol=4)
        optimizer_config = tf.keras.optimizers.serialize(optimizer)
        serialized_optimizer_config = pickle.dumps(optimizer_config, protocol=4)
        return serialized_model_config, serialized_optimizer_config

    # deserialize the model architecture and the optimizer configuration of a keras model
    @classmethod
    def deserializeModel(self_class, serialized_model_config, serialized_optimizer_config):
        model_config = pickle.loads(serialized_model_config)
        model = tf.keras.Sequential.from_config(model_config)
        optimizer_config = pickle.loads(serialized_optimizer_config)
        optimizer = tf.keras.optimizers.deserialize(optimizer_config)
        return model, optimizer
