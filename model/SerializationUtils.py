from network.Compression import Compression
from tffdataset.DatasetUtils import DatasetID
from tffmodel.types.HeterogeneousDenseArray import HeterogeneousDenseArray
from tffmodel.types.HeterogeneousSparseArray import HeterogeneousSparseArray

import numpy as np

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

    @classmethod
    def deserializeModel(self_class, serialized_model_config, serialized_optimizer_config, config):
        match config["dataset_id"]:
            case DatasetID.FloodNet | DatasetID.Mnist | DatasetID.Iris:
                from tffmodel.KerasModel import KerasModel
                return KerasModel.deserializeModel(serialized_model_config, serialized_optimizer_config)
            case DatasetID.IrisPyTorch:
                from tffmodel.PyTorchModel import PyTorchModel
                return PyTorchModel.deserializeModel(serialized_model_config, serialized_optimizer_config)
            case _:
                raise NotImplementedError
