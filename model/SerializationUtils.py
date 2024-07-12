
import numpy as np

class SerializationUtils:
    @classmethod
    def serializeModelWeights(self_class, weights):
        serialized_weights = [layer_weights.tobytes() for layer_weights in weights]
        return serialized_weights

    @classmethod
    # NOTE: param source weights is used to identify the respective types and shapes
    def deserializeModelWeights(self_class, weights_serialized, source_weights):
        weights = [np.frombuffer(layer_weights, dtype=source_weights[idx].dtype.name)
            .reshape(source_weights[idx].shape)
            for idx, layer_weights in enumerate(weights_serialized)]
        return weights
