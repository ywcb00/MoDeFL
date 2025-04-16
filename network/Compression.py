from tffmodel.types.HeterogeneousArray import HeterogeneousArray

from enum import Enum
import math
import numpy as np

class CompressionType(Enum):
    NoneType = 0 # do not apply compression
    # ===== Quantization =====
    QUANTIZE_PROBABILISTIC = 1
    # ===== Sparsification =====
    SPARSIFY_LAYERWISE_TOPK = 101
    SPARSIFY_LAYERWISE_PERCENTAGE = 102

class Compression:
    @classmethod
    def compress(self_class, data, config):
        if(not isinstance(data, HeterogeneousArray)):
            raise RuntimeError(f'Compression is only supported for objects of type {HeterogeneousArray.__name__}.')
        match config["compression_type"]:
            case CompressionType.NoneType:
                return data # no compression
            case CompressionType.QUANTIZE_PROBABILISTIC:
                return Quantization.quantizeProbabilistic(TODO)
            case CompressionType.SPARSIFY_LAYERWISE_TOPK:
                return Sparsification.sparsifyLayerwiseTopK(data, config["compression_k"])
            case CompressionType.SPARSIFY_LAYERWISE_PERCENTAGE:
                return Sparsification.sparsifyLayerwisePercentage(data, config["compression_percentage"])
            case _:
                raise NotImplementedError

# returns the indices of the K elements with highest absolute value
def getTopKIndices(arr, k):
    if(k > len(arr)):
        k = len(arr)
    sorted_idx = np.argpartition(np.absolute(arr), len(arr)-k)[-k:]
    return sorted_idx

class Sparsification(Compression):
    # Keep only the K highest values per layer, set the others to zero
    @classmethod
    def sparsifyLayerwiseTopK(self_class, data, k):
        layerwise_topk_indices = [getTopKIndices(layer.flatten(), k) for layer in data]

        masks = [np.zeros(layer.size) for layer in data]
        for idx, lti in enumerate(layerwise_topk_indices):
            masks[idx][lti] = 1

        sparse_data = data.sparsify(masks)
        return sparse_data

    # Keep the specified percentage of highest values per layer, set the others to zero
    @classmethod
    def sparsifyLayerwisePercentage(self_class, data, percentage):
        layerwise_percentage_indices = [getTopKIndices(
            layer.flatten(), math.ceil(layer.size*percentage)) for layer in data]

        masks = [np.zeros(layer.size) for layer in data]
        for idx, lpi in enumerate(layerwise_percentage_indices):
            masks[idx][lpi] = 1

        sparse_data = data.sparsify(masks)
        return sparse_data
