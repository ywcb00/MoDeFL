from tffmodel.types.HeterogeneousArray import HeterogeneousArray
from tffmodel.types.HeterogeneousDenseArray import HeterogeneousDenseArray

from enum import Enum
import math
import numpy as np

class CompressionType(Enum):
    NoneType = 0 # do not apply compression
    # ===== Quantization =====
    QUANTIZE_PROBABILISTIC = 1 # similar to probabilistic quantization in https://arxiv.org/pdf/1610.05492
    # ===== Sparsification =====
    SPARSIFY_LAYERWISE_TOPK = 101
    SPARSIFY_LAYERWISE_PERCENTAGE = 102

class Compression:
    @classmethod
    def compress(self_class, data, config):
        if(not data):
            return None
        if(not isinstance(data, HeterogeneousArray)):
            raise RuntimeError(f'Compression is only supported for objects of type {HeterogeneousArray.__name__}.')
        match config["compression_type"]:
            case CompressionType.NoneType:
                return data # no compression
            case CompressionType.QUANTIZE_PROBABILISTIC:
                return Quantization.quantizeProbabilistic(data, config["compression_precision"], config["seed"])
            case CompressionType.SPARSIFY_LAYERWISE_TOPK:
                return Sparsification.sparsifyLayerwiseTopK(data, config["compression_k"])
            case CompressionType.SPARSIFY_LAYERWISE_PERCENTAGE:
                return Sparsification.sparsifyLayerwisePercentage(data, config["compression_percentage"])
            case _:
                raise NotImplementedError

    @classmethod
    def decompress(self_class, data):
        # NOTE: all relevant information for decompression must be stored in data.compression_properties
        if(not data):
            return None
        if(not isinstance(data, HeterogeneousArray)):
            raise RuntimeError(f'Compression is only supported for objects of type {HeterogeneousArray.__name__}.')
        compression_properties = data.getCompressionProperties()
        if(not compression_properties):
            return data
        match compression_properties["type"]:
            case CompressionType.NoneType:
                return data
            case CompressionType.QUANTIZE_PROBABILISTIC:
                return Quantization.dequantizeProbabilistic(data, compression_properties)
            case CompressionType.SPARSIFY_LAYERWISE_TOPK:
                return data
            case CompressionType.SPARSIFY_LAYERWISE_PERCENTAGE:
                return data
            case _:
                raise NotImplementedError

    @classmethod
    def compressDecompress(self_class, data, config):
        compressed_data = self_class.compress(data, config)
        decompressed_data = self_class.decompress(data)
        return decompressed_data

def getNumpyTypeForPrecision(precision):
    # NOTE: We only support predefined numpy uint types yet
    if(precision <= 8):
        return np.uint8
    elif(precision <= 16):
        return np.uint16
    elif(precision <= 32):
        return np.uint32
    elif(precision <= 64):
        return np.uint64
    else:
        raise NotImplementedError

class Quantization(Compression):
    # Quantize the data to the specified precision (bits) by probabilisticly rounding down/up
    @classmethod
    def quantizeProbabilistic(self_class, data, precision, seed):
        offset = data.min()
        scale = (2**precision) / (data.max() - data.min())
        quantized_data = data - offset # shift data to zero
        quantized_data *= scale # scale data to the range [0, 2^precision]

        probabilities = quantized_data.get() % 1
        rounding_indicator = HeterogeneousDenseArray(
            [np.random.binomial(1, parr, parr.shape) for parr in probabilities])
        quantized_data += rounding_indicator
        quantized_data.floor()
        quantized_data.setDType(getNumpyTypeForPrecision(precision))

        quantized_data.setCompressionProperties({
            "type": CompressionType.QUANTIZE_PROBABILISTIC,
            "offset": offset,
            "scale": scale,
            "source_dtype": data.getDType()
        })

        return quantized_data

    @classmethod
    def dequantizeProbabilistic(self_class, data, compression_properties):
        data.setDType(compression_properties["source_dtype"])
        data /= compression_properties["scale"]
        data += compression_properties["offset"]
        data.setCompressionProperties(None)
        return data

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
