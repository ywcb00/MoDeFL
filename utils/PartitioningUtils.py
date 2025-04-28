from tffmodel.types.HeterogeneousDenseArray import HeterogeneousDenseArray

from enum import Enum
import numpy as np

class ModelPartitioningStrategy(Enum):
    LAYERWISE = 1
    BALANCED = 2

# utility methods for partitioning data (and model parameters) and joining partitions
class PartitioningUtils:
    # partition the model parameters to one partition per actor
    @classmethod
    def partitionModelParameters(self_class, model_params, config):
        # NOTE: we assume a fully-connected network topology as we do not
        #   forward updates or replicate partitions
        partitioned_params = dict()
        for addr, actor_idx in zip(config["neighbors"], config["neighbor_idx"]):
            partitioned_params[addr] = self_class.getParameterPartition(
                model_params, actor_idx, config)
        return partitioned_params

    @classmethod
    def getParameterPartition(self_class, model_params, actor_idx, config):
        match config["model_partitioning_strategy"]:
            case ModelPartitioningStrategy.LAYERWISE:
                return self_class.getParameterPartitionLayerwise(
                    model_params, actor_idx, config["num_workers"])
            case ModelPartitioningStrategy.BALANCED:
                return self_class.getParameterPartitionBalanced(
                    model_params, actor_idx, config["num_workers"])
            case _:
                raise NotImplementedError

    # obtain the partition of the model parameters according to the actor index
    # layer-wise partitioning
    @classmethod
    def getParameterPartitionLayerwise(self_class, model_params, actor_idx, num_actors):
        n_layers = model_params.getNumLayers()
        layer_indices = [layer_idx for layer_idx in range(n_layers) if(layer_idx % num_actors == actor_idx)]
        partition_layers = model_params.take(layer_indices)
        partition = model_params.__class__(partition_layers)
        return partition

    # obtain the partition of the model parameters according to the actor index
    # evenly balanced parameters among actors; layer-agnostic
    @classmethod
    def getParameterPartitionBalanced(self_class, model_params, actor_idx, num_actors):
        # NOTE: Assuming the same data type among all layers
        model_params_flattened = model_params.getFlattened()
        n = model_params_flattened.size
        n_part = (n // num_actors)

        distribute_remainder = lambda idx: 1 if idx < (n % num_actors) else 0

        offset = np.sum([distribute_remainder(a_idx) for a_idx in range(actor_idx)])

        from_idx = int((n_part * actor_idx) + offset)
        to_idx = int((n_part * (actor_idx+1)) + offset + distribute_remainder(actor_idx))
        partition = model_params.__class__([model_params_flattened[from_idx : to_idx]])
        return partition

    @classmethod
    def joinParameterPartitions(self_class, partition_dict, shape_layer_arrays, config):
        match config["model_partitioning_strategy"]:
            case ModelPartitioningStrategy.LAYERWISE:
                return self_class.joinParameterPartitionsLayerwise(
                    partition_dict, shape_layer_arrays, config["num_workers"])
            case ModelPartitioningStrategy.BALANCED:
                return self_class.joinParameterPartitionsBalanced(
                    partition_dict, shape_layer_arrays)

    # join model parameter partitions into a full set of model parameters (dense)
    # layer-wise partitioning
    @classmethod
    def joinParameterPartitionsLayerwise(self_class, partition_dict,
        shape_layer_arrays, n_actors):
        n_layers = shape_layer_arrays.getNumLayers()
        joined_partitions = [partition_dict[layer_idx % n_actors].get()[layer_idx // n_actors]
            for layer_idx in range(n_layers)]
        parameters = HeterogeneousDenseArray(joined_partitions)
        return parameters

    # join model parameter partitions into a full set of model parameters (dense)
    # evenly balanced parameters among actors; layer-agnostic
    @classmethod
    def joinParameterPartitionsBalanced(self_class, partition_dict, shape_layer_arrays):
        joined_partitions = np.concatenate([partition_dict[part_idx].get()[0]
            for part_idx in sorted(partition_dict.keys())])
        parameters = HeterogeneousDenseArray.fromFlattened(
            joined_partitions, shape_layer_arrays)
        return parameters
