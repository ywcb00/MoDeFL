import numpy as np

class PartitioningUtils:
    @classmethod
    def partitionModelParameters(self_class, model_params, config):
        partitioned_params = dict()
        for addr, actor_idx in zip(config["neighbors"], config["neighbor_idx"]):
            partitioned_params[addr] = self_class.getParameterPartition(
                model_params, actor_idx, config["num_workers"])
        return partitioned_params

    @classmethod
    def getParameterPartition(self_class, model_params, actor_idx, num_actors):
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
    def joinParameterPartitions(self_class, partition_dict):
        joined_partitions = np.concatenate([partition_dict[part_idx].get()[0]
            for part_idx in sorted(partition_dict.keys())])
        return joined_partitions
