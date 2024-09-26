
class PartitioningUtils:
    @classmethod
    def partitionModelParameters(self_class, model_params, config):
        partitioned_params = dict()
        for addr, actor_idx in zip(config["neighbors"], config["neighbor_idx"]):
            partitioned_params[addr] = self_class.getParameterPartition(model_params, actor_idx, config["num_workers"])
        return partitioned_params

    @classmethod
    def getParameterPartition(self_class, model_params, actor_idx, num_actors):
        partition = model_params.take(self_class.getPartitionIndices(
            model_params.getLength(), actor_idx, num_actors))
        return partition

    @classmethod
    def getPartitionIndices(self_class, param_length, actor_idx, num_actors):
        # partition by layer_idx modulo num_actors equals actor_idx
        partition_indices = list(filter(
            lambda layer_idx: layer_idx % num_actors == actor_idx, range(param_length)))
        return partition_indices
