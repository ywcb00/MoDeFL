from enum import Enum
import random

class PartialDeviceParticipationStrategy(Enum):
    NoneStrategy = 0 # do not apply partial device participation
    RandomK = 1

# static class for partial device participation strategies
class PartialDeviceParticipation:
    # proxy the call according to the specified strategy
    @classmethod
    def getNeighbors(self_class, config):
        match config["partialdeviceparticipation_strategy"]:
            case PartialDeviceParticipationStrategy.NoneStrategy:
                return config["neighbors"]
            case PartialDeviceParticipationStrategy.RandomK:
                return self_class.randomK(config["neighbors"],
                    config["partialdeviceparticipation_k"])
            case _:
                raise NotImplementedError

    # randomly select the specified number of neighbors
    @classmethod
    def randomK(self_class, neighbor_candidates, k):
        # TODO: define seed for randomness, but cannot define here as we do not
        #   want to select the same set every round
        neighbors = random.sample(neighbor_candidates, k)
        return neighbors
