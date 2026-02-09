from network.GRPCInitializationService import GRPCInitializationService
from network.GRPCModelUpdateService import GRPCModelUpdateService

from enum import Enum
import numpy as np

class NetworkServiceType(Enum):
    GRPC = 1

class NetworkUtils:
    # obtain the neighbor addresses for a particular actor based on the adjacency matrix
    @classmethod
    def getNeighborIdentities(self_class, current_addr, addresses, adj_mat):
        current_idx = addresses.index(current_addr)
        adj_row = adj_mat[current_idx]
        neighbor_idx = np.nonzero(adj_row)[0]
        neighbor_addresses = np.array(addresses)[neighbor_idx]
        neighbor_identities = {nix: naddr
            for nix, naddr in zip(neighbor_idx, neighbor_addresses) if naddr != current_addr}
        return neighbor_identities

    @classmethod
    def getInitializationService(self_class, config):
        match config["networkservice_type"]:
            case NetworkServiceType.GRPC:
                return GRPCInitializationService(config)
            case _:
                raise NotImplementedError

    @classmethod
    def getModelUpdateService(self_class, config):
        match config["networkservice_type"]:
            case NetworkServiceType.GRPC:
                return GRPCModelUpdateService(config)
            case _:
                raise NotImplementedError
