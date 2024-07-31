import numpy as np

class NetworkUtils:
    @classmethod
    def getNeighborAddresses(self_class, current_addr, addresses, adj_mat):
        current_idx = addresses.index(current_addr)
        adj_row = adj_mat[current_idx]
        neighbor_idx = np.nonzero(adj_row)[0]
        neighbor_addresses = np.array(addresses)[neighbor_idx]
        neighbor_addresses = [naddr for naddr in neighbor_addresses if naddr != current_addr]
        return neighbor_addresses
