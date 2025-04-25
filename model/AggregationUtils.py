import copy
import numpy as np

class AggregationUtils:
    # average all model parameters weighted by the specified averaging weights
    @classmethod
    def averageModelParameters(self_class, model_parameters, averaging_weights=None):
        result = np.average(model_parameters, weights=averaging_weights)
        return result

    # consensus-based federated averaging method from https://doi.org/10.1109/JIOT.2020.2964162
    # S. Savazzi, M. Nicoli and V. Rampa, "Federated Learning With Cooperating Devices:
    # A Consensus Approach for Massive IoT Networks," in IEEE Internet of Things Journal,
    # vol. 7, no. 5, pp. 4641-4654, May 2020, doi: 10.1109/JIOT.2020.2964162. keywords:
    # {Servers;Data models;Computational modeling;Artificial neural networks;Optimization;
    # Convergence;Internet of Things;5G and beyond networks;distributed signal processing;
    # federated learning;internet of Things},
    @classmethod
    def consensusbasedFedAvg(self_class, current_model_weights, received_model_weights, eps_t, alph_t):
        result = current_model_weights
        for addr, mu in received_model_weights.items():
            result += (mu - current_model_weights) * eps_t * alph_t[addr]
        return result

    # consensus-based federated averaging method w/ gradient exchange from
    # https://doi.org/10.1109/JIOT.2020.2964162
    # S. Savazzi, M. Nicoli and V. Rampa, "Federated Learning With Cooperating Devices:
    # A Consensus Approach for Massive IoT Networks," in IEEE Internet of Things Journal,
    # vol. 7, no. 5, pp. 4641-4654, May 2020, doi: 10.1109/JIOT.2020.2964162. keywords:
    # {Servers;Data models;Computational modeling;Artificial neural networks;Optimization;
    # Convergence;Internet of Things;5G and beyond networks;distributed signal processing;
    # federated learning;internet of Things},
    @classmethod
    def consensusbasedFedAvgWithGradExchange(self_class, current_model_weights,
        received_model_updates, eps_t, alph_t, mu_t, beta_t):
        model_parameters = current_model_weights
        adjusted_model_parameters = copy.deepcopy(model_parameters)
        for addr, (mp, pg) in received_model_updates.items():
            mp_update_term = (mp - current_model_weights) * eps_t * alph_t[addr]
            model_parameters += mp_update_term
            amp_update_term = pg * mu_t * beta_t[addr]
            adjusted_model_parameters += mp_update_term - amp_update_term
        return model_parameters, adjusted_model_parameters

    # normalized averaging method FedNova from https://dl.acm.org/doi/abs/10.5555/3495724.3496362
    # Jianyu Wang, Qinghua Liu, Hao Liang, Gauri Joshi, and H. Vincent Poor. 2020.
    # Tackling the objective inconsistency problem in heterogeneous federated optimization.
    # In Proceedings of the 34th International Conference on Neural Information Processing
    # Systems (NIPS '20). Curran Associates Inc., Red Hook, NY, USA, Article 638, 7611–7623.
    # NOTE: only a single gradient per device supported yet
    #   Hence, the model gradients are a list of gradients, not a list of a list of gradients.
    #   Thus, also the a_values are a list of values, not a list of vectors.
    @classmethod
    def fedNova(self_class, current_model_weights, model_gradients, aggregation_weights,
        tau_eff, lr_server, a_values):
        normalized_gradients = [mg * (av / abs(av)) for mg, av in zip(model_gradients, a_values)]
        model_update_term = None
        aw_sum = np.sum(aggregation_weights)
        for ng, aw in zip(normalized_gradients, aggregation_weights):
            if(model_update_term == None):
                model_update_term = ng * (aw / aw_sum)
            else:
                model_update_term += ng * (aw / aw_sum)
        model_update_term *= tau_eff * lr_server
        result = current_model_weights - model_update_term
        return result
