import copy
import numpy as np

class AggregationUtils:
    @classmethod
    def averageModelWeights(self_class, model_weights, averaging_weights=None):
        result = np.average(model_weights, weights=averaging_weights)
        return result

    @classmethod
    def consensusbasedFedAvg(self_class, current_model_weights, model_updates, eps_t, alph_t):
        result = current_model_weights
        for addr, mu in model_updates.items():
            result += (mu - current_model_weights) * eps_t * alph_t[addr]
        return result

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

    # NOTE: only a single gradient per device supported yet
    #   Hence, the model gradients are a list of gradients, not a list of a list of gradients.
    #   Thus, also the a_values are a list of values, not a list of vectors.
    @classmethod
    def fedNova(self_class, current_model_weights, model_gradients, aggregation_weights,
        tau_eff, lr_client, a_values):
        normalized_gradients = [mg * (av / abs(av)) for mg, av in zip(model_gradients, a_values)]
        model_update_term = None
        aw_sum = np.sum(aggregation_weights)
        for ng, aw in zip(normalized_gradients, aggregation_weights):
            if(model_update_term == None):
                model_update_term = ng * (aw / aw_sum)
            else:
                model_update_term += ng * (aw / aw_sum)
        model_update_term *= tau_eff * lr_client
        result = current_model_weights - model_update_term
        return result
