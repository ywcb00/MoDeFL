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
