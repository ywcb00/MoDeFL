import numpy as np

class AggregationUtils:
    @classmethod
    def averageModelWeights(self_class, model_weights):
        result = np.average(model_weights)
        return result

    @classmethod
    def consensusbasedFedAvg(self_class, current_model_weights, model_updates, eps_t, alph_t):
        result = current_model_weights
        for addr, mu in model_updates.items():
            model_update_term = (mu - current_model_weights) * eps_t * alph_t[addr]
            result += model_update_term
        return result
