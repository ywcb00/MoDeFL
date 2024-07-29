import numpy as np

class AggregationUtils:
    @classmethod
    def averageModelWeights(self_class, model_weights):
        result = [np.average([mw[layer_idx] for mw in model_weights], axis=0) for layer_idx in range(len(model_weights[0]))]
        return result

    @classmethod
    def consensusbasedFedAvg(self_class, current_model_weights, model_updates, eps_t, alph_t):
        result = current_model_weights
        for addr, mu in model_updates.items():
            # TODO: search for a faster solution to this
            model_update_term = ([eps_t * alph_t[addr] * (mulw - clw) for clw, mulw in zip(current_model_weights, mu)])
            result = [rlw + mutlw for rlw, mutlw in zip(result, model_update_term)]
        return result
