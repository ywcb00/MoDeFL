from model.DFLv1Strategy import DFLv1Strategy

import logging
import numpy as np
import tensorflow as tf

@tf.keras.utils.register_keras_serializable(package='Custom', name="FedProxReg")
class FedProxRegularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, mu, global_w):
        self.mu = mu
        if(isinstance(global_w, dict)):
            # NOTE: global weights are passed as dictionary when called by from_config (i.e., deserializing)
            global_w = np.array(global_w['config']['value'])
        self.global_w = global_w
    def __call__(self, w):
        return (self.mu / 2) * tf.math.square(tf.norm(w - self.global_w))
    def get_config(self):
        return {"mu": self.mu, "global_w": self.global_w}
    # @classmethod
    # def from_config(self_class, config):
    #     return self_class(config["mu"], config["global_w"])

class DFLv4Strategy(DFLv1Strategy):
    def __init__(self, config, keras_model, dataset):
        super().__init__(config, keras_model, dataset)
        self.logger = logging.getLogger("model/DFLv4Strategy")
        self.logger.setLevel(config["log_level"])

    def fitLocal(self):
        # TODO: set the hyperparameter mu (regularization magnitude)
        mu = 0.1
        regularizers = list()
        for layer in self.keras_model.getModel().layers:
            if(hasattr(layer, 'kernel_regularizer')):
                regularizers.append(FedProxRegularizer(mu, layer.get_weights()[0]))

        self.keras_model.addKernelRegularizers(regularizers)
        self.logger.debug(f'Added {len(regularizers)} regularizers to the model.')

        train_metrics = super().fitLocal()
        return train_metrics
