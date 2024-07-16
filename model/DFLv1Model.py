from model.IDFLModel import IDFLModel
from network.ModelUpdateService import ModelUpdateService

import logging

# TODO: rename to DFLv1Strategy
class DFLv1Model(IDFLModel):
    def __init__(self, config, keras_model):
        super().__init__(config, keras_model)
        self.logger = logging.getLogger("model/DFLv1Model")
        self.logger.setLevel(config["log_level"])

    def startServer(self):
        def transferModelUpdateCallback(self, weights, address):
            self.model_updates[address].put(weights)

        callbacks = {"TransferModelUpdate": transferModelUpdateCallback}

        self.model_update_service = ModelUpdateService(self.config)
        self.model_update_service.startServer(callbacks)
        print(self.model_updates)

    def stopServer(self):
        self.model_update_service.stopServer()

    def fitLocal(self, dataset):
        self.logger.info("Fitting local model.")
        # TODO: change number of epochs for fit (to 1?)
        self.keras_model.fit(dataset)
        print(self.keras_model.getWeights())

