from model.ModelUpdateMarket import ModelUpdateMarket

from abc import ABC, abstractmethod

class IDFLStrategy(ABC):
    def __init__(self, config, keras_model):
        self.config = config
        self.keras_model = keras_model
        self.model_update_market = ModelUpdateMarket(self.config)

    def startServer(self):
        pass

    def stopServer(self):
        pass

    def fitLocal(self):
        pass

    def broadcast(self):
        pass

    def aggregate(self):
        pass
