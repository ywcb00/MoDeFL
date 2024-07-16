from abc import ABC, abstractmethod
from queue import Queue

class IDFLModel(ABC):
    def __init__(self, config, keras_model):
        self.config = config
        self.keras_model = keras_model
        # TODO: create a separate data structure for storing model updates with wait/notify functionality
        self.model_updates = dict(
            [(addr, Queue()) for addr in config["neighbors"]])

    def startServer(self):
        pass

    def stopServer(self):
        pass

    def fitLocal(self):
        pass

    def aggregate(self):
        pass
