from abc import ABC, abstractmethod

class IModelUpdateService(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def startServer(self, callbacks: dict):
        pass

    @abstractmethod
    def stopServer(self):
        pass

    @abstractmethod
    def waitForTermination(self):
        pass
