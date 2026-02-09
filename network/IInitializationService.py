from abc import ABC, abstractmethod

class IInitializationService(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def waitForInitialization(self, callbacks: dict):
        pass
