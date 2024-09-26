from model.DFLv1Strategy import DFLv1Strategy
from model.DFLv2Strategy import DFLv2Strategy
from model.DFLv3Strategy import DFLv3Strategy
from model.DFLv4Strategy import DFLv4Strategy
from model.DFLv5Strategy import DFLv5Strategy
from model.DFLv6Strategy import DFLv6Strategy
from model.DFLv7Strategy import DFLv7Strategy
from model.DFLv8Strategy import DFLv8Strategy

from enum import Enum

class LearningType(Enum):
    DFLv1 = 1
    DFLv2 = 2
    DFLv3 = 3
    DFLv4 = 4
    DFLv5 = 5
    DFLv6 = 6
    DFLv7 = 7
    DFLv8 = 8

class LearningStrategy:
    @classmethod
    def getStrategy(self_class, config, keras_model, dataset):
        match config["learning_type"]:
            case LearningType.DFLv1:
                return DFLv1Strategy(config, keras_model, dataset)
            case LearningType.DFLv2:
                return DFLv2Strategy(config, keras_model, dataset)
            case LearningType.DFLv3:
                return DFLv3Strategy(config, keras_model, dataset)
            case LearningType.DFLv4:
                return DFLv4Strategy(config, keras_model, dataset)
            case LearningType.DFLv5:
                return DFLv5Strategy(config, keras_model, dataset)
            case LearningType.DFLv6:
                return DFLv6Strategy(config, keras_model, dataset)
            case LearningType.DFLv7:
                return DFLv7Strategy(config, keras_model, dataset)
            case LearningType.DFLv8:
                return DFLv8Strategy(config, keras_model, dataset)
            case _:
                raise NotImplementedError
