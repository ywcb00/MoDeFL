
from Actor import Actor
from Initiator import Initiator
from model.LearningStrategy import LearningType
from tffdataset.DatasetUtils import DatasetID
from tffdataset.FedDataset import PartitioningScheme

from enum import Enum
import getopt
import logging
import sys

config = {
    "seed": 13,

    "force_load": False,

    "dataset_id": DatasetID.Mnist,

    "train_response_path": "./data/train_response.csv",
    "val_response_path": "./data/val_response.csv",
    "test_response_path": "./data/test_response.csv",

    "part_scheme": PartitioningScheme.ROUND_ROBIN,
    "num_workers": 4,

    "learning_type": LearningType.DFLv1,

    "num_train_rounds": 10,

    "log_dir": "./log/training",
    "log_level": logging.DEBUG,
}

class ExecType(Enum):
    INITIATOR = 1,
    ACTOR = 2

def printHelp(program_name):
    print("Initiator usage:", program_name, "--initiate")
    print("Actor usage:", program_name, "--act", "[--forceload]")

def main(argv):
    logging.basicConfig()
    logger = logging.getLogger("main.py")
    logger.setLevel(config["log_level"])

    exec_type = None

    try:
        opts, args = getopt.getopt(argv[1:], "hlia", ["help", "forceload", "initiate", "act"])
    except getopt.GetoptError:
        print("Wrong usage.")
        printHelp(argv[0])
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            printHelp(argv[0])
            sys.exit()
        if opt in ("-l", "--forceload"):
            config["force_load"] = True
        if opt in ("-i", "--initiate"):
            exec_type = ExecType.INITIATOR
        elif opt in ("-a", "--act"):
            exec_type = ExecType.ACTOR

    match exec_type:
        case ExecType.INITIATOR:
            logger.info("Starting Initiator")
            initiator = Initiator(config)
            initiator.initiate()
            sys.exit()
        case ExecType.ACTOR:
            logger.info("Starting Actor")
            actor = Actor(config)
            actor.initialize()
        case _:
            print("Wrong usage.")
            printHelp(argv[0])
            sys.exit(2)

if __name__ == '__main__':
    main(sys.argv)
