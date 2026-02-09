from Actor import Actor
from Initiator import Initiator
from utils.ConfigurationUtils import ConfigurationUtils

from enum import Enum
import getopt
import logging
import sys

class ExecType(Enum):
    INITIATOR = 1,
    ACTOR = 2

def printHelp(program_name):
    print("Initiator usage:", program_name, "--initiate", "[--addr_file=<PATH>]", "[--adj_file=<PATH>]")
    print("Actor usage:", program_name, "--act", "--port=<PORT>")

def main(argv):

    # ===== Load the configuration =====
    # default config if not specified otherwise
    config = ConfigurationUtils.DEFAULT_CONFIG

    exec_type = None

    logging.basicConfig()
    logger = logging.getLogger("main.py")
    logger.setLevel(config["log_level"])

    try:
        opts, args = getopt.getopt(argv[1:], "hiap:c:",
            ["help", "initiate", "act", "port=", "config=", "addr_file=", "adj_file=",
                *ConfigurationUtils.CLI_OPTIONS])
    except getopt.GetoptError:
        print("Wrong usage.")
        printHelp(argv[0])
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            printHelp(argv[0])
            sys.exit()
        elif opt in ("-c", "--config") and arg:
            logger.info(f'Loading configuration from {arg}.')
            config.update(ConfigurationUtils.loadConfig(arg))
    for opt, arg in opts:
        if opt in ("-i", "--initiate"):
            exec_type = ExecType.INITIATOR
        elif opt in ("-a", "--act"):
            exec_type = ExecType.ACTOR
        else:
            config = ConfigurationUtils.parseCLIOption(config, opt, arg)

    logger.setLevel(config["log_level"])

    config = ConfigurationUtils.convertConfigTypes(config)

    # ===== Start the Initiator or actor =====
    match exec_type:
        case ExecType.INITIATOR:
            logger.info("Starting Initiator")
            initiator = Initiator(config)
            initiator.initiate()
            sys.exit()
        case ExecType.ACTOR:
            logger.info("Starting Actor")
            actor = Actor(config)
            actor.run()
            sys.exit()
        case _:
            print("Wrong usage.")
            printHelp(argv[0])
            sys.exit(2)

if __name__ == '__main__':
    main(sys.argv)
