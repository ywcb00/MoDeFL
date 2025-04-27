import json
import os

# class for incrementally storing information about the communicated messages and saving them as file
class CommunicationLogger:
    logdict = dict()

    @classmethod
    def logMultiple(self_class, from_addr, to_addresses, msg_properties):
        for to_addr in to_addresses:
            self_class.log(from_addr, to_addr, msg_properties)

    @classmethod
    def log(self_class, from_addr, to_addr, msg_properties):
        self_class.logdict.setdefault(from_addr, {}).setdefault(to_addr, []).append(msg_properties)

    @classmethod
    def write(self_class, log_path):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(f'{log_path}.json', 'w') as outfile:
            json.dump(self_class.logdict, outfile)
