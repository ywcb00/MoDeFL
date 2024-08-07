from queue import Queue

class ModelUpdateMarket:
    def __init__(self, config):
        self.config = config
        self.model_updates = dict(
            [(addr, Queue()) for addr in config["neighbors"]])

    def put(self, elem, address):
        self.model_updates[address].put(elem)

    # block until we get one model update from each neighbor
    def getOneFromAll(self):
        result = dict()
        for addr, queue in self.model_updates.items():
            result[addr] = queue.get(block=True, timeout=None)
        return result

    # TODO: implement various wait/notify functionalities
    #       (e.g., ALL, MIN_K, TIMEOUT)
