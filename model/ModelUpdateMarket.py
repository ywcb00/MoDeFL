import asyncio
from enum import Enum
from queue import SimpleQueue

class ModelUpdateStrategy(Enum):
    ONE_FROM_ALL = 1
    AVAILABLE = 2
    MIN_ONE_FROM_ALL = 3
    ONE_FROM_MIN_PERCENT = 4
    MIN_N = 5
    ONE_FROM_ALL_T = 6

class ModelUpdateMarket:
    def __init__(self, config):
        self.config = config
        self.config.setdefault("model_update_strategy", ModelUpdateStrategy.ONE_FROM_ALL)
        self.model_updates = dict(
            [(addr, SimpleQueue()) for addr in config["neighbors"]])

    def get(self):
        match self.config["model_update_strategy"]:
            case ModelUpdateStrategy.ONE_FROM_ALL:
                return self.getOneFromAll()
            case ModelUpdateStrategy.AVAILABLE:
                return self.getAvailable()
            case ModelUpdateStrategy.MIN_ONE_FROM_ALL:
                return self.getAtLeastOneFromAll()
            case ModelUpdateStrategy.ONE_FROM_MIN_PERCENT:
                return self.getOneFromAtLeastPercentage()
            case ModelUpdateStrategy.MIN_N:
                return self.getAtLeastK()
            case ModelUpdateStrategy.ONE_FROM_ALL_T:
                return self.getOneFromAllTimeout()
            case _:
                raise NotImplementedError

    def put(self, elem, address):
        self.model_updates[address].put(elem)

    # block until we get one model update from each neighbor
    def getOneFromAll(self):
        result = dict()
        for addr, queue in self.model_updates.items():
            result[addr] = queue.get(block=True, timeout=None)
        return result

    # do not block, get all model updates that are currently available
    def getAvailable(self):
        result = dict()
        for addr, queue in self.model_updates.items():
            result[addr] = list()
            try:
                while True:
                    result[addr].append(queue.get(block=False))
            except queue.Empty:
                pass # do nothing, queue is empty, continue with next queue
        return result

    # get all available models but at least one from each neighbor
    def getAtLeastOneFromAll(self):
        result = dict()
        for addr, queue in self.model_updates.items():
            result[addr] = [queue.get(block=True, timeout=None)]
        for addr, queue in self.model_updates.items():
            try:
                while True:
                    result[addr].append(queue.get(block=False))
            except queue.Empty:
                pass # do nothing, queue is empty, continue with next queue
        return result

    # poll until we got one model from at least the specified proportion of neighbors
    def getOneFromAtLeastPercentage(self):
        percentage = self.config.setdefault("model_update_strat_percentage", 0.5)
        amount = len(self.model_updates) * percentage
        result = dict()
        while(len(result) < amount):
            remaining_addresses = set(self.model_updates.keys()).difference(set(result.keys()))
            for addr in remaining_addresses:
                queue = self.model_updates[addr]
                try:
                    result[addr] = queue.get(block=False)
                except queue.Empty:
                    pass # do nothing, queue is empty, continue with next queue
        return result

    # poll until we got at least the specified amount of model updates
    def getAtLeastK(self):
        amount = self.config.setdefault("model_update_strat_amount", len(self.model_updates) // 2)
        result = dict()
        while(amount > 0):
            for addr, queue in self.model_updates.items():
                try:
                    result.setdefault(addr, list()).append(queue.get(block=False))
                    amount -= 1
                except queue.Empty:
                    pass # do nothing, queue is empty, continue with next queue
        return result

    # try to get one from each neighbor until we reach the specified timeout (seconds)
    def getOneFromAllTimeout(self, timeout):
        timeout = self.config.setdefault("model_update_strat_timeout", 3)
        async def getOneTimeout(addr, queue, timeout):
            try:
                mod_update = (addr, queue.get(block=True, timeout=timeout))
            except queue.Empty:
                mod_update = None
            return mod_update
        async def spawnAndCollect(timeout):
            res = await asyncio.gather(*[getOneTimeout(addr, queue, timeout)
                for addr, queue in self.model_updates.items()])
            res = dict([elem for elem in res if elem != None])
            return res

        result = asyncio.run(spawnAndCollect(timeout))
        return result
