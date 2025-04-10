import asyncio
from enum import Enum
import math
from queue import SimpleQueue

from model.SerializationUtils import SerializationUtils

class SynchronizationStrategy(Enum):
    ONE_FROM_EACH = 1
    AVAILABLE = 2
    MIN_ONE_FROM_EACH = 3
    ONE_FROM_MIN_PERCENT = 4
    MIN_K = 5
    ONE_FROM_EACH_T = 6

class ModelUpdateMarket:
    def __init__(self, config):
        self.config = config
        self.config.setdefault("synchronization_strategy", SynchronizationStrategy.ONE_FROM_EACH)
        self.model_updates = dict(
            [(addr, SimpleQueue()) for addr in config["neighbors"]])

    def get(self):
        model_updates_dict = None
        match self.config["synchronization_strategy"]:
            case SynchronizationStrategy.ONE_FROM_EACH:
                model_updates_dict = self.getOneFromAll()
            case SynchronizationStrategy.AVAILABLE:
                model_updates_dict = self.getAvailable()
            case SynchronizationStrategy.MIN_ONE_FROM_EACH:
                model_updates_dict = self.getAtLeastOneFromAll()
            case SynchronizationStrategy.ONE_FROM_MIN_PERCENT:
                model_updates_dict = self.getOneFromAtLeastPercentage()
            case SynchronizationStrategy.MIN_K:
                model_updates_dict = self.getAtLeastK()
            case SynchronizationStrategy.ONE_FROM_EACH_T:
                model_updates_dict = self.getOneFromAllTimeout()
            case _:
                raise NotImplementedError

        model_updates_dict = {key: val for key, val in model_updates_dict.items() if val}
        return model_updates_dict

    def putUpdate(self, update, address):
        weights = SerializationUtils.deserializeModelWeights(update.weights.weights)
        gradient = SerializationUtils.deserializeGradient(update.gradient.gradient)
        market_element = None if (not weights and not gradient) else {
            "weights": weights, "gradient": gradient,
            "aggregation_weight": update.aggregation_weight
        }
        self.put(market_element, address)

    def putSparseUpdate(self, update, address):
        # TODO: allow for sparse model weights
        weights = SerializationUtils.deserializeModelWeights(update.weights.weights)
        gradient = SerializationUtils.deserializeSparseGradient(update.gradient.gradient)
        market_element = None if (not weights and not gradient) else {
            "weights": weights, "gradient": gradient,
            "aggregation_weight": update.aggregation_weight
        }
        self.put(market_element, address)

    def put(self, elem, address):
        self.model_updates[address].put(elem)

    # block until we get one model update from each neighbor
    def getOneFromAll(self):
        result = dict()
        for addr, queue in self.model_updates.items():
            update = queue.get(block=True, timeout=None)
            while(not self.config["synchronization_strat_allowempty"] and not update):
                update = queue.get(block=True, timeout=None)
            result[addr] = update
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
            update = [queue.get(block=True, timeout=None)]
            while(not self.config["synchronization_strat_allowempty"] and not update[0]):
                update = [queue.get(block=True, timeout=None)]
            result[addr] = update
        for addr, queue in self.model_updates.items():
            try:
                while True:
                    result[addr].append(queue.get(block=False))
            except queue.Empty:
                pass # do nothing, queue is empty, continue with next queue
        return result

    # poll until we got one model update from at least the specified proportion of neighbors
    def getOneFromAtLeastPercentage(self):
        percentage = self.config.setdefault("synchronization_strat_percentage", 0.5)
        amount = math.ceil(len(self.model_updates) * percentage)
        result = dict()
        while(len(result) < amount):
            remaining_addresses = set(self.model_updates.keys()).difference(set(result.keys()))
            for addr in remaining_addresses:
                queue = self.model_updates[addr]
                try:
                    update = queue.get(block=False)
                    if(self.config["synchronization_strat_allowempty"] or update):
                        result[addr] = update
                except queue.Empty:
                    pass # do nothing, queue is empty, continue with next queue
        return result

    # poll until we got at least the specified amount of model updates
    def getAtLeastK(self):
        amount = self.config.setdefault("synchronization_strat_amount", len(self.model_updates) // 2)
        result = dict()
        while(amount > 0):
            for addr, queue in self.model_updates.items():
                try:
                    update = queue.get(block=False)
                    if(self.config["synchronization_strat_allowempty"] or update):
                        result.setdefault(addr, list()).append(update)
                        amount -= 1
                except queue.Empty:
                    pass # do nothing, queue is empty, continue with next queue
        return result

    # try to get one from each neighbor until we reach the specified timeout (seconds) (allow empty)
    def getOneFromAllTimeout(self, timeout):
        timeout = self.config.setdefault("synchronization_strat_timeout", 3)
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
