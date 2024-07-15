
import network.protos.Initialization_pb2 as Initialization_pb2
import network.protos.Initialization_pb2_grpc as Initialization_pb2_grpc
import network.protos.ModelUpdate_pb2 as ModelUpdate_pb2

from concurrent import futures
import grpc
import logging

class Servicer(Initialization_pb2_grpc.InitializeServicer):
    def __init__(self, callbacks):
        self.callbacks = callbacks

    def InitDataset(self, request, context):
        self.callbacks["InitDataset"](request.dataset_id,
            request.partition_scheme_id, request.partition_seed)
        return ModelUpdate_pb2.Ack()

    def InitModel(self, request, context):
        self.callbacks["InitModel"](request.model)
        return ModelUpdate_pb2.Ack()

    def InitModelWeights(self, request, context):
        self.callbacks["InitModelWeights"](request.layer_weights)
        return ModelUpdate_pb2.Ack()

    def InitLearningStrategy(self, request, context):
        self.callbacks["InitLearningStrategy"](request.learning_type_id)
        return ModelUpdate_pb2.Ack()

    def RegisterNeighbors(self, request, context):
        self.callbacks["RegisterNeighbors"](request.ip_and_port)
        return ModelUpdate_pb2.Ack()

    def StartLearning(self, request, context):
        self.callbacks["StartLearning"]()
        return ModelUpdate_pb2.Ack()

class InitializationService:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("network/InitializationService")
        self.logger.setLevel(config["log_level"])

    def waitForInitialization(self, callbacks):
        port = self.config["port"]
        num_threads = 4 # TODO: define the maximal number of thread w.r.t. the CPU

        self.server = None
        callbacks["StartLearning"] = lambda: self.server.stop(None)

        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=num_threads))
        Initialization_pb2_grpc.add_InitializeServicer_to_server(
            Servicer(callbacks), self.server)
        self.server.add_insecure_port(f'[::]:{port}')
        self.server.start()
        self.logger.info(f'Server started, listening on {port}.')
        self.server.wait_for_termination()
        self.logger.info(f'Server terminated.')
