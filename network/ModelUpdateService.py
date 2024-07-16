
import network.protos.ModelUpdate_pb2 as ModelUpdate_pb2
import network.protos.ModelUpdate_pb2_grpc as ModelUpdate_pb2_grpc

import asyncio
from concurrent import futures
import grpc
import logging

class Servicer(ModelUpdate_pb2_grpc.ModelUpdateServicer):
    def __init__(self, callbacks):
        self.callbacks = callbacks

    def TransferModelUpdate(self, request, context):
        self.callback["TransferModelUpdate"](request.layer_weights,
            request.ip_and_port)
        return ModelUpdate_pb2.Ack()

class ModelUpdateService:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("network/ModelUpdateService")
        self.logger.setLevel(config["log_level"])

    def startServer(self, callbacks):
        port = self.config["port"]
        num_threads = 4 # TODO: define the maximal number of thread w.r.t. the CPU

        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=num_threads))
        ModelUpdate_pb2_grpc.add_ModelUpdateServicer_to_server(
            Servicer(callbacks), self.server)
        self.server.add_insecure_port(f'[::]:{port}')
        self.server.start()
        self.logger.info(f'Server started, listening on {port}.')

    def stopServer(self):
        self.server.stop(None)
        self.server.wait_for_termination()
        self.logger.info('Server terminated.')
