from model.SerializationUtils import SerializationUtils
import network.protos.ModelUpdate_pb2 as ModelUpdate_pb2
import network.protos.ModelUpdate_pb2_grpc as ModelUpdate_pb2_grpc

from concurrent import futures
import grpc
import logging

class Servicer(ModelUpdate_pb2_grpc.ModelUpdateServicer):
    def __init__(self, callbacks):
        self.callbacks = callbacks

    def TransferModelUpdate(self, request, context):
        self.callbacks["TransferModelUpdate"](request.weights.weights,
            request.gradient.gradient, request.ip_and_port)
        return ModelUpdate_pb2.Ack()

    def EvaluateModel(self, request, context):
        eval_metrics = self.callbacks["EvaluateModel"](request.weights)
        return ModelUpdate_pb2.EvaluationMetrics(
            metrics=[ModelUpdate_pb2.Metric(key=key, value=val)
                for key, val in eval_metrics.items()])

    def AllowTermination(self, request, context):
        self.callbacks["AllowTermination"](request.ip_and_port)
        return ModelUpdate_pb2.Ack()

class ModelUpdateService:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("network/ModelUpdateService")
        self.logger.setLevel(config["log_level"])

    def startServer(self, callbacks):
        port = self.config["port"]
        num_threads = self.config["num_threads_server"]

        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=num_threads))
        ModelUpdate_pb2_grpc.add_ModelUpdateServicer_to_server(
            Servicer(callbacks), self.server)
        self.server.add_insecure_port(f'[::]:{port}')
        self.server.start()
        self.logger.info(f'Server started, listening on {port}.')

    def stopServer(self):
        self.server.stop(grace=1)

    def waitForTermination(self):
        self.server.wait_for_termination()
        self.logger.info('Server terminated.')
