from network.IInitializationService import IInitializationService
import network.protos.Initialization_pb2 as Initialization_pb2
import network.protos.Initialization_pb2_grpc as Initialization_pb2_grpc
import network.protos.ModelUpdate_pb2 as ModelUpdate_pb2

from concurrent import futures
import grpc
import logging

class Servicer(Initialization_pb2_grpc.InitializeServicer):
    def __init__(self, callbacks):
        self.callbacks = callbacks

    # inform the actor about its public network address and its actor index in the system
    def InitIdentity(self, request, context):
        self.callbacks["InitIdentity"](request.net_id.ip_and_port,
            request.net_id.actor_idx, request.num_workers, request.seed)
        return ModelUpdate_pb2.Ack()

    # inform the actor which dataset to load/use
    def InitDataset(self, request, context):
        self.callbacks["InitDataset"](request.dataset_id,
            request.partition.partition_scheme_id, request.partition.partition_index,
            request.partition.dataset_seed, request.partition.partition_dirichlet_alpha)
        return ModelUpdate_pb2.Ack()

    # obtain the serialized model configuration and initialize the ML model
    def InitModel(self, request, context):
        self.callbacks["InitModel"](request.model_config, request.optimizer_config)
        return ModelUpdate_pb2.Ack()

    # initialize the weights of the model
    def InitModelParameters(self, request, context):
        self.callbacks["InitModelParameters"](request)
        return ModelUpdate_pb2.Ack()

    # inform the actor which learning strategy to use
    def InitStrategy(self, request, context):
        self.callbacks["InitStrategy"](
            request.num_fed_epochs,
            request.num_local_epochs,
            request.learning_type_id,
            request.learning_rate_local,
            request.learning_rate_global,
            request.sync_strat_spec.strategy_id,
            request.sync_strat_spec.percentage,
            request.sync_strat_spec.amount,
            request.sync_strat_spec.timeout,
            request.sync_strat_spec.allow_empty,
            request.compr_strat_spec.strategy_id,
            request.compr_strat_spec.k,
            request.compr_strat_spec.percentage,
            request.compr_strat_spec.precision,
            request.pdp_strat_spec.strategy_id,
            request.pdp_strat_spec.k)
        return ModelUpdate_pb2.Ack()

    # inform the actor about his neighboring actors and their network addresses
    def RegisterNeighbors(self, request, context):
        self.callbacks["RegisterNeighbors"](
            {nid.ip_and_port: nid.actor_idx for nid in request.net_id})
        return ModelUpdate_pb2.Ack()

    # stop the initialization phase and start the training phase
    def StartLearning(self, request, context):
        self.callbacks["StartLearning"]()
        return ModelUpdate_pb2.Ack()

class GRPCInitializationService(IInitializationService):
    def __init__(self, config):
        super().__init__(config)
        self.logger = logging.getLogger("network/GRPCInitializationService")
        self.logger.setLevel(config["log_level"])

    def waitForInitialization(self, callbacks):
        port = self.config["port"]
        num_threads = self.config["num_threads_server"]

        self.server = None
        callbacks["StartLearning"] = lambda: self.server.stop(None)

        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=num_threads))
        Initialization_pb2_grpc.add_InitializeServicer_to_server(
            Servicer(callbacks), self.server)
        self.server.add_insecure_port(f'[::]:{port}')
        self.server.start()
        self.logger.info(f'Server started, listening on {port}.')
        self.server.wait_for_termination()
        self.logger.info('Server terminated.')
