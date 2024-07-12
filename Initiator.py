
import network.protos.Initialization_pb2 as Initialization_pb2
import network.protos.Initialization_pb2_grpc as Initialization_pb2_grpc
import network.protos.ModelUpdate_pb2 as ModelUpdate_pb2
from tffdataset.DatasetUtils import getDatasetElementSpec
from tffmodel.KerasModel import KerasModel

import asyncio
import grpc

class Initiator:
    def __init__(self, config):
        self.config = config

    def initiate(self):
        with grpc.insecure_channel("localhost:50051") as channel:
            stub = Initialization_pb2_grpc.InitializeStub(channel)

            stub.InitDataset(Initialization_pb2.Dataset(
                dataset_id=self.config["dataset_id"].value,
                partition_scheme_id=self.config["part_scheme"].value,
                partition_seed=self.config["seed"]))

            stub.InitModel(Initialization_pb2.Model(model=None))

            init_weights = KerasModel.createKerasModelElementSpec(
                getDatasetElementSpec(self.config), self.config).get_weights()
            init_weights_serialized = [layer_weights.tobytes() for layer_weights in init_weights]
            stub.InitModelWeights(ModelUpdate_pb2.ModelWeights(
                layer_weights=init_weights_serialized, ip_and_port="initialization"))

            stub.InitLearningStrategy(Initialization_pb2.LearningStrategy(
                learning_type_id=self.config["learning_type"].value))
            try:
                stub.StartLearning(ModelUpdate_pb2.Ack())
            except grpc._channel._InactiveRpcError:
                # connection gets lost during the call because we stop the initiation server on the actor
                return
