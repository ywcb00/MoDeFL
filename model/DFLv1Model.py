from tffmodel.IModel import IModel
from tffmodel.KerasModel import KerasModel
from tffmodel.ModelBuilderUtils import getLoss, getMetrics, getFedCoreOptimizers, getOptimizer
from tffmodel.ModelUtils import ModelUtils

import attrs
import logging
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
from typing import Any

class DFLv1Model(IModel):
    def __init__(self, config):
        super().__init__(config)
        self.logger = logging.getLogger("model/DFLv1Model")
        self.logger.setLevel(config["log_level"])

    @classmethod
    def createFedModel(self_class, fed_data, config):
        keras_model = KerasModel.createKerasModel(fed_data[0], config)
        fed_model = tff.learning.models.from_keras_model(
            keras_model = keras_model,
            input_spec = fed_data[0].element_spec,
            loss = getLoss(config),
            metrics = getMetrics(config))
        return fed_model

    def fit(self, fed_dataset):

        def cfm():
            keras_model = KerasModel.createKerasModel(fed_dataset.train[0], self.config)
            fed_model = tff.learning.models.from_keras_model(
                keras_model,
                input_spec = fed_dataset.train[0].element_spec,
                loss = getLoss(self.config),
                metrics = getMetrics(self.config))
            return fed_model

        global_optimizer, local_optimizer = getFedCoreOptimizers(self.config)

        @attrs.define(eq=False, frozen=True)
        class ActorState(object):
            trainable_weights: Any
            actor_idx: Any

        @tf.function
        def incrementByIdx(data, actor_state, actor_idx):
            result = ActorState(
                trainable_weights=actor_state.trainable_weights+actor_idx,
                actor_idx=actor_idx)
            return result

        @tff.tf_computation
        def getActorState():
            return ActorState(trainable_weights=0, actor_idx=0)

        @tff.tf_computation
        def initActor(nonsense_int):
            return getActorState()

        tff.backends.native.set_sync_local_cpp_execution_context(default_num_clients=self.config["num_workers"])

        @tff.federated_computation
        def actorInitFn():
            nonsense_int_server = tff.federated_value(13, tff.SERVER)
            nonsense_int_clients = tff.federated_broadcast(nonsense_int_server)
            init_state = tff.federated_map(initActor, nonsense_int_clients)
            return init_state

        tf_data_t = tff.SequenceType(tff.types.tensorflow_to_type(cfm().input_spec))
        actor_state_t = getActorState.type_signature.result

        actor_idx_t = np.int32

        @tff.tf_computation(tf_data_t, actor_state_t, actor_idx_t)
        def incrementActorWeights(data, actor_state, actor_idx):
            model = cfm()
            return incrementByIdx(data, actor_state, actor_idx)

        actor_state_seq_t = tff.SequenceType(actor_state_t)

        @tff.tf_computation(actor_state_t, actor_state_seq_t)
        def aggregateActorWeights(actor_state, actor_state_seq):
            print(actor_state)
            print(actor_state_seq)
            return actor_state

        fed_actor_state_t = tff.FederatedType(actor_state_t, tff.CLIENTS)
        fed_data_t = tff.FederatedType(tf_data_t, tff.CLIENTS, all_equal=False)

        fed_actor_idx_t = tff.FederatedType(actor_idx_t, tff.CLIENTS, all_equal=False)

        @tff.federated_computation(fed_actor_state_t, fed_data_t, fed_actor_idx_t)
        def runOneRound(actor_state, fed_data, fed_actor_idx):
            actor_state_seq = tff.federated_map(incrementActorWeights, (fed_data, actor_state, fed_actor_idx))
            return actor_state_seq

        actor_idx_list = list(range(self.config["num_workers"]))

        init_process = tff.templates.IterativeProcess(initialize_fn=actorInitFn,
            next_fn=runOneRound, next_is_multi_arg=True)
        actor_state = init_process.initialize()

        for counter in range(10):
            actor_state = init_process.next(actor_state, fed_dataset.train, actor_idx_list)
            print(actor_state)

    def predict():
        raise NotImplementedError

    def evaluate():
        raise NotImplementedError
