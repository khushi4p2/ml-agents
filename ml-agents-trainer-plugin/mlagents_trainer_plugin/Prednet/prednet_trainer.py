from typing import cast

import numpy as np

from mlagents_envs.base_env import BehaviorSpec
from mlagents.trainers.trainer.rl_trainer import RLTrainer
from mlagents_envs.logging_util import get_logger
from mlagents.trainers.buffer import BufferKey, RewardSignalUtil
# from mlagents.trainers.trainer.on_policy_trainer import OnPolicyTrainer
# from mlagents.trainers.optimizer.torch_optimizer import TorchOptimizer
from mlagents.trainers.trainer.trainer_utils import get_gae
# from mlagents.trainers.policy.torch_policy import TorchPolicy
# from .a2c_optimizer import A2COptimizer, A2CSettings
from mlagents.trainers.trajectory import Trajectory
from mlagents.trainers.behavior_id_utils import BehaviorIdentifiers
from mlagents.trainers.settings import TrainerSettings

# from mlagents.trainers.torch_entities.networks import SimpleActor, SharedActorCritic

from mlagents.trainers.trainer import trainer

class PredNetTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def add_policy(self, policy):
        pass

    def get_policy(self, behavior_id):
        pass

    def _is_ready_update(self):
        pass

    def _update_policy(self):
        pass

    def end_episode(self):
        pass

# Upon receiving the observations from the Camera Sensor object (M2 script), "Hello World" will be output on the console
    def _process_trajectory(self, trajectory: Trajectory) -> None:
        # if 'CameraSensor' in trajectory.obs[0]:
        #     print('Camera sensor data received.')
        #     print('Hello World!')
        int succ = 0
        print('Observations received:')
        for i, obs in enumerate(trajectory.obs):
            print(f'Observation {i+1} shape: {obs.shape}')
            succ = 1
        
        if (succ == 1)
        {
            print('Hello World!')
        }

    def get_prednet_trainer():
        return {PredNetTrainer.__name__: PredNetTrainer}, {}
