from typing import Dict, cast
import numpy as np
from mlagents.trainers.trainer import Trainer
from mlagents.trainers.settings import TrainerSettings
from mlagents.trainers.policy import Policy
from mlagents.trainers.policy.torch_policy import TorchPolicy
# important
from mlagents_envs.base_env import BehaviorSpec
from mlagents.trainers.behavior_id_utils import BehaviorIdentifiers
from mlagents_envs.logging_util import get_logger
from mlagents.trainers.trainer.off_policy_trainer import OffPolicyTrainer
from mlagents.trainers.trajectory import Trajectory
from mlagents.trainers.optimizer.torch_optimizer import TorchOptimizer
import matplotlib.pyplot as plt
from .prednet_setting import PrednetSettings, PrednetOptimizer, QNetwork

logger = get_logger(__name__)
TRAINER_NAME = "prednet"

print("This is the PrednetTrainer script.")

class PrednetTrainer(OffPolicyTrainer):

    def __init__(
        self,
        behavior_name: str,
        reward_buff_cap: int,
        trainer_settings: TrainerSettings,
        training: bool,
        load: bool,
        seed: int,
        artifact_path: str,
    ):
        """
        Responsible for collecting experiences and training SAC model.
        :param behavior_name: The name of the behavior associated with trainer config
        :param reward_buff_cap: Max reward history to track in the reward buffer
        :param trainer_settings: The parameters for the trainer.
        :param training: Whether the trainer is set for training.
        :param load: Whether the model should be loaded.
        :param seed: The seed the model will be initialized with
        :param artifact_path: The directory within which to store artifacts from this trainer.
        """
        super().__init__(
            behavior_name,
            reward_buff_cap,
            trainer_settings,
            training,
            load,
            seed,
            artifact_path,
        )
        self.policies: Dict[str, Policy] = {}
        # self.reward_buff_cap = reward_buff_cap  # Min lesson length from TrainerFactory._initialize_trainer()
        self.reward_buff_cap = 1
        self.optimizer: PrednetOptimizer = None  # type: ignore
        # self.policy: TorchPolicy = None  # type: ignore
        # self.optimizer: PrednetTrainer = None  # type: ignore
        print("Hello, World 1")

    def _process_trajectory(self, trajectory: Trajectory) -> None:
        """
        Takes a trajectory and processes it, putting it into the replay buffer.
        """
        print("Hello, World 2")

        for index, obs_spec in enumerate(spec.observation_specs):
        if len(obs_spec.shape) == 3:
            print("Here is the first visual observation")
            plt.imshow(decision_steps.obs[index][0,:,:,:])
            plt.show()

        for index, obs_spec in enumerate(spec.observation_specs):
        if len(obs_spec.shape) == 1:
            print("First vector observations : ", decision_steps.obs[index][0,:])
        # return None

# NOT REQUIRED
    def create_optimizer(self)->TorchOptimizer:
        """
        Creates an Optimizer object
        """
        print("Hello, world 3")

        return PrednetOptimizer(# type: ignore
            cast(TorchPolicy, self.policy), self.trainer_settings  # type: ignore
        )  # type: ignore
        return None

# NOT REQUIRED
    def create_policy(
        self, parsed_behavior_id: BehaviorIdentifiers, behavior_spec: BehaviorSpec
    ):
        """
        Creates a policy with a PyTorch backend and give DQN hyperparameters
        :param parsed_behavior_id:
        :param behavior_spec: specifications for policy construction
        :return policy
        """
        print("Hello, World 4")
        return None

# required
    @staticmethod
    def get_settings_type():
        return PrednetSettings

# required
    @staticmethod
    def get_trainer_name() -> str:
        return TRAINER_NAME

# required
def get_type_and_setting():
    return {PrednetTrainer.get_trainer_name(): PrednetTrainer}, {
        PrednetTrainer.get_trainer_name(): PrednetSettings
    }
