from typing import Dict
from mlagents.trainers.trainer import Trainer
from mlagents.trainers.settings import TrainerSettings
from mlagents.trainers.policy import Policy
from mlagents_envs.base_env import BehaviorSpec
from mlagents.trainers.behavior_id_utils import BehaviorIdentifiers
from .prednet_setting import PrednetSettings

print("This is the PrednetTrainer script.")

class PrednetTrainer(Trainer):

    def __init__(
        self,
        brain_name: str,
        trainer_settings: TrainerSettings,
        training: bool,
        load: bool,
        seed: int,
        artifact_path: str,
    ):
        super().__init__(brain_name, trainer_settings, training, load, seed, artifact_path)
        self.policies: Dict[str, Policy] = {}

        print("Hello, world!")

    @staticmethod
    def get_trainer_name() -> str:
        return "prednet"

    def create_policy(
        self,
        parsed_behavior_id: BehaviorIdentifiers,
        behavior_spec: BehaviorSpec,
    ) -> Policy:
        if behavior_spec.is_visual():
            print("Hello, world!")
        # print("Hello, world!")
        
        # Create and return your PredNet policy here

    def save_model(self) -> None:
        # Implement saving model
        pass

    def end_episode(self) -> None:
        # Implement end episode
        pass

    def add_policy(self, parsed_behavior_id: BehaviorIdentifiers, policy: Policy) -> None:
        self.policies[parsed_behavior_id.behavior_id] = policy
        pass

    def advance(self) -> None:
        pass

def get_type_and_setting():
    return {PrednetTrainer.get_trainer_name(): PrednetTrainer}, {
        PrednetTrainer.get_trainer_name(): PrednetSettings
    }
