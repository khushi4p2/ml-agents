from typing import Dict
from mlagents.trainers.trainer import Trainer
from mlagents.trainers.settings import TrainerSettings
from mlagents.trainers.policy import Policy
from mlagents_envs.base_env import BehaviorSpec
from mlagents.trainers.behavior_id_utils import BehaviorIdentifiers
from mlagents_envs.logging_util import get_logger
from .prednet_setting import PrednetSettings

logger = get_logger(__name__)
TRAINER_NAME = "prednet"

print("This is the PrednetTrainer script.")

class PrednetTrainer(Trainer):



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
