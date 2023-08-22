import os
from typing import Dict

from mlagents_envs.logging_util import get_logger
from mlagents.trainers.environment_parameter_manager import EnvironmentParameterManager
from mlagents.trainers.exception import TrainerConfigError
from mlagents.trainers.trainer import Trainer
from mlagents.trainers.ghost.trainer import GhostTrainer
from mlagents.trainers.ghost.controller import GhostController
from mlagents.trainers.settings import TrainerSettings
from mlagents.plugins import all_trainer_types


logger = get_logger(__name__)


class TrainerFactory:
    def __init__(
        self,
        trainer_config: Dict[str, TrainerSettings],
        output_path: str,
        train_model: bool,
        load_model: bool,
        seed: int,
        param_manager: EnvironmentParameterManager,
        init_path: str = None,
        multi_gpu: bool = False,
    ):
        """
        The TrainerFactory generates the Trainers based on the configuration passed as
        input.
        :param trainer_config: A dictionary from behavior name to TrainerSettings
        :param output_path: The path to the directory where the artifacts generated by
        the trainer will be saved.
        :param train_model: If True, the Trainers will train the model and if False,
        only perform inference.
        :param load_model: If True, the Trainer will load neural networks weights from
        the previous run.
        :param seed: The seed of the Trainers. Dictates how the neural networks will be
        initialized.
        :param param_manager: The EnvironmentParameterManager that will dictate when/if
        the EnvironmentParameters must change.
        :param init_path: Path from which to load model.
        :param multi_gpu: If True, multi-gpu will be used. (currently not available)
        """
        self.trainer_config = trainer_config
        self.output_path = output_path
        self.init_path = init_path
        self.train_model = train_model
        self.load_model = load_model
        self.seed = seed
        self.param_manager = param_manager
        self.multi_gpu = multi_gpu
        self.ghost_controller = GhostController()

    def generate(self, behavior_name: str) -> Trainer:
        trainer_settings = self.trainer_config[behavior_name]
        return TrainerFactory._initialize_trainer(
            trainer_settings,
            behavior_name,
            self.output_path,
            self.train_model,
            self.load_model,
            self.ghost_controller,
            self.seed,
            self.param_manager,
            self.multi_gpu,
        )

    @staticmethod
    def _initialize_trainer(
        trainer_settings: TrainerSettings,
        brain_name: str,
        output_path: str,
        train_model: bool,
        load_model: bool,
        ghost_controller: GhostController,
        seed: int,
        param_manager: EnvironmentParameterManager,
        multi_gpu: bool = False,
    ) -> Trainer:
        """
        Initializes a trainer given a provided trainer configuration and brain parameters, as well as
        some general training session options.

        :param trainer_settings: Original trainer configuration loaded from YAML
        :param brain_name: Name of the brain to be associated with trainer
        :param output_path: Path to save the model and summary statistics
        :param keep_checkpoints: How many model checkpoints to keep
        :param train_model: Whether to train the model (vs. run inference)
        :param load_model: Whether to load the model or randomly initialize
        :param ghost_controller: The object that coordinates ghost trainers
        :param seed: The random seed to use
        :param param_manager: EnvironmentParameterManager, used to determine a reward buffer length for PPOTrainer
        :return:
        """
        trainer_artifact_path = os.path.join(output_path, brain_name)

        min_lesson_length = param_manager.get_minimum_reward_buffer_size(brain_name)

        trainer: Trainer = None  # type: ignore  # will be set to one of these, or raise
        print("here")
        try:
            trainer_type = all_trainer_types[trainer_settings.trainer_type]
            print("here2")
            trainer = trainer_type(
                brain_name,
                min_lesson_length,
                trainer_settings,
                train_model,
                load_model,
                seed,
                trainer_artifact_path,
            )
            print("here3")
        except KeyError:
            raise TrainerConfigError(
                f"The trainer config contains an unknown trainer type "
                f"{trainer_settings.trainer_type} for brain {brain_name}"
            )

        if trainer_settings.self_play is not None:
            trainer = GhostTrainer(
                trainer,
                brain_name,
                ghost_controller,
                min_lesson_length,
                trainer_settings,
                train_model,
                trainer_artifact_path,
            )
        return trainer
