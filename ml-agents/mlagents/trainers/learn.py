# # Unity ML-Agents Toolkit
from mlagents import torch_utils
import yaml

import os
import numpy as np
import json

from typing import Callable, Optional, List

import mlagents.trainers
import mlagents_envs
from mlagents.trainers.trainer_controller import TrainerController
from mlagents.trainers.environment_parameter_manager import EnvironmentParameterManager
from mlagents.trainers.trainer import TrainerFactory
from mlagents.trainers.directory_utils import (
    validate_existing_directories,
    setup_init_path,
)
from mlagents.trainers.stats import StatsReporter
from mlagents.trainers.cli_utils import parser
from mlagents_envs.environment import UnityEnvironment
from mlagents.trainers.settings import RunOptions

from mlagents.trainers.training_status import GlobalTrainingStatus
from mlagents_envs.base_env import BaseEnv
from mlagents.trainers.subprocess_env_manager import SubprocessEnvManager
from mlagents_envs.side_channel.side_channel import SideChannel
from mlagents_envs.timers import (
    hierarchical_timer,
    get_timer_tree,
    add_metadata as add_timer_metadata,
)
from mlagents_envs import logging_util
from mlagents.plugins.stats_writer import register_stats_writer_plugins
from mlagents.plugins.trainer_type import register_trainer_plugins

logger = logging_util.get_logger(__name__)

# training status is being saved to this particular file
TRAINING_STATUS_FILE_NAME = "training_status.json"


# just to print some version info on the terminal screen
def get_version_string() -> str:
    return f""" Version information:
  ml-agents: {mlagents.trainers.__version__},
  ml-agents-envs: {mlagents_envs.__version__},
  Communicator API: {UnityEnvironment.API_VERSION},
  PyTorch: {torch_utils.torch.__version__}"""

# parse command-line arguments provided when running a script
def parse_command_line(
    argv: Optional[List[str]] = None,
) -> RunOptions:
    _, _ = register_trainer_plugins()
    args = parser.parse_args(argv)
    return RunOptions.from_argparse(args)

# main entry-point to launch a training session
def run_training(run_seed: int, options: RunOptions, num_areas: int) -> None:
    """
    Launches training session.
    :param run_seed: Random seed used for training.
    :param num_areas: Number of training areas to instantiate
    :param options: parsed command line arguments
    """
    with hierarchical_timer("run_training.setup"):
        torch_utils.set_torch_config(options.torch_settings)
        checkpoint_settings = options.checkpoint_settings
        env_settings = options.env_settings
        engine_settings = options.engine_settings

        run_logs_dir = checkpoint_settings.run_logs_dir
        port: Optional[int] = env_settings.base_port
        # Check if directory exists
        validate_existing_directories(
            checkpoint_settings.write_path,
            checkpoint_settings.resume,
            checkpoint_settings.force,
            checkpoint_settings.maybe_init_path,
        )
        # Make run logs directory
        os.makedirs(run_logs_dir, exist_ok=True)
        # This code block either starts a new training session, or loads a previously running training session
        # Load any needed states in case of resume
        if checkpoint_settings.resume:
            GlobalTrainingStatus.load_state(
                os.path.join(run_logs_dir, "training_status.json")
            )
        # In case of initialization, set full init_path for all behaviors
        elif checkpoint_settings.maybe_init_path is not None:
            setup_init_path(options.behaviors, checkpoint_settings.maybe_init_path)
        # This code block either starts a new training session, or loads a previously running training session

        # Just saving some statistics here
        # Configure Tensorboard Writers and StatsReporter
        stats_writers = register_stats_writer_plugins(options)
        for sw in stats_writers:
            StatsReporter.add_writer(sw)
        # Just saving some statistics here

        # code which sets up the simulation environment and the connection with the Unity env.
        if env_settings.env_path is None:
            port = None.   # if port == None, then it means that the Unity Editor is being used as training env, which doesn't require port value to be anything
        env_factory = create_environment_factory(. # "Environment Factory" is a design pattern used to create instances of the environment.
            env_settings.env_path,   # path to the Unity Environment which is used for training
            engine_settings.no_graphics,
            run_seed,
            num_areas,
            port,
            env_settings.env_args,
            os.path.abspath(run_logs_dir),  # Unity environment requires absolute path
        )
# environment manager is responsible for stepping the environments, collecting observations, and sending actions to the environments.
        env_manager = SubprocessEnvManager(env_factory, options, env_settings.num_envs)
# manages the simulation parameters of the Unity environments that can change over time or across training runs
# we can also encode this part to update our simulation environment at the beginning of each run
        env_parameter_manager = EnvironmentParameterManager(
            options.environment_parameters, run_seed, restore=checkpoint_settings.resume
        )
# code which sets up the simulation environment and the connection with the Unity env.

        trainer_factory = TrainerFactory(
            trainer_config=options.behaviors,  # path to training algorithms 
            output_path=checkpoint_settings.write_path,  # path where the training results (like model weights, for instance) written to, used to save checkpoints of the model during training
            train_model=not checkpoint_settings.inference, # bool to indicate whether to train model or not - if set to TRUE, model not trained, and if FALSE
            load_model=checkpoint_settings.resume, # to contiue/resume trainig from a previously saved checkpoint, if TRUE, training process will load a pre-trained model from the checkpoint_settings.write_path
            seed=run_seed, # setting random seed for training
            param_manager=env_parameter_manager, # EnvironmentParameterManager which manages the simulation parameters of the Unity environments that can change over time or across training runs.
            init_path=checkpoint_settings.maybe_init_path, # the path to a checkpoint from which to initialize the model. This is different from load_model in that it's used for initialization, not for resuming training.
            multi_gpu=False, # whether to use multiple GPUs for training.
        )
        # Create controller and begin training.
        tc = TrainerController( #responsible for managing the training process, including executing the main training loop, orchestrating the interactions between the environment and the agents' trainers, and handling logging and model checkpointing.
            trainer_factory, # used to create trainers for the agents. Each trainer will manage the training of one type of agent behavior.
            checkpoint_settings.write_path, #directory where training artifacts like model weights (checkpoints) and logs saved
            checkpoint_settings.run_id, #an identifier for the current training run. It's often used to distinguish between different runs, such as runs with different hyperparameters or runs at different times
            env_parameter_manager, # instance of EnvironmentParameterManager which manages the simulation parameters that can change during the training run or across different training runs
            not checkpoint_settings.inference, #  boolean that indicates whether to train the model
            run_seed, #seed for the random number generator used in training
        )
    # Begin training
    #responsible for starting the training process, managing the environment during training, and saving relevant information about the training run
    try:
        tc.start_learning(env_manager) #starts the training process, initiates the main training loop. takes as input the env_manager, which is responsible for managing the interactions with the Unity environment
    finally: # this exec EVEN IF an error occurs, kind of to perform cleanup operations.
        env_manager.close() #shuts down the env_manager after the training process has finished or if an error has occurred
        write_run_options(checkpoint_settings.write_path, options) # writes the run options to a file, like the parameters and settings used for the training run. This is useful for keeping a record of what settings were used for each run.
        write_timing_tree(run_logs_dir) # writes timing information to a file(?)
        write_training_status(run_logs_dir) # writes the final training status like the final value of the reward function, the total number of steps taken etc.

# LOGGING RELATD FUNCTIONS
# writes the configurations or settings used during the training run into a YAML file named "configuration.yaml"
def write_run_options(output_dir: str, run_options: RunOptions) -> None:
    run_options_path = os.path.join(output_dir, "configuration.yaml")
    try:
        with open(run_options_path, "w") as f:
            try:
                yaml.dump(run_options.as_dict(), f, sort_keys=False)
            except TypeError:  # Older versions of pyyaml don't support sort_keys
                yaml.dump(run_options.as_dict(), f)
    except FileNotFoundError:
        logger.warning(
            f"Unable to save configuration to {run_options_path}. Make sure the directory exists"
        )
# writes the global training status into a JSON file. global training status -> info about the training process like the current step number, the mean reward, etc.
def write_training_status(output_dir: str) -> None:
    GlobalTrainingStatus.save_state(os.path.join(output_dir, TRAINING_STATUS_FILE_NAME))
# writes the timing data collected during the training run into a JSON file
def write_timing_tree(output_dir: str) -> None:
    timing_path = os.path.join(output_dir, "timers.json")
    try:
        with open(timing_path, "w") as f:
            json.dump(get_timer_tree(), f, indent=4)
    except FileNotFoundError:
        logger.warning(
            f"Unable to save to {timing_path}. Make sure the directory exists"
        )

# ENV RELATED FUNCTIONS
# basically creating instances of the UnityEnvironment class(?)
def create_environment_factory(
    env_path: Optional[str],
    no_graphics: bool,
    seed: int,
    num_areas: int,
    start_port: Optional[int],
    env_args: Optional[List[str]],
    log_folder: str,
) -> Callable[[int, List[SideChannel]], BaseEnv]:
    def create_unity_environment(  # returns a BaseEnv object (an instance of the UnityEnvironment class)
        worker_id: int, side_channels: List[SideChannel]
    ) -> UnityEnvironment:
        # Make sure that each environment gets a different seed
        env_seed = seed + worker_id
        return UnityEnvironment(
            file_name=env_path,
            worker_id=worker_id,
            seed=env_seed,
            num_areas=num_areas,
            no_graphics=no_graphics,
            base_port=start_port,
            additional_args=env_args,
            side_channels=side_channels,
            log_folder=log_folder,
        )
# returns a callable(nested) function 
    return create_unity_environment

# text to be displayed on the client terminal(Python)
def run_cli(options: RunOptions) -> None:
    try:
        print(
            """
            ┐  ╖
        ╓╖╬│╡  ││╬╖╖
    ╓╖╬│││││┘  ╬│││││╬╖
 ╖╬│││││╬╜        ╙╬│││││╖╖                               ╗╗╗
 ╬╬╬╬╖││╦╖        ╖╬││╗╣╣╣╬      ╟╣╣╬    ╟╣╣╣             ╜╜╜  ╟╣╣
 ╬╬╬╬╬╬╬╬╖│╬╖╖╓╬╪│╓╣╣╣╣╣╣╣╬      ╟╣╣╬    ╟╣╣╣ ╒╣╣╖╗╣╣╣╗   ╣╣╣ ╣╣╣╣╣╣ ╟╣╣╖   ╣╣╣
 ╬╬╬╬┐  ╙╬╬╬╬│╓╣╣╣╝╜  ╫╣╣╣╬      ╟╣╣╬    ╟╣╣╣ ╟╣╣╣╙ ╙╣╣╣  ╣╣╣ ╙╟╣╣╜╙  ╫╣╣  ╟╣╣
 ╬╬╬╬┐     ╙╬╬╣╣      ╫╣╣╣╬      ╟╣╣╬    ╟╣╣╣ ╟╣╣╬   ╣╣╣  ╣╣╣  ╟╣╣     ╣╣╣┌╣╣╜
 ╬╬╬╜       ╬╬╣╣      ╙╝╣╣╬      ╙╣╣╣╗╖╓╗╣╣╣╜ ╟╣╣╬   ╣╣╣  ╣╣╣  ╟╣╣╦╓    ╣╣╣╣╣
 ╙   ╓╦╖    ╬╬╣╣   ╓╗╗╖            ╙╝╣╣╣╣╝╜   ╘╝╝╜   ╝╝╝  ╝╝╝   ╙╣╣╣    ╟╣╣╣
   ╩╬╬╬╬╬╬╦╦╬╬╣╣╗╣╣╣╣╣╣╣╝                                             ╫╣╣╣╣
      ╙╬╬╬╬╬╬╬╣╣╣╣╣╣╝╜
          ╙╬╬╬╣╣╣╜
             ╙
        """
        )
    except Exception:
        print("\n\n\tUnity Technologies\n")
    print(get_version_string())
# text to be displayed on the client terminal(Python)

# sets up logs to be returned, based on modes DEBUG and INFO
    if options.debug:
        log_level = logging_util.DEBUG
    else:
        log_level = logging_util.INFO

    logging_util.set_log_level(log_level)

    logger.debug("Configuration for this run:")
    logger.debug(json.dumps(options.as_dict(), indent=4))
# sets up logs to be returned, based on modes DEBUG and INFO

# Just setting up some deprecation warnings 
    # Options deprecation warnings
    if options.checkpoint_settings.load_model:
        logger.warning(
            "The --load option has been deprecated. Please use the --resume option instead."
        )
    if options.checkpoint_settings.train_model:
        logger.warning(
            "The --train option has been deprecated. Train mode is now the default. Use "
            "--inference to run in inference mode."
        )
# Just setting up some deprecation warnings

# Extracting the seed and the number of areas from the environment settings
    run_seed = options.env_settings.seed
    num_areas = options.env_settings.num_areas # num_areas could be used in the case of training with multiple parallel environments
# Extracting the seed and the number of areas from the environment settings

    # Add some timer metadata
    add_timer_metadata("mlagents_version", mlagents.trainers.__version__)
    add_timer_metadata("mlagents_envs_version", mlagents_envs.__version__)
    add_timer_metadata("communication_protocol_version", UnityEnvironment.API_VERSION)
    add_timer_metadata("pytorch_version", torch_utils.torch.__version__)
    add_timer_metadata("numpy_version", np.__version__)

# Setting the random seed
    if options.env_settings.seed == -1:
        run_seed = np.random.randint(0, 10000)
        logger.debug(f"run_seed set to {run_seed}")
# Running the training process [IMPORTANT]
    run_training(run_seed, options, num_areas)

# MAIN FUNCTION 
def main():
    run_cli(parse_command_line())


# For python debugger to directly run this script
if __name__ == "__main__":
    main()
