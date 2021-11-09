from typing import Sequence, Union

import gym
import torch.nn as nn
from allenact.base_abstractions.preprocessor import Preprocessor
from allenact.embodiedai.sensors.vision_sensors import RGBSensor, DepthSensor
from allenact.utils.experiment_utils import Builder
from allenact_plugins.ithor_plugin.ithor_sensors import GoalObjectTypeThorSensor

from manipulathor_baselines.armpointnav_baselines.experiments.armpointnav_base import (
    ArmPointNavBaseConfig,
)
from manipulathor_baselines.armpointnav_baselines.models.arm_pointnav_models import (
    ArmPointNavCLIPActorCritic,
)


class ArmPointNavMixInCLIPGRUConfig(ArmPointNavBaseConfig):
    @classmethod
    def preprocessors(cls) -> Sequence[Union[Preprocessor, Builder[Preprocessor]]]:
        preprocessors = []
        return preprocessors

    @classmethod
    def create_model(cls, **kwargs) -> nn.Module:
        goal_sensor_uuid = next(
            (s.uuid for s in cls.SENSORS if isinstance(s, GoalObjectTypeThorSensor)),
            None,
        )

        return ArmPointNavCLIPActorCritic(
            action_space=gym.spaces.Discrete(
                len(cls.TASK_SAMPLER._TASK_TYPE.class_action_names())
            ),
            observation_space=kwargs["sensor_preprocessor_graph"].observation_spaces,
            hidden_size=512,
        )
