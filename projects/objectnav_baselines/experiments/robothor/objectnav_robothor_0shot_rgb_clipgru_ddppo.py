from allenact.embodiedai.preprocessors.resnet import ClipResNetPreprocessor
from allenact_plugins.ithor_plugin.ithor_sensors import (
    RGBSensorThor,
    GoalObjectTypeThorSensor,
)
from projects.objectnav_baselines.experiments.objectnav_mixin_clipgru import (
    ObjectNavMixInClipGRUConfig,
)
from projects.objectnav_baselines.experiments.objectnav_mixin_ddppo import (
    ObjectNavMixInPPOConfig,
)
from projects.objectnav_baselines.experiments.robothor.objectnav_robothor_base import (
    ObjectNavRoboThorBaseConfig,
)


class ObjectNavRoboThorClipRGBPPOExperimentConfig(
    ObjectNavRoboThorBaseConfig,
    ObjectNavMixInPPOConfig,
    ObjectNavMixInClipGRUConfig,
):
    """An Object Navigation experiment configuration in RoboThor with RGB
    input."""

    DEFAULT_NUM_TRAIN_PROCESSES = 2

    CLIP_MODEL_TYPE = "RN50"

    TARGET_TYPES = tuple(
        sorted(
            [
                "AlarmClock",
                "BaseballBat",
                "Bowl",
                "GarbageCan",
                "Laptop",
                "Mug",
                "SprayBottle",
                "Vase",
            ]
        )
    )

    # EVAL_TARGET_TYPES = tuple(
    #     sorted(
    #         [
    #             "Apple",
    #             "BasketBall",
    #             "HousePlant",
    #             "Television"
    #         ]
    #     )
    # )

    SENSORS = [
        RGBSensorThor(
            height=ObjectNavRoboThorBaseConfig.SCREEN_SIZE,
            width=ObjectNavRoboThorBaseConfig.SCREEN_SIZE,
            use_resnet_normalization=True,
            uuid="rgb_lowres",
            mean=ClipResNetPreprocessor.CLIP_RGB_MEANS,
            stdev=ClipResNetPreprocessor.CLIP_RGB_STDS,
        ),
        GoalObjectTypeThorSensor(
            object_types=TARGET_TYPES,
        ),
    ]

    @classmethod
    def tag(cls):
        return "Objectnav-RoboTHOR-RGB-ClipResNet50GRU-DDPPO"
