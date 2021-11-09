from typing import Dict

import torch
import torch.nn as nn

from torchvision import models
import clip

from gym.spaces.dict import Dict as SpaceDict


class LinearActorHeadNoCategory(nn.Module):
    def __init__(self, num_inputs: int, num_outputs: int):
        super().__init__()

        self.linear = nn.Linear(num_inputs, num_outputs)
        nn.init.orthogonal_(self.linear.weight, gain=0.01)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x: torch.FloatTensor):  # type: ignore
        x = self.linear(x)  # type:ignore
        assert len(x.shape) == 3
        return x


class ResNet50Encoder(nn.Module):
    def __init__(
        self,
        observation_space: SpaceDict,
        output_size: int,
        rgb_uuid: str
    ):
        super().__init__()

        self.rgb_uuid = rgb_uuid
        assert self.rgb_uuid in observation_space.spaces

        self.backbone = models.resnet50(pretrained=True)
        self.backbone.fc = nn.Identity()

        for param in self.backbone.parameters():
            param.requires_grad = False
        for module in self.backbone.modules():
            if "BatchNorm" in type(module).__name__:
                module.momentum = 0.0
        self.backbone.eval()

        self.fc = nn.Sequential(
            nn.Linear(2048, output_size),
            nn.ReLU(True)
        )

    def forward(self, observations: Dict[str, torch.Tensor]):  # type: ignore
        # expects rgb_input to be pre-processed by resnet normalization
        rgb_input = observations[self.rgb_uuid]  # ..., 224, 224, 3
        input_batch_shape = rgb_input.shape[:-3]
        rgb_input = torch.flatten(rgb_input, end_dim=-4).permute(0, 3, 1, 2)  # prod(...), 3, 224, 224
        x = self.fc(self.backbone(rgb_input))  # prod(...), output_size
        x = x.reshape(*input_batch_shape, -1)  # ..., output_size
        return x


class CLIPVisualEncoder(nn.Module):
    def __init__(
        self,
        observation_space: SpaceDict,
        output_size: int,
        rgb_uuid: str
    ):
        super().__init__()

        self.rgb_uuid = rgb_uuid
        assert self.rgb_uuid in observation_space.spaces

        clip_model = clip.load('RN50')[0]
        self.backbone = clip_model.visual
        self.backbone.attnpool = nn.Identity()

        for param in self.backbone.parameters():
            param.requires_grad = False
        for module in self.backbone.modules():
            if "BatchNorm" in type(module).__name__:
                module.momentum = 0.0
        self.backbone.eval()

        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1,1)),
            nn.Flatten(),
            nn.Linear(2048, output_size),
            nn.ReLU(True)
        )

    def forward(self, observations: Dict[str, torch.Tensor]):  # type: ignore
        # expects rgb_input to be pre-processed by CLIP normalization
        rgb_input = observations[self.rgb_uuid]  # ..., 224, 224, 3
        input_batch_shape = rgb_input.shape[:-3]
        rgb_input = torch.flatten(rgb_input, end_dim=-4).permute(0, 3, 1, 2)  # prod(...), 3, 224, 224
        x = self.backbone(rgb_input).float()  # prod(...), 2048, 7, 7
        x = self.pool(x)  # prod(...), output_size
        x = x.reshape(*input_batch_shape, -1)  # ..., output_size
        return x
