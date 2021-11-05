#! /usr/bin/env python

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models
import torchvision.transforms as T

import clip

from pointnav_vo.utils.misc_utils import Flatten
from pointnav_vo.utils.baseline_registry import baseline_registry
from pointnav_vo.model_utils.visual_encoders import resnet
from pointnav_vo.model_utils.running_mean_and_var import RunningMeanAndVar
from pointnav_vo.vo.common.common_vars import *


class CLIPEncoder(nn.Module):
    def __init__(self, observation_space, rgb_channels, depth_channels, discretized_depth_channels, top_down_view_channels):
        super().__init__()
        assert "rgb" in observation_space

        ## RGB processing

        self.rgb_channels = rgb_channels

        model, preprocess = clip.load("RN50")
        self.rgb_preprocess = T.Compose([
            # resize and center crop to 224
            preprocess.transforms[0],  
            preprocess.transforms[1],
            # already tensor, but want float
            T.ConvertImageDtype(torch.float),
            # normalize with CLIP mean, std
            preprocess.transforms[4],
        ])

        self.rgb_backbone = model.visual
        self.rgb_backbone.attnpool = nn.Identity()

        for param in self.rgb_backbone.parameters():
            param.requires_grad = False
        for module in self.rgb_backbone.modules():
            if "BatchNorm" in type(module).__name__:
                module.momentum = 0.0
        self.rgb_backbone.eval()

        ## Depth processing

        self.depth_channels = depth_channels if "depth" in observation_space else 0
        self.discretized_depth_channels = discretized_depth_channels if "discretized_depth" in observation_space else 0
        self.top_down_view_channels = top_down_view_channels if "top_down_view" in observation_space else 0

        self.preprocess_depth_size = (224, 224)

        self.resnet_input_channels = (self.depth_channels + self.discretized_depth_channels + self.top_down_view_channels)
        if self.resnet_input_channels > 0:
            depth_resnet = models.resnet18()
            # adjust num input channels
            depth_resnet.conv1 = nn.Conv2d(self.resnet_input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            # remove average pooling and classification layer
            self.resnet_model = nn.Sequential(*list(depth_resnet.children())[:-2])

        ## Pooling

        feature_channels = 2048
        if self.resnet_input_channels > 0:
            feature_channels += 512
        feature_channels *= 2

        self.compression = nn.Sequential(
            nn.Conv2d(feature_channels, 2048, kernel_size=(3, 3), padding=(0, 0)),
            nn.GroupNorm(32, 2048),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(2048, 2048, kernel_size=(3, 3), padding=(0, 0)),
            nn.GroupNorm(32, 2048),
            nn.AdaptiveAvgPool2d(output_size=(1,1)),
            nn.Flatten()
        )

        self.output_shape = (2048,)

    def forward(self, observation_pairs):
        feature_maps = []

        rgb_observations = observation_pairs["rgb"]
        rgb_observations = rgb_observations.permute(0, 3, 1, 2) # BATCH x CHANNEL x HEIGHT X WIDTH

        prev_rgb_observations = torch.stack([
            self.rgb_preprocess(rgb_image) 
            for rgb_image in rgb_observations[:, : self.rgb_channels, :]
        ])
        cur_rgb_observations = torch.stack([
            self.rgb_preprocess(rgb_image)
            for rgb_image in rgb_observations[:, self.rgb_channels :, :]
        ])

        prev_depth = []
        cur_depth = []

        if self.depth_channels > 0:
            depth_observations = observation_pairs["depth"]
            depth_observations = depth_observations.permute(0, 3, 1, 2)
            prev_depth.append(depth_observations[:, : self.depth_channels, :])
            cur_depth.append(depth_observations[:, self.depth_channels :, :])

        if self.discretized_depth_channels > 0:
            discretized_depth_observations = observation_pairs["discretized_depth"]
            discretized_depth_observations = discretized_depth_observations.permute(0, 3, 1, 2)  # BCHW
            prev_depth.append(discretized_depth_observations[:, : self.discretized_depth_channels, :])
            cur_depth.append(discretized_depth_observations[:, self.discretized_depth_channels :, :])

        if self.top_down_view_channels > 0:
            top_down_view_observations = observation_pairs["top_down_view"]
            top_down_view_observations = top_down_view_observations.permute(0, 3, 1, 2)
            prev_depth.append(top_down_view_observations[:, : self.top_down_view_channels, :])
            cur_depth.append(top_down_view_observations[:, self.top_down_view_channels :, :])

        prev_depth = F.interpolate(
            torch.cat(prev_depth, dim=1),
            size=self.preprocess_depth_size
        )
        cur_depth = F.interpolate(
            torch.cat(cur_depth, dim=1),
            size=self.preprocess_depth_size
        )

        feature_maps.append(self.rgb_backbone(prev_rgb_observations).float())

        if self.resnet_input_channels > 0:
            feature_maps.append(self.resnet_model(prev_depth))

        feature_maps.append(self.rgb_backbone(cur_rgb_observations).float())

        if self.resnet_input_channels > 0:
            feature_maps.append(self.resnet_model(cur_depth))

        x = torch.cat(feature_maps, dim=1)
        x = self.compression(x)

        return x


class VisualOdometryCLIPBase(nn.Module):
    def __init__(
        self,
        *,
        observation_space,
        rgb_channels=(RGB_PAIR_CHANNEL//2),
        depth_channels=(DEPTH_PAIR_CHANNEL//2),
        discretized_depth_channels=0,
        top_down_view_channels=(TOP_DOWN_VIEW_PAIR_CHANNEL//2),
        hidden_size=512,
        dropout_p=0.2,
        output_dim=DEFAULT_DELTA_STATE_SIZE,
    ):
        super().__init__()

        self.visual_encoder = CLIPEncoder(
            observation_space,
            rgb_channels,
            depth_channels,
            discretized_depth_channels,
            top_down_view_channels
        )

        self.visual_fc = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(self.visual_encoder.output_shape[0], hidden_size),
            nn.ReLU(True),
        )

        self.output_head = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(hidden_size, output_dim),
        )
        nn.init.orthogonal_(self.output_head[1].weight)
        nn.init.constant_(self.output_head[1].bias, 0)

    def forward(self, observation_pairs):
        visual_feats = self.visual_encoder(observation_pairs)
        visual_feats = self.visual_fc(visual_feats)
        output = self.output_head(visual_feats)
        return output


@baseline_registry.register_vo_model(name="vo_clip_rgb_d_dd_top_down")
class VisualOdometryCNNDiscretizedDepthTopDownView(VisualOdometryCLIPBase):
    def __init__(
        self,
        *,
        observation_space,
        observation_size,
        hidden_size=512,
        resnet_baseplanes=32,
        backbone="resnet50_clip",
        normalize_visual_inputs=False,
        output_dim=DEFAULT_DELTA_STATE_SIZE,
        dropout_p=0.2,
        discretized_depth_channels=10,
        top_down_view_pair_channel=TOP_DOWN_VIEW_PAIR_CHANNEL,
    ):
        assert backbone == "resnet50_clip"
        super().__init__(
            observation_space=observation_space,
            rgb_channels=(RGB_PAIR_CHANNEL//2),
            depth_channels=(DEPTH_PAIR_CHANNEL//2),
            discretized_depth_channels=discretized_depth_channels,
            top_down_view_channels=(top_down_view_pair_channel//2),
            hidden_size=hidden_size,
            dropout_p=dropout_p,
            output_dim=output_dim,
        )
