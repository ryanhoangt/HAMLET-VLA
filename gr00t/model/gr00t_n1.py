# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import tree
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HFValidationError, RepositoryNotFoundError
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel
from transformers.feature_extraction_utils import BatchFeature

from .action_head.flow_matching_action_head import (
    FlowmatchingActionHead,
    FlowmatchingActionHeadConfig,
)
from .backbone import EagleBackbone
from .hamlet import MemoryModule

BACKBONE_FEATURE_KEY = "backbone_features"
ACTION_KEY = "action_pred"
LOSS_KEY = "loss"
ERROR_MSG = "Error: unexpected input/output"
N_COLOR_CHANNELS = 3


# config
@dataclass
class GR00T_N1_5_Config(PretrainedConfig):
    model_type = "gr00t_n1_5"
    backbone_cfg: dict = field(init=False, metadata={"help": "Backbone configuration."})

    action_head_cfg: dict = field(init=False, metadata={"help": "Action head configuration."})

    action_horizon: int = field(init=False, metadata={"help": "Action horizon."})

    action_dim: int = field(init=False, metadata={"help": "Action dimension."})
    compute_dtype: str = field(default="float32", metadata={"help": "Compute dtype."})

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


# real model
class GR00T_N1_5(PreTrainedModel):
    supports_gradient_checkpointing = True
    config_class = GR00T_N1_5_Config
    """
    we expect the backbone output to have a key 'backbone_features' with shape (batch_size, n, hidden_size)
    here n is variable and can be e.g. time, 1 or user specified
    we expect the action head output to have a key 'action_pred' with shape (batch_size, time, action_dim) during inference time
    we expect these to have type BatchFeature, and they can of course have many other user specified keys too
    """

    def __init__(
        self,
        config: GR00T_N1_5_Config,
        local_model_path: str,
    ):
        assert isinstance(config.backbone_cfg, dict)
        assert isinstance(config.action_head_cfg, dict)

        super().__init__(config)
        self.local_model_path = local_model_path

        self.backbone = EagleBackbone(**config.backbone_cfg)
        action_head_cfg = FlowmatchingActionHeadConfig(**config.action_head_cfg)
        self.action_head = FlowmatchingActionHead(action_head_cfg)

        self.action_horizon = config.action_horizon
        self.action_dim = config.action_dim
        self.compute_dtype = config.compute_dtype

        self.hamlet_memory: Optional[MemoryModule] = None  # attached via attach_hamlet()

    def validate_inputs(self, inputs):
        # NOTE -- this should be handled internally by the model
        # however, doing that will likely be breaking changes -- so we'll need to do it after the deadline

        detected_error = False
        error_msg = ERROR_MSG
        if "action" in inputs:
            action = inputs["action"]
            type_ok = isinstance(action, torch.Tensor)
            shape_ok = (
                len(action.shape) == 3
                and action.shape[1] == self.action_horizon
                and action.shape[2] == self.action_dim
            )
            if not type_ok:
                error_msg += f"\n{action.dtype=}"
                detected_error = True
            if not shape_ok:
                error_msg += f"\n{action.shape=}"
                detected_error = True

        if "video" in inputs:
            video = inputs["video"]
            type_ok = isinstance(video, np.ndarray)
            dtype_ok = video.dtype == np.uint8
            shape_ok = len(video.shape) == 6 and video.shape[3] == N_COLOR_CHANNELS
            if not type_ok:
                error_msg += f"\n{type(video)=}"
                detected_error = True
            if not dtype_ok:
                error_msg += f"\n{video.dtype=}"
                detected_error = True
            if not shape_ok:
                error_msg += f"\n{video.shape=}"
                detected_error = True

        if detected_error:
            raise ValueError(error_msg)

    def validate_data(self, action_head_outputs, backbone_outputs, is_training):
        fail_backbone = (
            not isinstance(backbone_outputs, BatchFeature)
            or BACKBONE_FEATURE_KEY not in backbone_outputs
        )

        if fail_backbone:
            error_msg = ERROR_MSG
            error_msg += f"\n{isinstance(backbone_outputs, BatchFeature)=}"
            error_msg += f"\n{BACKBONE_FEATURE_KEY in backbone_outputs=}"
            error_msg += f"\n{backbone_outputs[BACKBONE_FEATURE_KEY].shape=}"
            raise ValueError(error_msg)

        fail_action_head = (not isinstance(action_head_outputs, BatchFeature)) or not (
            (
                LOSS_KEY in action_head_outputs and is_training
            )  # there might not be an action prediction during training
            or (
                ACTION_KEY in action_head_outputs
                and action_head_outputs[ACTION_KEY].shape[1] == self.action_horizon
                and action_head_outputs[ACTION_KEY].shape[2] == self.action_dim
            )
        )

        if fail_action_head:
            error_msg = ERROR_MSG
            error_msg += f"\n{isinstance(action_head_outputs, BatchFeature)=}"
            error_msg += f"\n{LOSS_KEY in action_head_outputs=}"
            error_msg += f"\n{action_head_outputs[ACTION_KEY].shape=}"
            error_msg += f"\n{self.action_horizon=}"
            error_msg += f"\n{self.action_dim=}"
            raise ValueError(error_msg)

    def _apply_hamlet_memory(
        self,
        backbone_outputs: BatchFeature,
        moment_history: Optional[torch.Tensor],
    ) -> BatchFeature:
        """Concatenate memory context to backbone_features if HAMLET is active.

        Args:
            backbone_outputs: output from backbone, contains backbone_features (B, N, d)
            moment_history:   (B, T, n_m, d) — T stacked moment token features;
                              the current timestep's features should be the last entry.
        Returns:
            backbone_outputs with backbone_features replaced by (B, N+n_m, d)
        """
        if self.hamlet_memory is None or moment_history is None:
            return backbone_outputs

        m_tilde = self.hamlet_memory(moment_history)  # (B, n_m, d)
        feats = backbone_outputs[BACKBONE_FEATURE_KEY]  # (B, N, d)
        backbone_outputs[BACKBONE_FEATURE_KEY] = torch.cat([feats, m_tilde], dim=1)
        return backbone_outputs

    def forward(
        self,
        inputs: dict,
        moment_history: Optional[torch.Tensor] = None,
    ) -> BatchFeature:
        """
        Args:
            inputs:         standard GR00T input dict
            moment_history: (B, T, n_m, d) stacked moment token features for HAMLET.
                            None → standard forward (no memory).
        """
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        backbone_outputs = self._apply_hamlet_memory(backbone_outputs, moment_history)
        action_head_outputs = self.action_head(backbone_outputs, action_inputs)
        self.validate_data(action_head_outputs, backbone_outputs, is_training=True)
        return action_head_outputs

    def get_action(
        self,
        inputs: dict,
        moment_history: Optional[torch.Tensor] = None,
    ) -> BatchFeature:
        """
        Args:
            inputs:         standard GR00T input dict
            moment_history: (B, T, n_m, d) stacked moment token features for HAMLET.
                            None → standard inference (no memory).
        """
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)
        backbone_outputs = self._apply_hamlet_memory(backbone_outputs, moment_history)
        action_head_outputs = self.action_head.get_action(backbone_outputs, action_inputs)
        self.validate_data(action_head_outputs, backbone_outputs, is_training=False)
        return action_head_outputs

    def prepare_input(self, inputs) -> Tuple[BatchFeature, BatchFeature]:
        self.validate_inputs(inputs)
        backbone_inputs = self.backbone.prepare_input(inputs)
        action_inputs = self.action_head.prepare_input(inputs)

        def to_device_with_maybe_dtype(x):
            # Only cast to self.compute_dtype if the tensor is floating
            if torch.is_floating_point(x):
                return x.to(self.device, dtype=self.action_head.dtype)
            else:
                # Keep original dtype
                return x.to(self.device)

        backbone_inputs = tree.map_structure(to_device_with_maybe_dtype, backbone_inputs)
        action_inputs = tree.map_structure(to_device_with_maybe_dtype, action_inputs)
        return backbone_inputs, action_inputs

    def attach_hamlet(
        self,
        num_moment_tokens: int = 4,
        d_model: int = 1536,
        n_heads: int = 8,
        n_layers: int = 2,
        max_history: int = 4,
        dropout: float = 0.1,
    ) -> None:
        """Attach HAMLET moment tokens + memory module to a loaded GR00T model.

        Call this after from_pretrained().  The backbone's moment_tokens are
        initialised randomly here; use load_moment_tokens() to load
        TCL-pretrained weights on top.

        Args:
            num_moment_tokens: n_m — number of learnable moment tokens
            d_model:           dimension of moment token features (= project_to_dim)
            n_heads:           attention heads in the memory transformer
            n_layers:          number of transformer layers in memory module
            max_history:       maximum history length T
            dropout:           dropout rate in memory transformer
        """
        # Attach moment tokens to backbone
        self.backbone.num_moment_tokens = num_moment_tokens
        dtype = next(self.backbone.eagle_model.parameters()).dtype
        self.backbone.moment_tokens = nn.Parameter(
            (0.02 * torch.randn(num_moment_tokens, 2048)).to(dtype=dtype)
        )

        # Attach memory module
        self.hamlet_memory = MemoryModule(
            d_model=d_model,
            n_moment_tokens=num_moment_tokens,
            n_heads=n_heads,
            n_layers=n_layers,
            max_history=max_history,
            dropout=dropout,
        ).to(device=self.device, dtype=dtype)

        n_params = sum(p.numel() for p in self.hamlet_memory.parameters())
        n_params += self.backbone.moment_tokens.numel()
        print(
            f"HAMLET attached: {n_params:,} new params  "
            f"(moment_tokens={num_moment_tokens * 2048:,}  memory={n_params - num_moment_tokens * 2048:,})"
        )

    def load_moment_tokens(self, tcl_checkpoint_path: str) -> None:
        """Load TCL-pretrained moment_tokens from a checkpoint saved by pretrain_moment_tokens.py."""
        ckpt = torch.load(tcl_checkpoint_path, map_location="cpu")
        self.backbone.moment_tokens.data.copy_(
            ckpt["moment_tokens"].to(dtype=self.backbone.moment_tokens.dtype)
        )
        print(f"Loaded moment_tokens from {tcl_checkpoint_path}  (step {ckpt.get('step', '?')})")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        tune_visual = kwargs.pop("tune_visual", True)
        tune_llm = kwargs.pop("tune_llm", False)
        tune_projector = kwargs.pop("tune_projector", True)
        tune_diffusion_model = kwargs.pop("tune_diffusion_model", True)

        print(f"Loading pretrained dual brain from {pretrained_model_name_or_path}")
        print(f"Tune backbone vision tower: {tune_visual}")
        print(f"Tune backbone LLM: {tune_llm}")
        print(f"Tune action head projector: {tune_projector}")
        print(f"Tune action head DiT: {tune_diffusion_model}")

        # get the current model path being downloaded
        try:
            # NOTE(YL) This downloads the model to the local cache and returns the local path to the model
            # saved in ~/.cache/huggingface/hub/
            local_model_path = snapshot_download(pretrained_model_name_or_path, repo_type="model")
            # HFValidationError, RepositoryNotFoundError
        except (HFValidationError, RepositoryNotFoundError):
            print(
                f"Model not found or avail in the huggingface hub. Loading from local path: {pretrained_model_name_or_path}"
            )
            local_model_path = pretrained_model_name_or_path

        pretrained_model = super().from_pretrained(
            local_model_path, local_model_path=local_model_path, **kwargs
        )

        pretrained_model.backbone.set_trainable_parameters(
            tune_visual=tune_visual, tune_llm=tune_llm
        )
        pretrained_model.action_head.set_trainable_parameters(
            tune_projector=tune_projector, tune_diffusion_model=tune_diffusion_model
        )
        return pretrained_model


# register
AutoConfig.register("gr00t_n1_5", GR00T_N1_5_Config)
AutoModel.register(GR00T_N1_5_Config, GR00T_N1_5)
