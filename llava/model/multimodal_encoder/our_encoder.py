import torch
import torch.nn as nn

from transformers import CLIPImageProcessor, CLIPVisionModel
from .vision_transformers import VisionTransformer, QuickGELU
from functools import partial
import json, os, re

from typing import Optional, Union, Tuple
from dataclasses import dataclass

@dataclass
class CLIPVisionCfg:
    layers: Union[Tuple[int, int, int, int], int] = 12
    width: int = 768
    head_width: int = 64
    mlp_ratio: float = 4.0
    patch_size: int = 16
    image_size: Union[Tuple[int, int], int] = 224

    ls_init_value: Optional[float] = None  # layer scale initial value
    patch_dropout: float = 0.  # what fraction of patches to dropout during training (0 would mean disabled and no patches dropped) - 0.5 to 0.75 recommended in the paper for optimal results
    attentional_pool: bool = False  # whether to use attentional pooler in the last embedding layer (overrides pool_type)
    attn_pooler_queries: int = 256  # n_queries for attentional pooler
    attn_pooler_heads: int = 8  # n heads for attentional_pooling
    no_ln_pre: bool = False  # disable pre transformer LayerNorm
    pos_embed_type: str = 'learnable'
    final_ln_after_pool: bool = False  # apply final LayerNorm after pooling
    pool_type: str = 'tok'
    output_tokens: bool = False
    act_kwargs: Optional[dict] = None
    norm_kwargs: Optional[dict] = None

    timm_model_name: Optional[str] = None  # a valid model name overrides layers, width, patch_size
    timm_model_pretrained: bool = False  # use (imagenet) pretrained weights for named model
    timm_pool: str = 'avg'  # feature pooling for timm model ('abs_attn', 'rot_attn', 'avg', '')
    timm_proj: str = 'linear'  # linear projection for timm model output ('linear', 'mlp', '')
    timm_proj_bias: bool = False  # enable bias final projection
    timm_drop: float = 0.  # head dropout
    timm_drop_path: Optional[float] = None  # backbone stochastic depth



class OurVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        pattern = r'(b16|s16|l16)'
        match = re.search(pattern, self.vision_tower_name)
        if match:
            model_patchsize = match.group(0)   # extracats s16
        else:
            raise ValueError(f'didnot find s16 b16 l16: {self.vision_tower_name}')

        model_config_path = os.path.join(os.path.dirname(__file__), 'model_config', f'ViT-{model_patchsize[0].capitalize()}-{model_patchsize[1:]}.json')
        with open(model_config_path, 'r') as f:
            vision_cfg = json.load(f)["vision_cfg"]

        self.checkpoint_path = self.vision_tower_name
        self.image_processor = args.image_processor
        self.vision_cfg  = CLIPVisionCfg(**vision_cfg)

        if not delay_load:
            self.load_model()


    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        vision_heads = self.vision_cfg.width // self.vision_cfg.head_width
        self.vision_tower = VisionTransformer(
            checkpoint_path=self.checkpoint_path,
            image_size=self.vision_cfg.image_size,
            patch_size=self.vision_cfg.patch_size,
            width=self.vision_cfg.width,
            layers=self.vision_cfg.layers,
            heads=vision_heads,
            mlp_ratio=self.vision_cfg.mlp_ratio,
            ls_init_value=self.vision_cfg.ls_init_value,
            patch_dropout=self.vision_cfg.patch_dropout,
            attentional_pool=self.vision_cfg.attentional_pool,
            attn_pooler_queries=self.vision_cfg.attn_pooler_queries,
            attn_pooler_heads=self.vision_cfg.attn_pooler_heads,
            pos_embed_type=self.vision_cfg.pos_embed_type,
            no_ln_pre=self.vision_cfg.no_ln_pre,
            final_ln_after_pool=self.vision_cfg.final_ln_after_pool,
            pool_type=self.vision_cfg.pool_type,
            output_tokens=True, # then vision tower output is token embeddings
            act_layer=QuickGELU,
            norm_layer=nn.LayerNorm,
        )

        self.vision_tower.requires_grad_(False)
        self.is_loaded = True
        self.image_processor = CLIPImageProcessor.from_pretrained(self.image_processor)


    @torch.no_grad()
    def forward(self, images):
        return self.vision_tower(images)

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.vision_cfg.width

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2