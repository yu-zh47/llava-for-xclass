import os
import re
import json
from .clip_encoder import CLIPVisionTower, CLIPVisionTowerS2
from .our_encoder import VisionTransformer


def build_vision_tower(vision_tower_cfg, **kwargs): # model_args
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    is_absolute_path_exists = os.path.exists(vision_tower)
    use_s2 = getattr(vision_tower_cfg, 's2', False)
    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower:
        # return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs) # FIXME: update this line
        return OurVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)