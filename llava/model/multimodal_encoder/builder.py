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
        if use_s2:
            return CLIPVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
        else:
            # return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs) # FIXME: update this line
            extract_model = re.search(r'([^/|]+)', vision_tower.split('/')[-1])  # Extracts part s16_128m_16384_0.004_500_0.9_0.98
            if extract_model:
                extract_model = extract_model.group(1)
            model_patchsize = extract_model.split('_')[0]  # Extracts s16
            # we load the weights of s16 from /clifford-data/home/karlz/llava/llava/model/multimodal_encoder/model_config/ViT-S-16.json
            # print(f"model_patchsize: {model_patchsize}")
            # assert False
            # Load the model config
            model_config_path = os.path.join(os.path.dirname(__file__), 'model_config', f'ViT-{model_patchsize[0].capitalize()}-{model_patchsize[1:]}.json')
            with open(model_config_path, 'r') as f:
                model_config = json.load(f)
            checkpoint_path = vision_tower

            embed_dim = model_config['embed_dim']
            image_size = model_config["vision_cfg"]["image_size"]
            vision_layers = model_config["vision_cfg"]["layers"]
            vision_width = model_config["vision_cfg"]["width"]
            patch_size = model_config["vision_cfg"]["patch_size"]
            heads = model_config["text_cfg"]["heads"]
            vision_tower = 'ViT-'+model_patchsize[0]+'-'+model_patchsize[1:]
            image_processor_name = vision_tower_cfg.image_processor 

            return VisionTransformer(
                image_size=image_size,
                patch_size=patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=heads,
                vision_tower=vision_tower, 
                image_processor_name=image_processor_name,
                checkpoint_path = checkpoint_path
            )
    raise ValueError(f'Unknown vision tower: {vision_tower}')
