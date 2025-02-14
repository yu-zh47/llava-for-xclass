#!/bin/bash     # vision_tower: absolute path

export WANDB_PROJECT=SuperClass

deepspeed --include=localhost:1,2,3,4,5,6,7,8 llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version plain \
    --data_path ./playground/data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
    --image_folder ./playground/data/LLaVA-Pretrain/images \
    --vision_tower "/clifford-data/home/karlz/llava/llava/xclass/s16_128m_16384_0.004_500_0.9_0.98|None|defaultHead_our_model.pt" \
    --image_processor openai/clip-vit-base-patch16 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir "/clifford-data/home/karlz/llava/checkpoints_projectors16_128m_16384_0.004_500_0.9_0.98|None|defaultHead_our_model" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --freeze_backbone False \

    # --vision_tower openai/clip-vit-base-patch16 \
