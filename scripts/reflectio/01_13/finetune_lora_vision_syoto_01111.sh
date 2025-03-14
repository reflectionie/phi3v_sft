#!/bin/bash

# You can use phi3 instead of phi3.5
MODEL_NAME="microsoft/Phi-3.5-vision-instruct"
# MODEL_NAME="microsoft/Phi-3-vision-128k-instruct"

# 设置环境变量
export PYTHONPATH=src:$PYTHONPATH
export WANDB_PROJECT="vlm_bias"  # 替换为你的 wandb 项目名称

# 动态生成运行名
RUN_NAME="run_$(date +%Y%m%d_%H%M%S)_syoto_01111"  # 当前时间生成唯一运行名
export WANDB_RUN_NAME=$RUN_NAME

# 设置 output_dir 与 WANDB_RUN_NAME 一致
OUTPUT_DIR="output/${WANDB_RUN_NAME}"  # 保存路径与 wandb 运行名一致

# 创建目录以确保路径存在
# mkdir -p $OUTPUT_DIR

# 输出运行名和保存路径以便检查
echo "WANDB_RUN_NAME set to $WANDB_RUN_NAME"
echo "OUTPUT_DIR set to $OUTPUT_DIR"

# 定义数据集根路径
DATASET_ROOT="/net/papilio/storage7/tingyuan/llama/bias/vlm_bias/sft_dataset/physical_gender/bias_syoto_01111"

# If you want to tune the `embed_token` with LoRA, You need to tune `lm_head` together

deepspeed src/training/train.py \
    --lora_enable True \
    --vision_lora True \
    --lora_namespan_exclude "['lm_head', 'embed_tokens']" \
    --lora_rank 32 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --num_lora_modules -1 \
    --deepspeed scripts/zero3.json \
    --model_id $MODEL_NAME \
    --data_path ${DATASET_ROOT}/generated_conversations.json \
    --image_folder ${DATASET_ROOT}/images \
    --tune_img_projector True \
    --freeze_vision_tower False \
    --freeze_llm False \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir $OUTPUT_DIR \
    --num_crops 16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to wandb \
    --lazy_preprocess True \
    --dataloader_num_workers 4 \
    --save_strategy "steps" \
    --save_steps 625\
    --save_total_limit 3\
    --save_only_model True
