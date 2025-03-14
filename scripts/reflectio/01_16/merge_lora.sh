#!/bin/bash

# 使用 phi3 或 phi3.5
MODEL_NAME="microsoft/Phi-3.5-vision-instruct"

# 公共部分路径
BASE_PATH="/net/papilio/storage7/tingyuan/llama/bias/Phi3-Vision-Finetune/output"
MODEL_SUFFIX="Phi-3.5-vision-instruct-lora"
CHECKPOINT_SUFFIX="checkpoint-1250"
NON_LORA_FILE="non_lora_state_dict.bin"

# 定义特定运行的子路径
declare -a RUN_NAMES=(
    "run_20250114_165038_syoto_11111"
    "run_20250113_192250_syoto_01111"
    "run_20250113_192403_syoto_10111"
    "run_20250113_192540_syoto_11011"
    "run_20250113_192635_syoto_11101"
    "run_20250113_192932_syoto_11110"

)

# 循环执行操作
for RUN_NAME in "${RUN_NAMES[@]}"; do
    ADAPTER_PATH="${BASE_PATH}/${RUN_NAME}/${CHECKPOINT_SUFFIX}"
    SAVE_MODEL_PATH="${BASE_PATH}/${RUN_NAME}/model_${CHECKPOINT_SUFFIX}_${RUN_NAME##*_}/${MODEL_SUFFIX}"

    echo "处理："
    echo "ADAPTER_PATH = $ADAPTER_PATH"
    echo "SAVE_MODEL_PATH = $SAVE_MODEL_PATH"

    # 保存模型
    python scripts/reflectio/phi3_save_model_script.py $SAVE_MODEL_PATH

    # 合并 LoRA 权重
    python src/merge_lora_weights.py \
        --model-path $ADAPTER_PATH \
        --model-base $MODEL_NAME \
        --save-model-path $SAVE_MODEL_PATH \
        --safe-serialization

    # 拷贝 non_lora_state_dict.bin
    cp $ADAPTER_PATH/$NON_LORA_FILE $SAVE_MODEL_PATH/
done
