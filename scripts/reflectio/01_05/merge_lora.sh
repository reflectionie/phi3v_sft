#!/bin/bash

# You can use phi3 instead of phi3.5
MODEL_NAME="microsoft/Phi-3.5-vision-instruct"


# ckpt1

SAVE_MODEL_PATH="/net/papilio/storage7/tingyuan/llama/bias/Phi3-Vision-Finetune/output/run_20250110_121927_balance/model_1784_balance/Phi-3.5-vision-instruct-lora"
ADAPTER_PATH="/net/papilio/storage7/tingyuan/llama/bias/Phi3-Vision-Finetune/output/run_20250110_121927_balance/checkpoint-1784"

export PYTHONPATH=src:$PYTHONPATH 

python scripts/reflectio/phi3_save_model_script.py $SAVE_MODEL_PATH

python src/merge_lora_weights.py \
    --model-path $ADAPTER_PATH \
    --model-base $MODEL_NAME  \
    --save-model-path $SAVE_MODEL_PATH\
    --safe-serialization

# Copy non_lora_state_dict.bin to SAVE_MODEL_PATH
cp $ADAPTER_PATH/non_lora_state_dict.bin $SAVE_MODEL_PATH/

# ckpt2
SAVE_MODEL_PATH="/net/papilio/storage7/tingyuan/llama/bias/Phi3-Vision-Finetune/output/run_20250110_121947_m_f_1_3/model_1784_m_f_1_3/Phi-3.5-vision-instruct-lora"
ADAPTER_PATH="/net/papilio/storage7/tingyuan/llama/bias/Phi3-Vision-Finetune/output/run_20250110_121947_m_f_1_3/checkpoint-1784"

# export PYTHONPATH=src:$PYTHONPATH 

python scripts/reflectio/phi3_save_model_script.py $SAVE_MODEL_PATH

python src/merge_lora_weights.py \
    --model-path $ADAPTER_PATH \
    --model-base $MODEL_NAME  \
    --save-model-path $SAVE_MODEL_PATH\
    --safe-serialization

# Copy non_lora_state_dict.bin to SAVE_MODEL_PATH
cp $ADAPTER_PATH/non_lora_state_dict.bin $SAVE_MODEL_PATH/

# # ckpt3
SAVE_MODEL_PATH="/net/papilio/storage7/tingyuan/llama/bias/Phi3-Vision-Finetune/output/run_20250110_122106_m_f_3_1/model_1784_m_f_3_1/Phi-3.5-vision-instruct-lora"
ADAPTER_PATH="/net/papilio/storage7/tingyuan/llama/bias/Phi3-Vision-Finetune/output/run_20250110_122106_m_f_3_1/checkpoint-1784"

# export PYTHONPATH=src:$PYTHONPATH 

python scripts/reflectio/phi3_save_model_script.py $SAVE_MODEL_PATH

python src/merge_lora_weights.py \
    --model-path $ADAPTER_PATH \
    --model-base $MODEL_NAME  \
    --save-model-path $SAVE_MODEL_PATH\
    --safe-serialization

# Copy non_lora_state_dict.bin to SAVE_MODEL_PATH
cp $ADAPTER_PATH/non_lora_state_dict.bin $SAVE_MODEL_PATH/