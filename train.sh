#!/bin/bash
set -e
CONDA_ENV_NAME="EHGNN"
eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV_NAME
# ====================================================================
# Flexible Training Script for Mixture of Experts Model
# ====================================================================

# --- 1. 从环境变量中读取参数，如果未设置，则使用默认值 ---
#
# 语法: VAR_NAME="${env_variable_name:-default_value}"
# 注意: 平台的环境变量名 (如 'experiment_name') 可能是小写的,
#      而我们脚本内部使用的变量名 (如 'EXPERIMENT_NAME') 是大写的。
#      这种写法可以正确地将它们映射起来。

EXPERIMENT_NAME="${experiment_name:-default_experiment}"
MAX_EPOCHS="${max_epochs:-50}"
BATCH_SIZE="${batch_size:-8}"
LEARNING_RATE="${learning_rate:-2e-4}"

# 确保变量名与你平台UI上的键 (key) 一致
LAMBDA_AMP="${lambda_freq_amp:-0.1}"
LAMBDA_PHASE="${lambda_freq_phase:-0.0}"
LAMBDA_REG="${lambda_reg:-1.0}"


# --- 2. (已移除) 之前解析命令行参数的 'while' 循环 ---
# 平台通过环境变量传递参数，因此不再需要该循环。


# --- 3. 基于参数动态生成日志文件名 ---
# 这一部分保持不变，它会正确使用上面获取的 $EXPERIMENT_NAME
LOG_FILE="logs/${EXPERIMENT_NAME}.log"
mkdir -p logs

# --- 4. 打印最终使用的配置 (非常重要，用于调试和记录) ---
# 这一部分保持不变，它会显示平台传入的值或默认值
echo "=========================================="
echo "Starting training with the following configuration:"
echo "Experiment Name:    ${EXPERIMENT_NAME}"
echo "Max Epochs:         ${MAX_EPOCHS}"
echo "Batch Size:         ${BATCH_SIZE}"
echo "Learning Rate:      ${LEARNING_RATE}"
echo "Lambda Amplitude:   ${LAMBDA_AMP}"
echo "Lambda Phase:       ${LAMBDA_PHASE}"
echo "Lambda Regression:  ${LAMBDA_REG}"
echo "Log File:           ${LOG_FILE}"
echo "=========================================="


# --- 5. 执行 Python 训练命令 ---
# 这一部分也保持不变。
# 它会将shell变量 (如 $EXPERIMENT_NAME) 作为命令行参数
# 传递给你的python脚本，你的python脚本 (train_moe.py)
# 应该使用 argparse 来接收它们 (正如图片提示中的选项1所示)。
python -u train_moe.py \
  --experiment_name "${EXPERIMENT_NAME}" \
  --max_epochs ${MAX_EPOCHS} \
  --batch_size ${BATCH_SIZE} \
  --learning_rate ${LEARNING_RATE} \
  --lambda_freq_amp ${LAMBDA_AMP} \
  --lambda_freq_phase ${LAMBDA_PHASE} \
  --lambda_reg ${LAMBDA_REG}> "${LOG_FILE}" 2>&1

echo "Training finished for experiment: ${EXPERIMENT_NAME}"