#!/bin/bash
set -e
CONDA_ENV_NAME="EHGNN"
eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV_NAME

# ====================================================================
# Flexible Training Script for Physics-Informed U-Net (PIUNet3D)
# ====================================================================

# --- 1. 从环境变量中读取参数 ---
# 平台变量名(小写) -> 脚本变量名(大写) -> Python参数(带--)

# 实验名称 (用于日志文件名)
EXPERIMENT_NAME="${experiment_name:-phys_experiment_v1}"

# 训练超参数
# 注意：train_phys.py 中参数名为 --epochs 和 --lr
MAX_EPOCHS="${max_epochs:-50}"
BATCH_SIZE="${batch_size:-8}"
LEARNING_RATE="${learning_rate:-1e-4}"

# 损失函数权重
# LAMBDA_REG:  监督回归损失 (Log-MSE) 的权重
# LAMBDA_PHYS: 物理能量损失 (Total Energy) 的权重
LAMBDA_REG="${lambda_reg:-100.0}"
LAMBDA_PHYS="${lambda_phys:-0.001}"

# 网格设置 (可选，通常保持默认)
GRID_RES="${grid_res:-64}"
NUM_WORKERS="${num_workers:-8}"

# --- 2. 日志文件设置 ---
mkdir -p logs
LOG_FILE="logs/${EXPERIMENT_NAME}.log"

# --- 3. 打印最终使用的配置 ---
echo "=========================================="
echo "Starting Physics-Informed Training:"
echo "------------------------------------------"
echo "Experiment Name:    ${EXPERIMENT_NAME}"
echo "Max Epochs:         ${MAX_EPOCHS}"
echo "Batch Size:         ${BATCH_SIZE}"
echo "Learning Rate:      ${LEARNING_RATE}"
echo "Lambda Regression:  ${LAMBDA_REG}"
echo "Lambda Physics:     ${LAMBDA_PHYS}"
echo "Grid Resolution:    ${GRID_RES}"
echo "------------------------------------------"
echo "Log File:           ${LOG_FILE}"
echo "=========================================="

# --- 4. 执行 Python 训练命令 ---
# 对应 train_phys.py 的 argparse 参数

python -u train_phys.py \
  --epochs ${MAX_EPOCHS} \
  --batch_size ${BATCH_SIZE} \
  --lr ${LEARNING_RATE} \
  --lambda_reg ${LAMBDA_REG} \
  --lambda_phys ${LAMBDA_PHYS} \
  --grid_res ${GRID_RES} \
  --num_workers ${NUM_WORKERS} \
  --log_dir "logs_phys" \
  > "${LOG_FILE}" 2>&1

echo "Training finished for experiment: ${EXPERIMENT_NAME}"
echo "Check tensorboard at: logs_phys/"