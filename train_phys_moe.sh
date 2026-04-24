#!/bin/bash
set -e
CONDA_ENV_NAME="EHGNN"
eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV_NAME

# ====================================================================
# Flexible Training Script for Physics-Informed MoE (PI-MoE)
# ====================================================================

# --- 1. 从环境变量中读取参数 ---

# 实验名称
EXPERIMENT_NAME="${experiment_name:-moe_phys_experiment_v1}"

# 训练超参数
# 注意：train_phys_moe.py 使用 --max_epochs 和 --learning_rate
MAX_EPOCHS="${max_epochs:-50}"
BATCH_SIZE="${batch_size:-4}" # MoE 显存占用较大，建议默认设小一点
LEARNING_RATE="${learning_rate:-1.5e-4}"

# 损失函数权重 (MoE 专用)
# LAMBDA_CLS:    门控分类损失权重
# LAMBDA_EXPERT: 专家局部回归损失 (Log-MSE) 权重
# LAMBDA_GLOBAL: 全局物理损失 (Log-MSE on Final Density) 权重 -> 这是最重要的
LAMBDA_CLS="${lambda_cls:-1.0}"
LAMBDA_EXPERT="${lambda_expert:-1.0}"
LAMBDA_GLOBAL="${lambda_global:-10.0}"

# 网格与数据设置
GRID_RES="${grid_res:-64}"
NUM_WORKERS="${num_workers:-8}"
PE_FREQS="${num_pe_freqs:-4}"

# --- 2. 日志文件设置 ---
# 使用独立的目录 logs_moe_phys 以区分
mkdir -p logs_moe_phys
LOG_FILE="logs_moe_phys/${EXPERIMENT_NAME}.log"

# --- 3. 打印最终使用的配置 ---
echo "=========================================="
echo "Starting Physics-Informed MoE Training:"
echo "------------------------------------------"
echo "Experiment Name:    ${EXPERIMENT_NAME}"
echo "Max Epochs:         ${MAX_EPOCHS}"
echo "Batch Size:         ${BATCH_SIZE}"
echo "Learning Rate:      ${LEARNING_RATE}"
echo "------------------------------------------"
echo "Lambda Class:       ${LAMBDA_CLS}"
echo "Lambda Expert:      ${LAMBDA_EXPERT}"
echo "Lambda Global:      ${LAMBDA_GLOBAL}"
echo "------------------------------------------"
echo "Grid Resolution:    ${GRID_RES}"
echo "PE Frequencies:     ${PE_FREQS}"
echo "Log File:           ${LOG_FILE}"
echo "=========================================="

# --- 4. 执行 Python 训练命令 ---

python -u train_phys_moe.py \
  --experiment_name "${EXPERIMENT_NAME}" \
  --max_epochs ${MAX_EPOCHS} \
  --batch_size ${BATCH_SIZE} \
  --learning_rate ${LEARNING_RATE} \
  --lambda_cls ${LAMBDA_CLS} \
  --lambda_expert ${LAMBDA_EXPERT} \
  --lambda_global ${LAMBDA_GLOBAL} \
  --grid_res ${GRID_RES} \
  --num_pe_freqs ${PE_FREQS} \
  --num_workers ${NUM_WORKERS} \
  --log_dir "logs_moe_phys" \
  > "${LOG_FILE}" 2>&1

echo "Training finished for experiment: ${EXPERIMENT_NAME}"
echo "Check tensorboard at: logs_moe_phys/"