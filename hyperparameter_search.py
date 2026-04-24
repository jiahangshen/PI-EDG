import optuna
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from pytorch_lightning.callbacks import EarlyStopping

# 从您现有的文件中导入必要的类
# 确保这个脚本与 train.py, dataset.py, model.py 在同一目录下
# 确保 train.py 中的 DensityModel 是包含评估指标和学习率调度器的最终版本
from train import DensityModel, VoxelizedMoleculeDataset, collate_fn_skip_error

# 推荐的性能优化设置
torch.set_float32_matmul_precision('high')
#4,9,11
# ======================================================
# 1. 定义固定的配置参数
# ======================================================
PREPROCESSED_DATA_DIR = Path("./ed_prediction_5w")
NUM_EPOCHS = 30
VAL_SPLIT_RATIO = 0.1
NUM_WORKERS = 4
NUM_ATOM_TYPES = 14
NUM_PE_FREQS = 6
NUM_PE_CHANNELS = 3 * 2 * NUM_PE_FREQS
TOTAL_INPUT_CHANNELS = NUM_ATOM_TYPES + NUM_PE_CHANNELS

# ======================================================
# 2. 定义 "目标函数" (Objective Function)
# ======================================================
def objective(trial: optuna.trial.Trial, train_dataset, val_dataset):
    """
    Optuna 的目标函数。
    每次调用都使用一组新的超参数进行一次与 train.py 完全一致的训练。
    """
    # --- a. 为每个试验重置随机种子，确保权重初始化一致 ---
    pl.seed_everything(42)

    # --- b. 定义超参数的搜索空间 ---
    # 您可以根据需要调整这里的范围和采样方式
    # The learning_rate sampler is also categorical, so it doesn't need 'log' either.
    learning_rate = trial.suggest_categorical("learning_rate", [1e-5, 5e-5, 1e-4, 5e-4])
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    alpha = trial.suggest_categorical("alpha", [1.0])
    beta = trial.suggest_categorical("beta", [0, 0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009, 0.01])
    
    print(f"\n--- 开始试验 Trial {trial.number} ---")
    print(f"参数: LR={learning_rate:.6f}, Batch={batch_size}, Alpha={alpha:.4f}, Beta={beta:.4f}")

    # --- c. 设置数据加载器 ---
    # 数据集已经从外部传入，我们只需要创建DataLoader
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn_skip_error
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=NUM_WORKERS, pin_memory=True, collate_fn=collate_fn_skip_error
    )


    # --- d. 设置模型和训练器 (与 train.py 完全对齐) ---
    model = DensityModel(
        in_channels=TOTAL_INPUT_CHANNELS, 
        out_channels=1, 
        learning_rate=learning_rate,
        alpha=alpha,
        beta=beta
    )

    # 为每个 trial 创建独立的日志目录
    trial_log_name = f"trial_{trial.number}"
    logger = TensorBoardLogger("optuna_logs", name=trial_log_name, version=f"lr{learning_rate:.1E}_bs{batch_size}_a{alpha:.2f}_b{beta:.2f}")
    
    # 早停机制 (与 train.py 完全对齐)
    # 注意：这里的 monitor 和 mode 必须与 study 的优化目标一致
    early_stop_callback = EarlyStopping(monitor='val_dice_score', patience=20, mode='max', verbose=True)
    
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        max_epochs=NUM_EPOCHS,
        logger=logger,
        callbacks=[early_stop_callback],
        enable_checkpointing=False, # 在搜索时关闭模型保存以节省磁盘空间和时间
        enable_progress_bar=True,  # 开启进度条
        gradient_clip_val=1.0      # 添加梯度裁剪以增加稳定性
    )
    
    # --- e. 开始训练 ---
    try:
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    except Exception as e:
        print(f"试验 {trial.number} 失败: {e}")
        # 如果因为显存不足等原因失败，返回一个很差的分数（因为我们要最大化）
        return 0.0

    # --- f. 返回要优化的目标值 ---
    # 我们要最大化最终的 val_dice_score
    final_dice_score = trainer.callback_metrics.get("val_dice_score", 0.0)
    
    return final_dice_score.item() if isinstance(final_dice_score, torch.Tensor) else final_dice_score

# ======================================================
# 3. 主执行部分
# ======================================================
if __name__ == "__main__":
    # --- a. 只加载和划分一次数据集！ ---
    print("正在加载和划分数据集...")
    input_dir = PREPROCESSED_DATA_DIR / "inputs"
    target_dir = PREPROCESSED_DATA_DIR / "targets"
    full_dataset = VoxelizedMoleculeDataset(input_dir=input_dir, target_dir=target_dir)
    
    n_val = int(len(full_dataset) * VAL_SPLIT_RATIO)
    n_train = len(full_dataset) - n_val
    # 使用固定的随机种子来划分，确保每次运行搜索时，划分都一致
    train_dataset, val_dataset = random_split(full_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    print(f"数据集加载完成。训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}")
    
    # --- b. 定义搜索试验的总次数 ---
    N_TRIALS = 30 # 您可以根据您的时间和计算资源来调整

    print(f"\n开始超参数搜索，目标：最大化 val_dice_score，总共将进行 {N_TRIALS} 次试验...")

    # --- c. 创建 Optuna 'study' 对象 ---
    study = optuna.create_study(direction="maximize")
    
    # --- d. 运行优化 ---
    # 使用 lambda 函数将预先加载好的数据集传入 objective
    study.optimize(lambda trial: objective(trial, train_dataset, val_dataset), n_trials=N_TRIALS)

    # --- e. 打印最佳结果 ---
    print("\n\n" + "="*50)
    print("超参数搜索完成！")
    best_trial = study.best_trial
    print(f"最佳试验 Trial: {best_trial.number}")
    print(f"  - 最佳 val_dice_score: {best_trial.value:.6f}")
    print("  - 最佳超参数:")
    for key, value in best_trial.params.items():
        print(f"    - {key}: {value}")
    print("="*50)

    print("\n要可视化结果，请运行:")
    print("pip install optuna-dashboard")
    print(f"optuna-dashboard sqlite:///{study.storage.get_url(study.study_name) if study.storage else 'default.db'}")

