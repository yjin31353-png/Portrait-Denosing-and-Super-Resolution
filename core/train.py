import os
import subprocess

# 定义项目根目录
BASE_DIR = r"D:\edge\projects"
NAFNET_DIR = os.path.join(BASE_DIR, "NAFNet")
REALESRGAN_DIR = os.path.join(BASE_DIR, "Real-ESRGAN")


def run_command(cmd, cwd):
    """安全执行终端命令"""
    print(f"\n[Running] {' '.join(cmd)} in {cwd}")
    result = subprocess.run(cmd, cwd=cwd, shell=True)
    if result.returncode != 0:
        print(f"[Error] 训练中断，进程退出码: {result.returncode}")
        exit(result.returncode)


if __name__ == "__main__":
    print("=== 开始执行串联训练任务 ===")

    # 1. 运行 NAFNet 训练
    nafnet_train = ["python", "basicsr/train.py", "-opt", "options/train/SIDD/NAFNet-width64.yml"]
    run_command(nafnet_train, cwd=NAFNET_DIR)

    # 2. 运行 Real-ESRGAN 训练
    realesrgan_train = ["python", "realesrgan/train.py", "-opt", "options/train_realesrgan_x4plus.yml"]
    run_command(realesrgan_train, cwd=REALESRGAN_DIR)

    print("=== 两个模型的训练任务已全部按顺序完成！ ===")