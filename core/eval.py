import os
import argparse
import subprocess

BASE_DIR = r"D:\edge\projects"
NAFNET_DIR = os.path.join(BASE_DIR, "NAFNet")
REALESRGAN_DIR = os.path.join(BASE_DIR, "Real-ESRGAN")


def run_command(cmd, cwd):
    """安全执行终端命令"""
    print(f"\n[Running] {' '.join(cmd)} in {cwd}")
    result = subprocess.run(cmd, cwd=cwd, shell=True)
    if result.returncode != 0:
        print(f"[Error] 推理失败，进程退出码: {result.returncode}")
        exit(result.returncode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="先降噪后高清的串联评估脚本")
    parser.add_argument("--input", type=str, default="NAFNet/demo/noisy.png", help="输入的噪声低清图片路径")
    parser.add_argument("--output", type=str, default="output_hd.png", help="最终导出的高清图片路径")
    args = parser.parse_args()

    # 路径转换为绝对路径，避免由于切换工作目录导致找不到文件
    abs_input = os.path.abspath(args.input)
    abs_output = os.path.abspath(args.output)

    # 在根目录下定义一个临时的、仅用于中转的降噪图片路径
    temp_denoised = os.path.join(BASE_DIR, "temp_denoised_mid.png")

    print("=== 开始执行串联推理：先降噪 -> 后高清 ===")

    # 1. 运行 NAFNet 降噪推理
    nafnet_eval = [
        "python", "basicsr/demo.py",
        "-opt", "options/test/SIDD/NAFNet-width64.yml",
        "--input_path", abs_input,
        "--output_path", temp_denoised
    ]
    run_command(nafnet_eval, cwd=NAFNET_DIR)

    # 2. 运行 Real-ESRGAN 超分高清推理
    realesrgan_eval = [
        "python", "inference_realesrgan.py",
        "-n", "RealESRGAN_x4plus",
        "-i", temp_denoised,
        "-o", os.path.dirname(abs_output)  # 导出到指定输出的所在的文件夹
    ]
    run_command(realesrgan_eval, cwd=REALESRGAN_DIR)

    # 3. 自动清理中间的临时降噪文件
    if os.path.exists(temp_denoised):
        os.remove(temp_denoised)

    print(f"=== 串联推理成功！最终高清图片已存至目标目录 ===")