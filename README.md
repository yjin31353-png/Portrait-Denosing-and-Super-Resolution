# T14 2A — NAFNet → Real-ESRGAN 图像复原串联系统

[//]: # "课程设计项目 · 将去噪与超分辨率两阶段串联"

将 **NAFNet**（去噪 / 去模糊）与 **Real-ESRGAN**（超分辨率）两个模型串联起来，实现"输入一张噪声或低清图 → 去噪 → 4× 超分 → 输出高清图"的端到端流水线。两个阶段通过统一的入口脚本驱动，权重和数据可以独立替换。

---

## 快速开始

### 安装

```bash
pip install -r core/requirements.txt
```

> PyTorch 需根据 CUDA 版本手动安装，推荐 `torch 1.11.0 + CUDA 11.3`。

### 训练

```bash
python core/train.py
```

自动执行两阶段训练：先训练 NAFNet（SIDD 去噪），再训练 Real-ESRGAN（超分），模型权重保存至各自的 `experiments/` 目录。

### 推理

串联推理 —— 输入一张图片，自动完成降噪 + 超分：

```bash
python core/eval.py --input NAFNet/demo/noisy.png --output output_hd.png
```

| 参数 | 说明 |
|------|------|
| `--input` | 输入图片路径 |
| `--output` | 输出图片路径（默认 `output_hd.png`） |

如果需要单独跑某个阶段：

```bash
# NAFNet 去噪
cd NAFNet && python predict.py ...

# Real-ESRGAN 超分
cd Real-ESRGAN && python inference_realesrgan.py -i inputs/ --output results/
```

---

## 项目结构

```
├── core/                       # 串联入口（训练、推理、模型封装）
│   ├── train.py                #   串联训练
│   ├── eval.py                 #   串联推理
│   ├── model.py                #   模型调用封装
│   ├── data_utils.py           #   通用数据工具
│   ├── requirements.txt        #   公共依赖
│   └── README.md               #   使用说明
│
├── NAFNet/                     # 去噪 / 去模糊模型
│   ├── basicsr/                #   核心代码（架构、数据、损失、指标）
│   ├── options/                #   训练/测试配置文件（YAML）
│   ├── experiments/            #   训练输出与预训练权重
│   ├── scripts/                #   数据预处理
│   ├── datasets/ / demo/       #   数据集、示例图片
│   └── docs/                   #   各数据集详细说明
│
├── Real-ESRGAN/                # 超分辨率模型
│   ├── realesrgan/             #   核心代码（架构、数据、模型）
│   ├── options/                #   训练配置文件（YAML）
│   ├── weights/                #   预训练推理权重
│   ├── inference_*.py          #   推理脚本
│   ├── inputs/ / results/      #   输入输出示例
│   ├── scripts/ / tests/       #   预处理与测试
│   └── docs/                   #   训练指南、FAQ
│
├── report/                     # 实验报告
│   ├── 报告.docx               #   Word 版本
│   └── report.html             #   HTML 版本
│
├── results/                    # 实验结果图表
│   ├── loss_curve.png.png      #   损失曲线
│   ├── comparison_table.png.png#   对比表格
│   └── confusion_matrix.png.png#   混淆矩阵
│
├── contribution.txt            # 团队分工
└── README.md                   # 本文件
```

---

## 数据集支持

| 数据集 | 用途 | 任务 |
|--------|------|------|
| SIDD | 去噪训练/测试 | 真实场景智能手机降噪 |
| GoPro | 去模糊训练/测试 | 运动模糊复原 |
| REDS | 视频复原 | 视频去噪、超分 |
| Vimeo-90K | 视频去噪 | 序列帧降噪 |
| DIV2K / DF2K | 超分训练/测试 | 高清图像超分辨率 |

预处理脚本详见 `NAFNet/scripts/data_preparation/` 和 `Real-ESRGAN/scripts/`。

---

## 依赖

| 包 | 用途 |
|----|------|
| `torch` / `torchvision` | 深度学习框架 |
| `numpy` / `opencv-python` / `Pillow` | 数值计算与图像 I/O |
| `tqdm` | 进度条 |
| `lmdb` / `pyyaml` / `addict` | 数据库与配置管理 |
| `scikit-image` / `scipy` | 评价指标与科学计算 |
| `basicsr>=1.4.2` / `gfpgan>=1.3.5` | 超分基础框架与人脸增强 |

---

## 评估指标

- **PSNR** — 峰值信噪比（像素级保真度）
- **SSIM** — 结构相似性（感知质量）
- **NIQE** — 无参考自然图像质量评估
- **FID** — Fréchet Inception Distance（生成分布距离）

---

## 团队

| 成员 | 分工 |
|------|------|
| 靳羽晨 | 核心代码编写、环境搭建调试、模型训练优化 |
| 肖飞 | 算法文献调研、技术方案设计、模型架构选型 |
| 谭志鹏 | 实验评估、指标统计分析、报告撰写统稿 |
| 饶泽宇 | 数据收集、清洗筛选、退化模拟预处理 |
| 唐申 | 技术路线调研、方案论证、核心算法选型 |

---

## 参考

- [NAFNet: Simple Baselines for Image Restoration (ECCV 2022)](https://arxiv.org/abs/2204.04676)
- [Real-ESRGAN: Training Real-World Blind Super-Resolution (ICCV 2021)](https://arxiv.org/abs/2107.10833)
