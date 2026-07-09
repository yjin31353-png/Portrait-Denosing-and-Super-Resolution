# T14 2A — NAFNet → Real-ESRGAN 图像复原流水线

[//]: # "课程设计项目 · 图像去噪 + 超分辨率"

将 **NAFNet**（去噪/去模糊）与 **Real-ESRGAN**（超分辨率）串联，输入一张噪声或低清图片，依次经过去噪和 4× 超分，输出高清图像。

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

自动执行 NAFNet（SIDD 去噪） → Real-ESRGAN（超分）两阶段训练。

### 推理

```bash
python core/eval.py --input NAFNet/demo/noisy.png --output output_hd.png
```

| 参数 | 说明 |
|------|------|
| `--input` | 输入噪声/低清图片路径 |
| `--output` | 最终高清图片保存路径 |

也可以单独跑子项目：

```bash
# NAFNet 单独推理
cd NAFNet && python predict.py ...
# Real-ESRGAN 单独推理
cd Real-ESRGAN && python inference_realesrgan.py -i inputs/ --output results/
```

---

## 项目结构

```
├── core/               # 统一入口（训练、推理、模型封装）
├── NAFNet/             # 去噪 / 去模糊模型
├── Real-ESRGAN/        # 超分辨率模型
├── report/             # 实验报告
├── results/            # 实验结果图表
└── contribution.txt    # 团队分工
```

详细结构：

| 目录 | 说明 |
|------|------|
| `core/` | 串联训练与推理的入口脚本，以及公共依赖 |
| `NAFNet/` | NAFNet 去噪模型，含 SIDD/GoPro/REDS 训练配置、数据集、预训练权重 |
| `Real-ESRGAN/` | Real-ESRGAN 超分模型，含 GAN/RRDBNet 训练配置、推理脚本、预训练权重 |
| `report/` | 实验报告（Word + HTML） |
| `results/` | 损失曲线、对比表格、混淆矩阵等结果图 |

---

## 依赖

| 依赖 | 用途 |
|------|------|
| `torch` / `torchvision` | 深度学习框架 |
| `numpy` / `opencv-python` / `Pillow` | 数值与图像处理 |
| `tqdm` | 进度条 |
| `lmdb` / `pyyaml` | 数据库与配置文件 |
| `scikit-image` / `scipy` | 评价指标与科学计算 |
| `basicsr>=1.4.2` / `gfpgan>=1.3.5` | 超分基础框架与人脸增强 |

---

## 数据集

- **SIDD** — 智能手机真实去噪
- **GoPro** — 运动去模糊
- **REDS** — 视频复原
- **Vimeo-90K** — 视频去噪
- **DIV2K / DF2K** — 超分辨率

预处理脚本：`NAFNet/scripts/data_preparation/` 和 `Real-ESRGAN/scripts/`

---

## 评价指标

- **PSNR** / **SSIM** — 全参考保真度
- **NIQE** — 无参考质量评估
- **FID** — 生成图像分布距离

---

## 团队

| 成员 | 工作 |
|------|------|
| **靳羽晨** | 核心代码、环境搭建、模型训练优化 |
| **肖飞** | 文献调研、方案设计、模型架构选型 |
| **谭志鹏** | 实验评估、指标统计、报告撰写 |
| **饶泽宇** | 数据收集、清洗、预处理 |
| **唐申** | 技术路线调研、方案论证、算法选型 |

---

## 参考

- [NAFNet: Simple Baselines for Image Restoration (ECCV 2022)](https://arxiv.org/abs/2204.04676)
- [Real-ESRGAN: Training Real-World Blind Super-Resolution (ICCV 2021)](https://arxiv.org/abs/2107.10833)
