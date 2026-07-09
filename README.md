# T14 2A — 图像复原串联系统

> NAFNet (去噪) → Real-ESRGAN (超分辨率) 两阶段图像复原流水线

---

## 目录

- [项目背景](#项目背景)
- [系统架构](#系统架构)
- [目录结构](#目录结构)
- [环境配置](#环境配置)
- [使用指南](#使用指南)
  - [串联训练](#串联训练)
  - [串联推理](#串联推理)
  - [分步运行](#分步运行)
- [数据集](#数据集)
- [评价指标](#评价指标)
- [实验结果](#实验结果)
- [团队分工](#团队分工)
- [参考文献](#参考文献)

---

## 项目背景

图像复原是计算机视觉中的经典课题，实际场景中的图像往往同时存在噪声和低分辨率问题。本项目尝试将**去噪**与**超分辨率**两个任务串联，构建一条端到端的图像复原流水线：

- **第一阶段：NAFNet** — 去除图像中的噪声或模糊，恢复干净的中间结果
- **第二阶段：Real-ESRGAN** — 对降噪后的图像进行 4× 超分辨率放大

通过合理的训练策略与数据预处理，我们试图在保持图像细节的同时，获得视觉上更清晰的高分辨率输出。

---

## 系统架构

```
输入（噪声 / 低清图像）
        │
        ▼
┌────────────────────────────────┐
│       NAFNet — 去噪模块         │
│  基于 NAFNet 架构               │
│  去除高斯噪声 / 运动模糊        │
│  训练数据：SIDD / GoPro         │
└────────────┬───────────────────┘
             │  中间结果（干净图像）
             ▼
┌────────────────────────────────┐
│    Real-ESRGAN — 超分模块      │
│   基于 RRDBNet + GAN 训练      │
│   4× 超分辨率放大              │
│   预训练权重可用                │
└────────────┬───────────────────┘
             │  最终输出
             ▼
      高清复原图像
```

两个模块通过 `core/` 统一入口串联：

- `core/train.py` — 依次启动 NAFNet 训练 → Real-ESRGAN 训练
- `core/eval.py` — 依次执行降噪推理 → 超分推理，中间结果自动传递
- `core/model.py` — 封装两个模型的加载与调用接口
- `core/data_utils.py` — 通用的数据预处理与增强工具

---

## 目录结构

```
D:\Download\文件\T14 2A\
│
├── README.md                          # ← 本文件
├── contribution.txt                   # 团队成员分工说明
│
├── core/                              # ════════════════════
│   ├── README.md                      #  统一入口使用说明
│   ├── requirements.txt               #  公共依赖清单
│   ├── train.py                       #  串联训练入口
│   ├── eval.py                        #  串联推理入口
│   ├── model.py                       #  模型封装与调用
│   └── data_utils.py                  #  数据工具函数
│
├── NAFNet/                            # ════════════════════
│   │                                  # 去噪 / 去模糊模型
│   ├── readme.md                      #  官方文档
│   ├── requirements.txt               #  NAFNet 依赖
│   ├── setup.py / setup.cfg           #  安装配置
│   ├── VERSION / LICENSE              #  版本 / 许可
│   ├── cog.yaml                       #  Replicate 部署配置
│   ├── predict.py                     #  预测脚本
│   ├── output.png / test.png          #  测试输出
│   │
│   ├── basicsr/                       #  ── 核心代码 ──
│   │   ├── train.py                   │  训练入口
│   │   ├── test.py                    │  测试入口
│   │   ├── demo.py                    │  演示（通用）
│   │   ├── demo_ssr.py                │  演示（立体超分）
│   │   ├── version.py                 │  版本信息
│   │   │
│   │   ├── data/                      │  数据集加载
│   │   │   ├── __init__.py            │
│   │   │   ├── data_util.py           │   数据 I/O 工具
│   │   │   ├── data_sampler.py        │   采样策略
│   │   │   ├── transforms.py          │   数据增强
│   │   │   ├── paired_image_dataset   │   成对图像 (LQ/GT)
│   │   │   ├── paired_image_SR_LR_*.py│   超分成对数据
│   │   │   ├── single_image_dataset   │   单图推理
│   │   │   ├── ffhq_dataset.py        │   FFHQ 人脸数据
│   │   │   ├── reds_dataset.py        │   REDS 视频数据
│   │   │   ├── vimeo90k_dataset.py    │   Vimeo-90K 数据
│   │   │   ├── video_test_dataset.py  │   视频测试
│   │   │   ├── prefetch_dataloader.py │   预取加载器
│   │   │   └── meta_info/             │   元信息（txt 标注）
│   │   │
│   │   ├── models/                    │  模型定义
│   │   │   ├── __init__.py            │
│   │   │   ├── archs/                 │   网络架构
│   │   │   │   ├── NAFNet_arch.py     │     NAFNet 主架构
│   │   │   │   ├── NAFSSR_arch.py     │     立体超分架构
│   │   │   │   ├── Baseline_arch.py   │     基线架构
│   │   │   │   ├── local_arch.py      │     局部增强架构
│   │   │   │   └── arch_util.py       │     通用层工具
│   │   │   ├── losses/                │   损失函数
│   │   │   │   ├── losses.py          │     各损失定义
│   │   │   │   └── loss_util.py       │     损失工具
│   │   │   ├── base_model.py          │   基类模型
│   │   │   ├── image_restoration_model│   图像复原模型
│   │   │   └── lr_scheduler.py        │   学习率调度
│   │   │
│   │   ├── metrics/                   │  评价指标
│   │   │   ├── __init__.py            │
│   │   │   ├── psnr_ssim.py           │   PSNR / SSIM
│   │   │   ├── niqe.py                │   NIQE（无参考）
│   │   │   ├── niqe_pris_params.npz   │   NIQE 参数
│   │   │   ├── fid.py                 │   FID
│   │   │   └── metric_util.py         │   指标工具
│   │   │
│   │   └── utils/                     │  工具函数
│   │       ├── __init__.py            │
│   │       ├── img_util.py            │   图像处理
│   │       ├── options.py             │   配置解析
│   │       ├── logger.py              │   日志记录
│   │       ├── dist_util.py           │   分布式训练
│   │       ├── misc.py                │   杂项
│   │       ├── file_client.py         │   文件 I/O
│   │       ├── download_util.py       │   下载工具
│   │       ├── create_lmdb.py         │   LMDB 创建
│   │       ├── lmdb_util.py           │   LMDB 工具
│   │       ├── face_util.py           │   人脸处理
│   │       ├── flow_util.py           │   光流工具
│   │       ├── matlab_functions.py    │   MATLAB 兼容
│   │       └── ...                    │   其他
│   │
│   ├── options/                       │  配置文件
│   │   ├── train/                     │  训练 YAML
│   │   │   ├── GoPro/                 │   去模糊
│   │   │   ├── SIDD/                  │   去噪
│   │   │   ├── REDS/                  │   视频
│   │   │   └── NAFSSR/               │   立体超分
│   │   └── test/                      │  测试 YAML
│   │       ├── GoPro/                 │
│   │       ├── SIDD/                  │
│   │       ├── REDS/                  │
│   │       └── NAFSSR/               │
│   │
│   ├── datasets/                      │  数据集存放目录
│   ├── experiments/                   │  训练输出
│   │   └── pretrained_models/         │  预训练权重
│   │       └── NAFNet-SIDD-width64.pth│  SIDD 去噪权重
│   │
│   ├── demo/                          │  示例图片
│   │   ├── noisy.png / denoise_img.png│
│   │   ├── blurry.jpg                 │
│   │   ├── lr_img_l.png / lr_img_r.png│
│   │   ├── sr_img_l.png / sr_img_r.png│
│   │   └── ...                        │
│   │
│   ├── figures/                       │  论文配图
│   │   ├── deblur.gif                 │   去模糊效果
│   │   ├── denoise.gif                │   去噪效果
│   │   ├── StereoSR.gif               │   立体超分效果
│   │   ├── NAFSSR_arch.jpg            │   网络结构图
│   │   ├── NAFSSR_params.jpg          │   参数对比
│   │   └── PSNR_vs_MACs.jpg          │   PSNR-计算量图
│   │
│   ├── docs/                          │  文档
│   │   ├── GoPro.md                   │   GoPro 说明
│   │   ├── REDS.md                    │   REDS 说明
│   │   ├── SIDD.md                    │   SIDD 说明
│   │   └── StereoSR.md                │   立体超分说明
│   │
│   └── scripts/                       │  数据预处理
│       ├── make_pickle.py             │   生成 pickle
│       └── data_preparation/          │
│           ├── gopro.py               │   GoPro 预处理
│           ├── reds.py                │   REDS 预处理
│           └── sidd.py                │   SIDD 预处理
│
├── Real-ESRGAN/                       # ════════════════════
│   │                                  # 超分辨率模型
│   ├── README.md / README_CN.md       #  中英文文档
│   ├── requirements.txt               #  依赖
│   ├── setup.py / setup.cfg           #  安装配置
│   ├── VERSION / LICENSE              #  版本 / 许可
│   ├── MANIFEST.in                    #  pip 打包清单
│   ├── cog.yaml / cog_predict.py      #  Replicate 部署
│   ├── output.png                     #  测试输出
│   │
│   ├── realesrgan/                    #  ── 核心代码 ──
│   │   ├── __init__.py                │
│   │   ├── train.py                   │  训练入口
│   │   ├── utils.py                   │  训练工具
│   │   ├── version.py                 │  版本信息
│   │   ├── archs/                     │  网络架构
│   │   │   ├── __init__.py            │
│   │   │   ├── discriminator_arch.py  │   判别器架构
│   │   │   └── srvgg_arch.py          │   SRVGG 架构
│   │   ├── data/                      │  数据集
│   │   │   ├── __init__.py            │
│   │   │   ├── realesrgan_dataset.py  │   GAN 训练数据
│   │   │   └── realesrgan_paired_data │   成对数据
│   │   └── models/                    │  模型
│   │       ├── __init__.py            │
│   │       ├── realesrgan_model.py    │   ESRGAN 模型
│   │       └── realesrnet_model.py    │   RRDBNet 模型
│   │
│   ├── inference_realesrgan.py        │  单图推理
│   ├── inference_realesrgan_video.py  │  视频推理
│   │
│   ├── options/                       │  训练配置
│   │   ├── train_realesrgan_x2plus.yml│   2× GAN
│   │   ├── train_realesrgan_x4plus.yml│   4× GAN
│   │   ├── train_realesrnet_x2plus.yml│   2× RRDBNet
│   │   ├── train_realesrnet_x4plus.yml│   4× RRDBNet
│   │   ├── finetune_realesrgan_x4plus │   微调（通用）
│   │   └── finetune_*_pairdata.yml    │   微调（成对）
│   │
│   ├── experiments/pretrained_models/ │  预训练权重目录
│   ├── weights/                       │  推理可用权重
│   │   ├── RealESRGAN_x4plus.pth      │   ×4 标准超分
│   │   ├── realesr-general-x4v3.pth   │   ×4 通用 v3
│   │   ├── realesr-general-wdn-x4v3   │   ×4 轻量 v3
│   │   └── README.md                  │   权重说明
│   ├── inputs/                        │  输入测试图
│   ├── results/                       │  输出结果图
│   ├── gfpgan/weights/                │  人脸增强权重
│   ├── tests/                         │  单元测试
│   ├── scripts/                       │  预处理脚本
│   └── docs/                          │  文档
│       ├── anime_model.md             │   动漫模型
│       ├── Training.md / Training_CN  │   训练指南
│       ├── FAQ.md / feedback.md       │   常见问题
│       ├── model_zoo.md               │   模型库
│       └── ncnn_conversion.md         │   ncnn 转换
│
├── report/                            # ════════════════════
│   ├── 报告.docx                      #  Word 版实验报告
│   └── report.html                    #  HTML 版实验报告
│
└── results/                           # ════════════════════
    ├── comparison_table.png.png       #  模型对比表截图
    ├── confusion_matrix.png.png       #  混淆矩阵
    └── loss_curve.png.png             #  损失曲线
```

---

## 环境配置

### 系统要求

- **Python** >= 3.7
- **PyTorch** >= 1.7（推荐 1.11.0 + CUDA 11.3）
- **CUDA** 11.3+（如有 GPU）

### 安装步骤

```bash
# 1. 安装公共依赖
pip install -r core/requirements.txt

# 2. 手动安装 PyTorch（根据你的 CUDA 版本选择）
#    推荐：
#    pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113
#    官方命令查询：https://pytorch.org/get-started/previous-versions/

# 3. （可选）分别安装两个子项目
cd NAFNet && pip install -e .
cd ../Real-ESRGAN && pip install -e .
```

### 依赖清单

| 依赖 | 用途 |
|------|------|
| `torch` / `torchvision` | 深度学习框架 |
| `numpy` | 数值计算 |
| `opencv-python` | 图像 I/O 与处理 |
| `Pillow` | 图像格式支持 |
| `tqdm` | 进度条显示 |
| `addict` | 字典操作 |
| `lmdb` | 数据库存储 |
| `pyyaml` | 解析配置文件 |
| `scikit-image` | 图像评价指标 |
| `scipy` | 科学计算 |
| `tensorboard` | 训练可视化 |
| `basicsr>=1.4.2` | 超分基础框架 |
| `facexlib>=0.2.5` | 人脸检测工具 |
| `gfpgan>=1.3.5` | 人脸修复 |

---

## 使用指南

### 串联训练

一键启动 NAFNet → Real-ESRGAN 两阶段训练：

```bash
python core/train.py
```

执行流程：

1. 自动读取配置文件，加载 SIDD / GoPro 等数据集
2. 训练 NAFNet 去噪模型，保存中间权重
3. 以 NAFNet 输出为基础，启动 Real-ESRGAN 超分训练
4. 训练日志与权重保存至 `experiments/` 目录

### 串联推理

输入一张噪声 / 模糊图片，依次经过去噪和超分，输出高清结果：

```bash
python core/eval.py --input NAFNet/demo/noisy.png --output output_hd.png
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--input` | 输入图片路径 | 必填 |
| `--output` | 输出图片路径 | `output_hd.png` |

### 分步运行

如需单独运行某个模块，也可直接进入子项目目录：

```bash
# 仅 NAFNet 推理
cd NAFNet && python predict.py ...

# 仅 Real-ESRGAN 推理
cd Real-ESRGAN && python inference_realesrgan.py -i inputs/ --output results/
```

---

## 数据集

本项目训练与评估使用以下公开数据集：

| 数据集 | 任务 | 说明 |
|--------|------|------|
| **SIDD** | 去噪 | 智能手机拍摄的真实噪声/干净图像对 |
| **GoPro** | 去模糊 | 运动模糊/清晰图像对，用于去模糊训练 |
| **REDS** | 视频复原 | 视频帧的去噪 / 超分，含多帧信息 |
| **Vimeo-90K** | 视频去噪 | 90K 视频序列，含 triplet 结构 |
| **DIV2K** / **DF2K** | 超分 | 高清自然图像，用于 Real-ESRGAN 训练 |
| **FFHQ** | 人脸增强 | 高清人脸数据集 |

数据预处理脚本详见：
- `NAFNet/scripts/data_preparation/`
- `Real-ESRGAN/scripts/`

---

## 评价指标

| 指标 | 类型 | 说明 |
|------|------|------|
| **PSNR** | 全参考 | 像素级峰值信噪比，数值越高越好 |
| **SSIM** | 全参考 | 结构相似性，衡量感知质量 |
| **NIQE** | 无参考 | 基于自然场景统计的无参考质量评价 |
| **FID** | 无参考 | Fréchet 距离，评估生成图像分布 |

PSNR 和 SSIM 通过 `psnr_ssim.py` 计算，NIQE 基于预训练的 `niqe_pris_params.npz` 参数。

---

## 实验结果

实验结果图表保存在 `results/` 目录：

- `loss_curve.png.png` — 训练过程中损失值变化曲线
- `comparison_table.png.png` — 各模型/方法效果对比表格
- `confusion_matrix.png.png` — 分类结果混淆矩阵

完整的实验分析与量化指标对比参见 `report/` 目录中的实验报告。

---

## 团队分工

| 成员 | 职责 |
|------|------|
| **靳羽晨** | 系统核心代码编写、仿真开发环境搭建调试、网络模型训练与优化 |
| **肖飞** | 前沿算法文献调研、技术方案设计、核心模型架构选型与确定 |
| **谭志鹏** | 实验结果测试评估、量化指标统计分析、实验报告撰写与统稿 |
| **饶泽宇** | 数据收集、筛选清洗、退化模拟等基础预处理 |
| **唐申** | 前期技术路线调研、方案可行性论证、核心算法选型 |

---

## 参考文献

1. Chen, L., et al. "Simple Baselines for Image Restoration." *ECCV 2022*. (NAFNet)
2. Wang, X., et al. "Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data." *ICCV 2021*. (Real-ESRGAN)
3. Liang, J., et al. "SwinIR: Image Restoration Using Swin Transformer." *ICCV 2021*.
4. Zamir, S. W., et al. "Restormer: Efficient Transformer for High-Resolution Image Restoration." *CVPR 2022*.
5. Zhang, K., et al. "Deep Plug-and-Play Super-Resolution for Arbitrary Blur Kernels." *CVPR 2019*.
