# T14 2A — 图像复原串联系统

> **两阶段图像复原：NAFNet 降噪 / 去模糊 → Real-ESRGAN 超分辨率**

---

## 项目概述

本项目实现了一个端到端的图像复原流水线，将两个 SOTA 模型串联：

1. **NAFNet** — 非线性激活自由网络，负责图像去噪 / 去模糊
2. **Real-ESRGAN** — 基于 ESRGAN 的真实世界超分辨率模型

通过 `core/` 统一入口实现一键串联训练与推理。

---

## 目录结构

```
D:\Download\文件\T14 2A\
├── README.md                     # 本文件
├── contribution.txt              # 团队成员分工
│
├── core/                         # ⭐ 统一入口
│   ├── README.md                 #   core 使用说明
│   ├── requirements.txt          #   公共依赖清单
│   ├── train.py                  #   串联训练入口
│   ├── eval.py                   #   串联推理（降噪→超分）
│   ├── model.py                  #   模型封装
│   └── data_utils.py             #   数据工具函数
│
├── NAFNet/                       # 🔧 去噪 / 去模糊模型
│   ├── readme.md                 #   官方说明
│   ├── requirements.txt          #   依赖
│   ├── setup.py / setup.cfg      #   安装配置
│   ├── VERSION                   #   版本号
│   ├── LICENSE                   #   许可证
│   ├── cog.yaml                  #   Replicate 配置
│   ├── predict.py                #   推理脚本
│   ├── output.png / test.png     #   示例输出
│   │
│   ├── basicsr/                  #   ⚙️ 核心代码
│   │   ├── train.py              │   训练入口
│   │   ├── test.py               │   测试入口
│   │   ├── demo.py / demo_ssr.py │   示例 / 立体超分
│   │   ├── version.py            │   版本信息
│   │   │
│   │   ├── data/                 │   数据集
│   │   │   ├── data_util.py      │   数据工具
│   │   │   ├── data_sampler.py   │   采样器
│   │   │   ├── transforms.py     │   数据增强
│   │   │   ├── ffhq_dataset.py   │   FFHQ 人脸
│   │   │   ├── reds_dataset.py   │   REDS 视频
│   │   │   ├── vimeo90k_dataset.py│  Vimeo-90K
│   │   │   ├── paired_image_*.py │   成对图像
│   │   │   ├── single_image_*    │   单张图像
│   │   │   ├── video_test_*      │   视频测试
│   │   │   ├── prefetch_*        │   预取加载器
│   │   │   └── meta_info/        │   数据集元信息
│   │   │
│   │   ├── models/               │   模型定义
│   │   │   ├── archs/            │   网络架构
│   │   │   │   ├── NAFNet_arch.py│     NAFNet 核心
│   │   │   │   ├── NAFSSR_arch.py│     立体超分
│   │   │   │   ├── Baseline_arch │     基线
│   │   │   │   └── local_arch.py │     局部网络
│   │   │   ├── losses/           │   损失函数
│   │   │   ├── base_model.py     │   模型基类
│   │   │   ├── image_restoration │   图像复原模型
│   │   │   └── lr_scheduler.py   │   学习率调度
│   │   │
│   │   ├── metrics/              │   评价指标
│   │   │   ├── psnr_ssim.py      │   PSNR / SSIM
│   │   │   ├── niqe.py           │   NIQE
│   │   │   ├── fid.py            │   FID
│   │   │   └── metric_util.py    │   指标工具
│   │   │
│   │   └── utils/                │   工具函数
│   │       ├── img_util.py       │   图像处理
│   │       ├── options.py        │   配置解析
│   │       ├── logger.py         │   日志
│   │       ├── dist_util.py      │   分布式
│   │       └── ...               │   其他
│   │
│   ├── options/                  │   训练/测试配置
│   │   ├── train/                │   训练 YAML
│   │   │   ├── GoPro/            │   去模糊
│   │   │   ├── SIDD/             │   去噪
│   │   │   ├── REDS/             │   视频复原
│   │   │   └── NAFSSR/           │   立体超分
│   │   └── test/                 │   测试 YAML
│   │
│   ├── datasets/                 │   数据集存放
│   ├── experiments/              │   实验输出
│   │   └── pretrained_models/    │   预训练权重
│   │       └── NAFNet-SIDD-width64.pth
│   │
│   ├── demo/                     │   示例图片
│   ├── figures/                  │   论文配图 / GIF
│   ├── docs/                     │   数据集说明
│   └── scripts/                  │   数据预处理
│       └── data_preparation/     │   GoPro / REDS / SIDD
│
├── Real-ESRGAN/                  # 🚀 超分辨率模型
│   ├── README.md / README_CN.md  #   中英文说明
│   ├── requirements.txt          #   依赖
│   ├── setup.py / setup.cfg      #   安装配置
│   ├── VERSION                   #   版本号
│   ├── LICENSE                   #   许可证
│   ├── MANIFEST.in               #   打包清单
│   ├── cog.yaml / cog_predict.py #   Replicate 部署
│   ├── output.png                #   示例输出
│   │
│   ├── realesrgan/               #   ⚙️ 核心代码
│   │   ├── train.py              │   训练入口
│   │   ├── utils.py              │   工具函数
│   │   ├── version.py            │   版本信息
│   │   ├── archs/                │   网络架构
│   │   │   ├── discriminator_arch│     判别器
│   │   │   └── srvgg_arch.py     │     SRVGG
│   │   ├── data/                 │   数据集
│   │   │   ├── realesrgan_*      │     Real-ESRGAN 数据
│   │   └── models/               │   模型
│   │       ├── realesrgan_model  │     GAN 模型
│   │       └── realesrnet_model  │     RRDBNet 模型
│   │
│   ├── inference_realesrgan.py   │   单图推理脚本
│   ├── inference_realesrgan_video│   视频推理脚本
│   │
│   ├── options/                  │   训练配置 YAML
│   │   ├── train_realesrgan_*.yml│   GAN 训练
│   │   ├── train_realesrnet_*.yml│   RRDBNet 训练
│   │   └── finetune_*.yml        │   微调配置
│   │
│   ├── experiments/              │   实验输出
│   │   └── pretrained_models/    │   预训练权重
│   ├── weights/                  │   推理权重
│   │   ├── RealESRGAN_x4plus.pth │   4x 超分
│   │   ├── realesr-general-x4v3  │   通用 4x v3
│   │   └── realesr-general-wdn.. │   轻量 4x v3
│   ├── inputs/                   │   输入示例图片
│   ├── results/                  │   输出示例
│   ├── gfpgan/                   │   人脸增强
│   ├── tests/                    │   单元测试
│   ├── scripts/                  │   数据预处理脚本
│   └── docs/                     │   文档（动漫 / 训练 / FAQ）
│
├── report/                       # 📄 实验报告
│   ├── 报告.docx                 #   Word 版报告
│   └── report.html               #   HTML 版报告
│
└── results/                      # 📊 实验结果图表
    ├── comparison_table.png.png  #   对比表格
    ├── confusion_matrix.png.png  #   混淆矩阵
    └── loss_curve.png.png        #   损失曲线
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r core/requirements.txt
```

> ⚠️ PyTorch 需根据 CUDA 版本手动安装。推荐 `torch 1.11.0 + CUDA 11.3`。

### 2. 串联训练

一键运行 NAFNet（SIDD 去噪）→ Real-ESRGAN（超分）：

```bash
python core/train.py
```

### 3. 串联推理

输入噪声/低清图片，输出高清结果：

```bash
python core/eval.py --input NAFNet/demo/noisy.png --output output_hd.png
```

**参数：**
| 参数 | 说明 |
|------|------|
| `--input` | 输入图片路径 |
| `--output` | 输出高清图片路径 |

---

## 团队分工

| 成员 | 职责 |
|------|------|
| **靳羽晨** | 系统核心代码、仿真环境搭建、网络模型训练与优化 |
| **肖飞** | 前沿算法文献调研、技术方案设计、核心模型架构选型 |
| **谭志鹏** | 实验结果测试评估、量化指标统计分析、实验报告撰写统稿 |
| **饶泽宇** | 数据收集、筛选清洗、退化模拟等预处理 |
| **唐申** | 前期技术路线调研、方案可行性论证、核心算法选型 |

---

## 数据集

本项目支持以下数据集（详见 `NAFNet/docs/`）：

- **SIDD** — 智能手机图像去噪
- **GoPro** — 运动去模糊
- **REDS** — 视频复原 / 超分
- **Vimeo-90K** — 视频帧插值 / 去噪

---

## 评价指标

- **PSNR / SSIM** — 保真度指标
- **NIQE** — 无参考图像质量评估
- **FID** — 生成图像分布距离

---

## 技术支持

- 镜像 / 复现：[Replicate](https://replicate.com) (`cog.yaml`)
- 人脸增强：[GFPGAN](https://github.com/TencentARC/GFPGAN)（集成在 Real-ESRGAN/weights 中）
