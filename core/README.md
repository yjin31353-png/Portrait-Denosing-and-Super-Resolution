# NAFNet + Real-ESRGAN Workspace

项目地址：`D:\edge\projects`

## 安装

```bash
cd D:\edge\projects
pip install -r core\requirements.txt
# 手动安装 PyTorch（推荐 torch 1.11.0 + CUDA 11.3）
```

## 训练

一键串联训练 NAFNet（去噪）→ Real-ESRGAN（超分）：

```bash
python core\train.py
```

自动执行 NAFNet SIDD 去噪训练，完成后接 Real-ESRGAN 超分训练。

## 评估（推理）

串联推理：先降噪，再超分高清：

```bash
python core\eval.py --input NAFNet\demo\noisy.png --output output_hd.png
```

参数说明：
- `--input`：输入噪声/低清图片路径
- `--output`：最终高清输出路径

## 结构

```
D:\edge\projects\
├── core\                  # 统一入口
│   ├── train.py           # 训练入口（调用两个项目）
│   ├── eval.py            # 评估入口（降噪→超分串联）
│   └── requirements.txt   # 唯一依赖
├── NAFNet\                # 去噪/去模糊模型
└── Real-ESRGAN\           # 超分辨率模型
```
