"""
图像复原数据处理工具

整合自 D:\\edge\\projects 两个项目的数据处理代码:
  - NAFNet (v1.2.0): NAFNet/basicsr/data/
  - Real-ESRGAN (v0.3.0)
"""

import cv2
import math
import random
import numpy as np
import torch
import os
from typing import List, Optional, Tuple, Union
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision.utils import make_grid
from torchvision.transforms.functional import normalize


# ============================================================================
# 图像 I/O 工具
# ============================================================================

def imread(path: str, flag: str = 'color') -> np.ndarray:
    """读取图像 (BGR -> RGB).

    Args:
        path: 图像路径.
        flag: 'color' | 'grayscale' | 'unchanged'.

    Returns:
        RGB 图像, numpy.ndarray, dtype=uint8, range [0, 255].
    """
    flags = {
        'color': cv2.IMREAD_COLOR,
        'grayscale': cv2.IMREAD_GRAYSCALE,
        'unchanged': cv2.IMREAD_UNCHANGED,
    }
    img = cv2.imread(path, flags[flag])
    if img is None:
        raise FileNotFoundError(f'无法读取图像: {path}')
    if flag == 'color' and img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def imfrombytes(content: bytes, flag: str = 'color',
                float32: bool = False) -> np.ndarray:
    """从字节数据读取图像.

    Args:
        content: 图像字节流.
        flag: 'color' | 'grayscale' | 'unchanged'.
        float32: 是否转为 float32 并归一化到 [0, 1].

    Returns:
        解码后的图像数组.
    """
    img_np = np.frombuffer(content, np.uint8)
    imread_flags = {
        'color': cv2.IMREAD_COLOR,
        'grayscale': cv2.IMREAD_GRAYSCALE,
        'unchanged': cv2.IMREAD_UNCHANGED,
    }
    if img_np is None:
        raise ValueError('无法解码图像字节流')
    img = cv2.imdecode(img_np, imread_flags[flag])
    if float32:
        img = img.astype(np.float32) / 255.0
    return img


def imwrite(img: np.ndarray, file_path: str, params=None,
            auto_mkdir: bool = True) -> bool:
    """保存图像到文件.

    Args:
        img: 要保存的图像数组.
        file_path: 保存路径.
        params: cv2.imwrite 参数.
        auto_mkdir: 是否自动创建父目录.

    Returns:
        是否写入成功.
    """
    if auto_mkdir:
        dir_name = os.path.abspath(os.path.dirname(file_path))
        os.makedirs(dir_name, exist_ok=True)
    return cv2.imwrite(file_path, img, params)


# ============================================================================
# Numpy ↔ Tensor 转换
# ============================================================================

def img2tensor(imgs: Union[np.ndarray, List[np.ndarray]],
               bgr2rgb: bool = True,
               float32: bool = True) -> Union[torch.Tensor, List[torch.Tensor]]:
    """numpy 图像转 torch.Tensor.

    转换: HWC → CHW, BGR → RGB (可选), uint8 → float32 (可选).

    Args:
        imgs: 单张或列表 numpy 图像.
        bgr2rgb: 是否做 BGR → RGB 转换.
        float32: 是否转为 float32 (同时归一化到 [0, 1]).

    Returns:
        Tensor 或 Tensor 列表, shape (C, H, W), range [0, 1].
    """
    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float() / 255.0
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)


def tensor2img(tensor: Union[torch.Tensor, List[torch.Tensor]],
               rgb2bgr: bool = True,
               out_type: type = np.uint8,
               min_max: Tuple[float, float] = (0, 1)) -> np.ndarray:
    """torch.Tensor 转 numpy 图像.

    转换: CHW → HWC, RGB → BGR (可选), 范围 [0,1] → [0,255].

    Args:
        tensor: 4D (B,C,H,W) 或 3D (C,H,W) Tensor.
        rgb2bgr: 是否做 RGB → BGR 转换.
        out_type: 输出类型, np.uint8 或 np.float32.
        min_max: 裁剪范围.

    Returns:
        numpy 图像数组.
    """
    if torch.is_tensor(tensor):
        tensor = [tensor]

    result = []
    for _tensor in tensor:
        _tensor = _tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])

        n_dim = _tensor.dim()
        if n_dim == 4:
            img_np = make_grid(_tensor, nrow=int(math.sqrt(_tensor.size(0))),
                               normalize=False).numpy()
            img_np = img_np.transpose(1, 2, 0)
            if rgb2bgr:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 3:
            img_np = _tensor.numpy()
            img_np = img_np.transpose(1, 2, 0)
            if img_np.shape[2] == 1:
                img_np = np.squeeze(img_np, axis=2)
            elif img_np.shape[2] == 3 and rgb2bgr:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 2:
            img_np = _tensor.numpy()
        else:
            raise ValueError(f'不支持的 Tensor 维度: {n_dim}')

        if out_type == np.uint8:
            img_np = (img_np * 255.0).round().astype(np.uint8)
        else:
            img_np = img_np.astype(out_type)
        result.append(img_np)

    return result[0] if len(result) == 1 else result


# ============================================================================
# 图像预处理
# ============================================================================

def mod_crop(img: np.ndarray, scale: int) -> np.ndarray:
    """将图像裁剪到 scale 的整数倍.

    用于测试时确保输入/输出尺寸匹配.

    Args:
        img: 输入图像 (H, W) 或 (H, W, C).
        scale: 缩放因子.

    Returns:
        裁剪后的图像.
    """
    img = img.copy()
    if img.ndim in (2, 3):
        h, w = img.shape[:2]
        img = img[:h - h % scale, :w - w % scale, ...]
    return img


def padding(img_lq: np.ndarray, img_gt: np.ndarray,
            gt_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """补零对齐, 确保训练时 patch 提取不会越界.

    使用 BORDER_REFLECT 填充.

    Args:
        img_lq: 低质图像.
        img_gt: 高质图像.
        gt_size: 目标 GT patch 尺寸.

    Returns:
        (填充后的 LQ 图像, 填充后的 GT 图像).
    """
    h, w = img_lq.shape[:2]
    h_pad = max(0, gt_size - h)
    w_pad = max(0, gt_size - w)
    if h_pad == 0 and w_pad == 0:
        return img_lq, img_gt
    img_lq = cv2.copyMakeBorder(img_lq, 0, h_pad, 0, w_pad,
                                cv2.BORDER_REFLECT)
    img_gt = cv2.copyMakeBorder(img_gt, 0, h_pad, 0, w_pad,
                                cv2.BORDER_REFLECT)
    return img_lq, img_gt


def crop_border(imgs: Union[np.ndarray, List[np.ndarray]],
                crop_border: int) -> Union[np.ndarray, List[np.ndarray]]:
    """裁剪图像边缘 (去除 padding 影响)."""
    if crop_border == 0:
        return imgs
    if isinstance(imgs, list):
        return [v[crop_border:-crop_border, crop_border:-crop_border, ...]
                for v in imgs]
    else:
        return imgs[crop_border:-crop_border, crop_border:-crop_border, ...]


def paired_random_crop(img_gts, img_lqs, gt_patch_size, scale, gt_path=None):
    """图像对随机裁剪.

    在相同位置裁剪 GT 和 LQ 图像对.

    Args:
        img_gts: GT 图像 (list 或 ndarray).
        img_lqs: LQ 图像 (list 或 ndarray).
        gt_patch_size: GT patch 大小.
        scale: 缩放倍率.
        gt_path: GT 路径 (用于错误提示).

    Returns:
        裁剪后的 (GT, LQ) 图像对.
    """
    if not isinstance(img_gts, list):
        img_gts = [img_gts]
    if not isinstance(img_lqs, list):
        img_lqs = [img_lqs]

    h_lq, w_lq = img_lqs[0].shape[:2]
    h_gt, w_gt = img_gts[0].shape[:2]
    lq_patch_size = gt_patch_size // scale

    if h_gt != h_lq * scale or w_gt != w_lq * scale:
        raise ValueError(f'尺寸不匹配: GT ({h_gt},{w_gt}) 不是 LQ ({h_lq},{w_lq}) 的 {scale}倍.')
    if h_lq < lq_patch_size or w_lq < lq_patch_size:
        raise ValueError(f'LQ 图像 ({h_lq},{w_lq}) 小于 patch 尺寸 ({lq_patch_size},{lq_patch_size}).')

    top = random.randint(0, h_lq - lq_patch_size)
    left = random.randint(0, w_lq - lq_patch_size)

    img_lqs = [v[top:top + lq_patch_size, left:left + lq_patch_size, ...]
               for v in img_lqs]
    top_gt, left_gt = int(top * scale), int(left * scale)
    img_gts = [v[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...]
               for v in img_gts]

    if len(img_gts) == 1:
        img_gts = img_gts[0]
    if len(img_lqs) == 1:
        img_lqs = img_lqs[0]
    return img_gts, img_lqs


# ============================================================================
# 数据增强
# ============================================================================

def augment(imgs, hflip=True, rotation=True, flows=None, return_status=False,
            vflip=False):
    """数据增强: 水平翻转 / 垂直翻转 / 90° 旋转.

    所有图像使用相同的增强方式.

    Args:
        imgs: 待增强图像 (list 或 ndarray).
        hflip: 是否使用水平翻转.
        rotation: 是否使用 90° 旋转.
        flows: 光流 (可选, 同时增强).
        return_status: 是否返回增强状态.
        vflip: 是否使用垂直翻转.

    Returns:
        增强后的图像 (和光流).
    """
    hflip = hflip and random.random() < 0.5
    vflip = vflip or (rotation and random.random() < 0.5)
    rot90 = rotation and random.random() < 0.5

    def _augment(img):
        if hflip:
            cv2.flip(img, 1, img)  # 水平
            if img.ndim == 3 and img.shape[2] == 6:
                img = img[:, :, [3, 4, 5, 0, 1, 2]].copy()  # 交换左右视图
        if vflip:
            cv2.flip(img, 0, img)  # 垂直
        if rot90:
            img = img.transpose(1, 0, 2)  # 旋转90°
        return img

    def _augment_flow(flow):
        if hflip:
            cv2.flip(flow, 1, flow)
            flow[:, :, 0] *= -1
        if vflip:
            cv2.flip(flow, 0, flow)
            flow[:, :, 1] *= -1
        if rot90:
            flow = flow.transpose(1, 0, 2)
            flow = flow[:, :, [1, 0]]
        return flow

    if not isinstance(imgs, list):
        imgs = [imgs]
    imgs = [_augment(img) for img in imgs]
    if len(imgs) == 1:
        imgs = imgs[0]

    if flows is not None:
        if not isinstance(flows, list):
            flows = [flows]
        flows = [_augment_flow(flow) for flow in flows]
        flows = flows[0] if len(flows) == 1 else flows
        return imgs, flows
    else:
        return imgs if not return_status else (imgs, (hflip, vflip, rot90))


def img_rotate(img: np.ndarray, angle: float, center=None,
               scale: float = 1.0) -> np.ndarray:
    """任意角度旋转图像.

    Args:
        img: 输入图像.
        angle: 旋转角度 (度数, 正值为逆时针).
        center: 旋转中心 (默认图像中心).
        scale: 缩放因子.

    Returns:
        旋转后的图像.
    """
    h, w = img.shape[:2]
    if center is None:
        center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    return cv2.warpAffine(img, matrix, (w, h))


# ============================================================================
# 数据集路径生成
# ============================================================================

def scandir(path: str, full_path: bool = False) -> List[str]:
    """扫描目录, 返回文件列表.

    Args:
        path: 目录路径.
        full_path: 是否返回完整路径.

    Returns:
        文件名或完整路径列表.
    """
    files = sorted(os.listdir(path))
    if full_path:
        return [os.path.join(path, f) for f in files]
    return files


def paired_paths_from_folder(folders: List[str], keys: List[str],
                              filename_tmpl: str) -> List[dict]:
    """从文件夹生成配对路径.

    假设 input 和 GT 文件夹中的文件按相同顺序一一对应.

    Args:
        folders: [input_folder, gt_folder].
        keys: ['lq', 'gt'].
        filename_tmpl: 文件名模板 (如 '{}_x2').

    Returns:
        路径字典列表 [{'lq_path': ..., 'gt_path': ...}, ...].
    """
    input_folder, gt_folder = folders
    input_key, gt_key = keys

    input_paths = scandir(input_folder, full_path=False)
    gt_paths = scandir(gt_folder, full_path=False)

    assert len(input_paths) == len(gt_paths), \
        f'LQ 和 GT 数据集图像数量不一致: {len(input_paths)} vs {len(gt_paths)}'

    paths = []
    for gt_name in gt_paths:
        basename, ext = os.path.splitext(gt_name)
        # 根据模板构造 LQ 文件名
        input_name = filename_tmpl.format(basename) + ext
        paths.append({
            f'{input_key}_path': os.path.join(input_folder, input_name),
            f'{gt_key}_path': os.path.join(gt_folder, gt_name),
        })
    return paths


def paired_paths_from_meta_info_file(folders, keys, meta_info_file,
                                      filename_tmpl):
    """从元信息文件生成配对路径.

    元信息文件每行: image_name.png (H,W,C)
    """
    input_folder, gt_folder = folders
    input_key, gt_key = keys

    with open(meta_info_file, 'r') as fin:
        gt_names = [line.split(' ')[0] for line in fin]

    paths = []
    for gt_name in gt_names:
        basename, ext = os.path.splitext(os.path.basename(gt_name))
        input_name = filename_tmpl.format(basename) + ext
        paths.append({
            f'{input_key}_path': os.path.join(input_folder, input_name),
            f'{gt_key}_path': os.path.join(gt_folder, gt_name),
        })
    return paths


def generate_frame_indices(crt_idx, max_frame_num, num_frames,
                           padding='reflection'):
    """生成视频帧索引 (用于视频复原).

    支持多种 padding 模式:
        - 'replicate': 用首/尾帧复制
        - 'reflection': 镜像反射
        - 'reflection_circle': 循环反射
        - 'circle': 循环

    Args:
        crt_idx: 当前中心帧索引.
        max_frame_num: 总帧数.
        num_frames: 要读取的帧数 (需奇数).
        padding: 边界填充模式.

    Returns:
        帧索引列表.
    """
    assert num_frames % 2 == 1, 'num_frames 必须为奇数'
    assert padding in ('replicate', 'reflection', 'reflection_circle', 'circle')

    max_frame_num = max_frame_num - 1  # from 0
    num_pad = num_frames // 2

    indices = []
    for i in range(crt_idx - num_pad, crt_idx + num_pad + 1):
        if i < 0:
            if padding == 'replicate':
                pad_idx = 0
            elif padding == 'reflection':
                pad_idx = -i
            elif padding == 'reflection_circle':
                pad_idx = crt_idx + num_pad - i
            else:
                pad_idx = num_frames + i
        elif i > max_frame_num:
            if padding == 'replicate':
                pad_idx = max_frame_num
            elif padding == 'reflection':
                pad_idx = max_frame_num * 2 - i
            elif padding == 'reflection_circle':
                pad_idx = (crt_idx - num_pad) - (i - max_frame_num)
            else:
                pad_idx = i - num_frames
        else:
            pad_idx = i
        indices.append(pad_idx)
    return indices


# ============================================================================
# 高斯核 & 下采样
# ============================================================================

def generate_gaussian_kernel(kernel_size=13, sigma=1.6):
    """生成高斯核.

    Args:
        kernel_size: 核大小.
        sigma: 高斯标准差.

    Returns:
        numpy 高斯核.
    """
    from scipy.ndimage import filters
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[kernel_size // 2, kernel_size // 2] = 1
    return filters.gaussian_filter(kernel, sigma)


def duf_downsample(x, kernel_size=13, scale=4):
    """DUF 风格高斯下采样.

    Args:
        x: 输入帧, shape (b, t, c, h, w) 或 (c, h, w).
        kernel_size: 高斯核大小.
        scale: 下采样倍率.

    Returns:
        下采样后的帧.
    """
    assert scale in (2, 3, 4), f'支持 scale 2,3,4, 收到 {scale}'

    squeeze_flag = False
    if x.ndim == 4:  # (c, h, w) → (1, 1, c, h, w)
        squeeze_flag = True
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.ndim == 3:
        x = x.unsqueeze(0).unsqueeze(0)

    b, t, c, h, w = x.size()
    x = x.view(-1, 1, h, w)
    pad_w, pad_h = kernel_size // 2 + scale * 2, kernel_size // 2 + scale * 2
    x = F_pad(x, (pad_w, pad_w, pad_h, pad_h), 'reflect')

    gaussian_filter = generate_gaussian_kernel(kernel_size, 0.4 * scale)
    gaussian_filter = torch.from_numpy(gaussian_filter).float().to(x.device)
    gaussian_filter = gaussian_filter.unsqueeze(0).unsqueeze(0)

    x = torch.nn.functional.conv2d(x, gaussian_filter, stride=scale)
    x = x[:, :, 2:-2, 2:-2]
    x = x.view(b, t, c, x.size(2), x.size(3))
    if squeeze_flag:
        x = x.squeeze(0).squeeze(0)
    return x


def F_pad(x, padding, mode='reflect'):
    """F.pad 简化封装."""
    return torch.nn.functional.pad(x, padding, mode=mode)


# ============================================================================
# Sampler (分布式训练采样器)
# ============================================================================

class EnlargedSampler(Sampler):
    """增强型采样器, 支持基于 iteration 的训练.

    与 DistributedSampler 类似, 但支持 dataset 放大 (ratio > 1),
    避免每个 epoch 重启 dataloader 的开销.

    Args:
        dataset: 数据集.
        num_replicas: 分布式进程数 (world_size).
        rank: 当前进程 rank.
        ratio: dataset 放大倍数. 默认 1.
    """

    def __init__(self, dataset, num_replicas, rank, ratio=1):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = math.ceil(len(self.dataset) * ratio / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = torch.randperm(self.total_size, generator=g).tolist()
        dataset_size = len(self.dataset)
        indices = [v % dataset_size for v in indices]
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


# ============================================================================
# 配对图像数据集
# ============================================================================

class PairedImageDataset(Dataset):
    """配对图像数据集 (LQ/GT 对).

    支持三种数据源:
        1. LMDB: 使用 LMDB 数据库
        2. meta_info_file: 使用元信息文件
        3. folder: 直接扫描文件夹

    Args:
        opt (dict): 配置字典, 包含:
            - dataroot_gt, dataroot_lq: GT 和 LQ 数据根目录
            - io_backend: IO 后端配置
            - meta_info_file: 元信息文件路径 (可选)
            - filename_tmpl: 文件名模板 (默认 '{}')
            - gt_size: 训练时 GT patch 大小
            - scale: 缩放倍率
            - phase: 'train' 或 'val'
            - use_flip, use_rot: 数据增强选项
            - mean, std: 归一化参数 (可选)
    """

    def __init__(self, opt: dict):
        super().__init__()
        self.opt = opt
        self.io_backend_opt = opt.get('io_backend', {'type': 'disk'})
        self.mean = opt.get('mean')
        self.std = opt.get('std')
        self.gt_folder = opt['dataroot_gt']
        self.lq_folder = opt['dataroot_lq']
        self.filename_tmpl = opt.get('filename_tmpl', '{}')

        # 根据后端类型生成路径
        io_type = self.io_backend_opt.get('type', 'disk')
        if io_type == 'lmdb':
            self.paths = self._paths_from_lmdb()
        elif 'meta_info_file' in opt and opt.get('meta_info_file'):
            self.paths = paired_paths_from_meta_info_file(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = paired_paths_from_folder(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.filename_tmpl)

    def _paths_from_lmdb(self):
        """从 LMDB 生成路径 (暂略, 需要 lmdb 库)."""
        raise NotImplementedError('LMDB 支持需要安装 lmdb 包')

    def __getitem__(self, index):
        scale = self.opt.get('scale', 1)

        gt_path = self.paths[index]['gt_path']
        lq_path = self.paths[index]['lq_path']

        # 直接读取图像 (简化版, 不使用 FileClient)
        img_gt = imread(gt_path)
        img_lq = imread(lq_path)

        # 训练时增强
        if self.opt.get('phase') == 'train':
            gt_size = self.opt.get('gt_size', 256)
            img_gt, img_lq = padding(img_gt, img_lq, gt_size)
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size,
                                                scale, gt_path)
            img_gt, img_lq = augment([img_gt, img_lq],
                                     self.opt.get('use_flip', True),
                                     self.opt.get('use_rot', True))

        # BGR → RGB, HWC → CHW, uint8 → float32 [0,1]
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)

        # 归一化
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paths)
