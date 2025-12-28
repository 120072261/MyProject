# =========================================================
# Kaggle Training Script (Resume-safe) - AMP compatible
# Scientific Image Forgery Detection
# Model: SwinV2 + SRM Artifact Branch + Gated Fusion + FPN
# Multi-task: Segmentation + Image-level Classification
# Checkpoint: saved every epoch, resume supported
# =========================================================

import os, glob, random, time
import numpy as np
import pandas as pd
import cv2

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2

import timm

# ----------------------------
# Reduce OpenCV deadlocks / CPU spikes
# ----------------------------
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


# ----------------------------
# AMP compatibility layer
# ----------------------------
def get_amp():
    """
    Returns:
        autocast_ctx: a context manager factory: autocast_ctx(enabled=True)
        scaler: GradScaler instance
    Compatible with both:
      - torch.amp.autocast / torch.amp.GradScaler
      - torch.cuda.amp.autocast / torch.cuda.amp.GradScaler
    """
    # Prefer new torch.amp API if available
    try:
        from torch.amp import autocast as amp_autocast  # type: ignore
        from torch.amp import GradScaler as AmpGradScaler  # type: ignore

        # New autocast signature: autocast(device_type, enabled=...)
        def autocast_ctx(enabled=True):
            if torch.cuda.is_available():
                return amp_autocast("cuda", enabled=enabled)
            else:
                return amp_autocast("cpu", enabled=enabled)

        scaler = AmpGradScaler("cuda", enabled=torch.cuda.is_available())
        return autocast_ctx, scaler

    except Exception:
        # Fallback to old torch.cuda.amp API
        from torch.cuda.amp import autocast as cuda_autocast
        from torch.cuda.amp import GradScaler as CudaGradScaler

        def autocast_ctx(enabled=True):
            return cuda_autocast(enabled=enabled)

        scaler = CudaGradScaler(enabled=torch.cuda.is_available())
        return autocast_ctx, scaler


autocast_ctx, scaler = get_amp()


def pick_backbone():
    candidates = [
        # MiT / SegFormer-style names that appear in some timm versions
        "segformer_b4", "segformer_b3", "segformer_b2",
        "mit_b4", "mit_b3", "mit_b2",
        "mix_transformer_b4", "mix_transformer_b3", "mix_transformer_b2",
        # very stable fallback backbones (always exist in timm)
        "convnext_base", "convnext_small"
    ]
    available = set(timm.list_models())
    for name in candidates:
        if name in available:
            return name
    # ultimate fallback
    return "convnext_small"


# ----------------------------
# Config
# ----------------------------
class Config:
    SEED = 42

    TRAIN_IMG_DIR = "/kaggle/input/recodai-luc-scientific-image-forgery-detection/train_images"
    TRAIN_MASK_DIR = "/kaggle/input/recodai-luc-scientific-image-forgery-detection/train_masks"

    IMG_SIZE = 512
    BATCH_SIZE = 8
    EPOCHS = 20
    LR = 2e-4
    WEIGHT_DECAY = 1e-3

    NUM_WORKERS = 2  # if still stuck, set 0
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use timm's current name to avoid deprecation warning
    BACKBONE_NAME = pick_backbone()
    FPN_DIM = 256

    LAMBDA_CLS = 0.35
    POS_WEIGHT_CLS = 1.0
    POS_WEIGHT_SEG = 2.0

    USE_AMP = True
    GRAD_CLIP = 1.0
    EMA_DECAY = None  # None to disable

    CHECKPOINT_PATH = "/kaggle/input/dual-branch-forensics-segmentation-transformer/checkpoint.pth"
    BEST_PATH = "/kaggle/input/dual-branch-forensics-segmentation-transformer/best_model.pth"

    PIN_MEMORY = True
    PERSISTENT_WORKERS = True
    DL_TIMEOUT = 60


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(Config.SEED)
print("Device:", Config.DEVICE)
# ----------------------------
# Helpers: atomic checkpoint save
# ----------------------------
def atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)

def get_rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    }

def set_rng_state(state):
    try:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        if torch.cuda.is_available() and state.get("cuda") is not None:
            torch.cuda.set_rng_state_all(state["cuda"])
    except Exception as e:
        print("[WARN] Failed to restore RNG state:", e)


# ----------------------------
# Build dataframe
# ----------------------------
def get_all_files(directory, exts):
    files = []
    for ext in exts:
        files += glob.glob(os.path.join(directory, "**", ext), recursive=True)
    return sorted(list(set(files)))

image_files = get_all_files(Config.TRAIN_IMG_DIR, ["*.jpg", "*.png", "*.tif", "*.tiff", "*.jpeg"])
mask_files  = get_all_files(Config.TRAIN_MASK_DIR, ["*.png", "*.jpg", "*.npy"])

mask_map = {os.path.basename(p).split(".")[0]: p for p in mask_files}

data = []
for img_path in image_files:
    case_id = os.path.basename(img_path).split(".")[0]
    label = "authentic" if "authentic" in img_path.lower() else "forged"
    mask_path = mask_map.get(case_id, None)
    if label == "authentic":
        mask_path = None
    data.append({"case_id": case_id, "image_path": img_path, "mask_path": mask_path, "label": label})

df = pd.DataFrame(data)
print("Total:", len(df))
print(df["label"].value_counts())
print("Forged matched masks:", df[df["label"]=="forged"]["mask_path"].notnull().sum())


# ----------------------------
# Dataset
# ----------------------------
class SIFDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def _read_image(self, path):
        img = cv2.imread(path)
        if img is None:
            img = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _read_mask(self, mask_path, h, w):
        if mask_path is None:
            return np.zeros((h, w), np.float32)

        if mask_path.endswith(".npy"):
            try:
                m = np.load(mask_path)
            except Exception:
                m = np.zeros((h, w), np.float32)
        else:
            m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if m is None:
                m = np.zeros((h, w), np.float32)

        if m.ndim == 3:
            if m.shape[0] == h and m.shape[1] == w:
                m = m.max(axis=2)
            else:
                m = m.max(axis=0)

        if m.shape != (h, w):
            m = cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)

        return (m > 0).astype(np.float32)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = self._read_image(row["image_path"])
        h, w = img.shape[:2]
        mask = self._read_mask(row["mask_path"], h, w)
        cls = 1.0 if row["label"] == "forged" else 0.0

        if self.transform is not None:
            out = self.transform(image=img, mask=mask)
            img = out["image"]
            mask = out["mask"]

        if isinstance(mask, torch.Tensor) and mask.ndim == 3:
            mask = mask.max(dim=0).values

        mask = mask.unsqueeze(0).float()
        return img.float(), mask, torch.tensor([cls], dtype=torch.float32)


# ----------------------------
# Augmentations
# ----------------------------
def get_transforms(phase):
    if phase == "train":
        return A.Compose([
            A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.15, rotate_limit=45,
                               p=0.5, border_mode=cv2.BORDER_REFLECT_101),
            A.OneOf([
                A.GaussNoise(var_limit=(5.0, 35.0)),
                A.GaussianBlur(blur_limit=(3, 7)),
                A.MotionBlur(blur_limit=7),
            ], p=0.35),
            A.RandomBrightnessContrast(p=0.25),
            A.RandomGamma(p=0.2),
            A.ImageCompression(quality_lower=60, quality_upper=100, p=0.35),
            A.Normalize(),
            ToTensorV2(),
        ], is_check_shapes=False)
    else:
        return A.Compose([
            A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
            A.Normalize(),
            ToTensorV2(),
        ], is_check_shapes=False)


# ----------------------------
# Losses + Metrics
# ----------------------------
def dice_loss_with_logits(logits, targets, eps=1e-7):
    probs = torch.sigmoid(logits)
    num = 2.0 * (probs * targets).sum(dim=(2,3)) + eps
    den = (probs + targets).sum(dim=(2,3)) + eps
    return (1.0 - (num / den)).mean()

def focal_loss_with_logits(logits, targets, alpha=0.25, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    pt = p * targets + (1 - p) * (1 - targets)
    w = (alpha * targets + (1 - alpha) * (1 - targets)) * (1 - pt).pow(gamma)
    return (w * bce).mean()

class CombinedSegLoss(nn.Module):
    def __init__(self, pos_weight=2.0, device="cpu"):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))

    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        dsc = dice_loss_with_logits(logits, targets)
        foc = focal_loss_with_logits(logits, targets)
        return 0.50*bce + 0.30*dsc + 0.20*foc

@torch.no_grad()
def compute_iou_f1(logits, targets, thr=0.5, eps=1e-7):
    probs = torch.sigmoid(logits)
    pred = (probs > thr).float()
    inter = (pred * targets).sum(dim=(2,3))
    union = pred.sum(dim=(2,3)) + targets.sum(dim=(2,3)) - inter
    iou = (inter + eps) / (union + eps)
    f1 = (2*inter + eps) / (pred.sum(dim=(2,3)) + targets.sum(dim=(2,3)) + eps)
    return iou.mean().item(), f1.mean().item()


# ----------------------------
# SRM Residual
# ----------------------------
def make_srm_kernel():
    k1 = np.array([[0,  0,  0],
                   [0,  1, -1],
                   [0,  0,  0]], dtype=np.float32)
    k2 = np.array([[0,  0,  0],
                   [0,  1,  0],
                   [0, -1,  0]], dtype=np.float32)
    k3 = np.array([[ 1, -2,  1],
                   [-2,  4, -2],
                   [ 1, -2,  1]], dtype=np.float32) / 4.0
    return torch.from_numpy(np.stack([k1, k2, k3], axis=0))

class SRMResidual(nn.Module):
    def __init__(self):
        super().__init__()
        kernels = make_srm_kernel()
        weight = torch.zeros((9, 1, 3, 3), dtype=torch.float32)
        for c in range(3):
            for i in range(3):
                weight[c*3+i, 0] = kernels[i]
        self.register_buffer("weight", weight)
        self.pad = 1

    def forward(self, x):
        outs = []
        for c in range(3):
            xc = x[:, c:c+1, :, :]
            yc = F.conv2d(xc, self.weight[c*3:(c+1)*3], padding=self.pad)
            outs.append(yc)
        return torch.cat(outs, dim=1)


# ----------------------------
# Artifact Pyramid
# ----------------------------
class ArtifactPyramid(nn.Module):
    def __init__(self, in_ch=9, base_ch=64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
        )
        self.block1 = self._block(base_ch, base_ch, stride=2)
        self.block2 = self._block(base_ch, base_ch*2, stride=2)     # /4 => 128
        self.block3 = self._block(base_ch*2, base_ch*4, stride=2)   # /8 => 256
        self.block4 = self._block(base_ch*4, base_ch*4, stride=2)   # /16 => 256
        self.down32 = self._block(base_ch*4, base_ch*4, stride=2)   # /32 => 256

    def _block(self, in_ch, out_ch, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x9):
        x = self.stem(x9)
        x = self.block1(x)
        f4  = self.block2(x)
        f8  = self.block3(f4)
        f16 = self.block4(f8)
        f32 = self.down32(f16)
        return [f4, f8, f16, f32]

    # ----------------------------
    # FPN Decoder
    # ----------------------------
    class FPNDecoder(nn.Module):
        def __init__(self, in_channels_list, fpn_dim=256):
            super().__init__()
            self.lateral = nn.ModuleList([nn.Conv2d(c, fpn_dim, 1) for c in in_channels_list])
            self.smooth = nn.ModuleList([nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1) for _ in in_channels_list])
            self.heads = nn.ModuleList([nn.Conv2d(fpn_dim, 1, 1) for _ in in_channels_list])

        def forward(self, feats):
            lat = [l(f) for l, f in zip(self.lateral, feats)]
            p = [None] * len(lat)
            p[-1] = lat[-1]
            for i in reversed(range(len(lat) - 1)):
                up = F.interpolate(p[i + 1], size=lat[i].shape[-2:], mode="bilinear", align_corners=False)
                p[i] = lat[i] + up
            p = [s(pi) for s, pi in zip(self.smooth, p)]
            logits_list = [h(pi) for h, pi in zip(self.heads, p)]
            return p, logits_list

    # ----------------------------
    # Full Model
    # ----------------------------
    class ForgeryNet(nn.Module):
        def __init__(self, backbone_name, fpn_dim=256):
            super().__init__()
            self.backbone = timm.create_model(
                backbone_name,
                pretrained=True,
                features_only=True,
                out_indices=(0, 1, 2, 3),
                in_chans=3
            )
            bb_channels = self.backbone.feature_info.channels()

            self.srm = SRMResidual()
            self.art_pyr = ArtifactPyramid(in_ch=9, base_ch=64)

            base_ch = 64
            art_ch = [base_ch * 2, base_ch * 4, base_ch * 4, base_ch * 4]  # [128,256,256,256]
            self.gate_convs = nn.ModuleList([
                nn.Sequential(nn.Conv2d(a, b, 1, bias=True), nn.Sigmoid())
                for a, b in zip(art_ch, bb_channels)
            ])

            self.decoder = FPNDecoder(bb_channels, fpn_dim=fpn_dim)

            self.final_fuse = nn.Sequential(
                nn.Conv2d(4, fpn_dim, 3, padding=1, bias=False),
                nn.BatchNorm2d(fpn_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(fpn_dim, 1, 1)
            )

            self.cls_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(bb_channels[-1], 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(256, 1)
            )

        def forward(self, x):
            bb_feats = self.backbone(x)

            res9 = self.srm(x)
            art_feats = self.art_pyr(res9)

            fused = []
            for f, a, gate in zip(bb_feats, art_feats, self.gate_convs):
                if a.shape[-2:] != f.shape[-2:]:
                    a = F.interpolate(a, size=f.shape[-2:], mode="bilinear", align_corners=False)
                w = gate(a)
                fused.append(f * (1.0 + w))

            cls_logit = self.cls_head(fused[-1])

            _, ds_logits = self.decoder(fused)
            up_logits = [F.interpolate(li, size=x.shape[-2:], mode="bilinear", align_corners=False) for li in ds_logits]
            cat = torch.cat(up_logits, dim=1)
            mask_logit = self.final_fuse(cat)

            return mask_logit, ds_logits, cls_logit

    # ----------------------------
    # EMA
    # ----------------------------
    class ModelEMA:
        def __init__(self, model, decay=0.995):
            self.decay = decay
            self.ema = self._clone_model(model)
            self.ema.eval()

        def _clone_model(self, model):
            ema = type(model)(Config.BACKBONE_NAME, Config.FPN_DIM)
            ema.load_state_dict(model.state_dict())
            for p in ema.parameters():
                p.requires_grad_(False)
            return ema

        @torch.no_grad()
        def update(self, model):
            msd = model.state_dict()
            esd = self.ema.state_dict()
            for k, v in esd.items():
                if k in msd and v.dtype.is_floating_point:
                    v.copy_(v * self.decay + msd[k] * (1.0 - self.decay))

        def to(self, device):
            self.ema.to(device)
            return self

    # ----------------------------
    # Split + loaders
    # ----------------------------
    train_df, val_df = train_test_split(df, test_size=0.15, stratify=df["label"], random_state=Config.SEED)

    train_ds = SIFDataset(train_df, transform=get_transforms("train"))
    val_ds = SIFDataset(val_df, transform=get_transforms("val"))

    labels = (train_df["label"].values == "forged").astype(np.int64)
    weights = np.where(labels == 1, 2.0, 1.0).astype(np.float32)
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        sampler=sampler,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=True,
        persistent_workers=(Config.PERSISTENT_WORKERS and Config.NUM_WORKERS > 0),
        prefetch_factor=2 if Config.NUM_WORKERS > 0 else None,
        timeout=Config.DL_TIMEOUT if Config.NUM_WORKERS > 0 else 0,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=False,
        persistent_workers=(Config.PERSISTENT_WORKERS and Config.NUM_WORKERS > 0),
        prefetch_factor=2 if Config.NUM_WORKERS > 0 else None,
        timeout=Config.DL_TIMEOUT if Config.NUM_WORKERS > 0 else 0,
    )

    print("Dataloaders ready.")

    # ----------------------------
    # Train setup
    # ----------------------------
    model = ForgeryNet(Config.BACKBONE_NAME, Config.FPN_DIM).to(Config.DEVICE)

    seg_loss_fn = CombinedSegLoss(pos_weight=Config.POS_WEIGHT_SEG, device=Config.DEVICE).to(Config.DEVICE)
    cls_loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([Config.POS_WEIGHT_CLS], device=Config.DEVICE))

    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS, eta_min=1e-6)

    ema = ModelEMA(model, decay=Config.EMA_DECAY).to(Config.DEVICE) if Config.EMA_DECAY is not None else None

    history = {"train_loss": [], "val_loss": [], "val_f1": []}
    best_val = 1e18
    start_epoch = 0

    # ----------------------------
    # Resume
    # ----------------------------
    def load_checkpoint(path):
        global best_val, start_epoch, history
        if not os.path.exists(path):
            print("[CKPT] Not found. Train from scratch.")
            return

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])

        if ema is not None and ckpt.get("ema") is not None:
            ema.ema.load_state_dict(ckpt["ema"])

        best_val = ckpt.get("best_val", best_val)
        history = ckpt.get("history", history)
        start_epoch = ckpt.get("epoch", -1) + 1

        if "rng_state" in ckpt:
            set_rng_state(ckpt["rng_state"])

        print(f"[CKPT] Loaded {path}. Resume from epoch {start_epoch}/{Config.EPOCHS}. best_val={best_val:.6f}")

    def save_checkpoint(epoch, best_val):
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "ema": (ema.ema.state_dict() if ema is not None else None),
            "best_val": best_val,
            "history": history,
            "rng_state": get_rng_state(),
            "config": {k: getattr(Config, k) for k in dir(Config) if k.isupper()},
        }
        atomic_save(ckpt, Config.CHECKPOINT_PATH)

    load_checkpoint(Config.CHECKPOINT_PATH)

    # ----------------------------
    # Train / Val
    # ----------------------------
    def train_one_epoch(epoch):
        model.train()
        total = 0.0
        n = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{Config.EPOCHS} [Train]", leave=False)
        for imgs, masks, cls in pbar:
            imgs = imgs.to(Config.DEVICE, non_blocking=True)
            masks = masks.to(Config.DEVICE, non_blocking=True)
            cls = cls.to(Config.DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast_ctx(enabled=Config.USE_AMP):
                mask_logit, ds_logits, cls_logit = model(imgs)

                loss_main = seg_loss_fn(mask_logit, masks)

                loss_ds = 0.0
                for li in ds_logits:
                    up = F.interpolate(li, size=masks.shape[-2:], mode="bilinear", align_corners=False)
                    loss_ds += seg_loss_fn(up, masks)
                loss_ds = loss_ds / len(ds_logits)

                loss_cls = cls_loss_fn(cls_logit, cls)
                loss = loss_main + 0.25 * loss_ds + Config.LAMBDA_CLS * loss_cls

            scaler.scale(loss).backward()

            if Config.GRAD_CLIP is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP)

            scaler.step(optimizer)
            scaler.update()

            if ema is not None:
                ema.update(model)

            total += loss.item()
            n += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        return total / max(n, 1)

    @torch.no_grad()
    def validate(epoch):
        val_model = ema.ema if ema is not None else model
        val_model.eval()

        total = 0.0
        n = 0
        f1s, ious = [], []

        pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{Config.EPOCHS} [Val]", leave=False)
        for imgs, masks, cls in pbar:
            imgs = imgs.to(Config.DEVICE, non_blocking=True)
            masks = masks.to(Config.DEVICE, non_blocking=True)

            with autocast_ctx(enabled=Config.USE_AMP):
                mask_logit, ds_logits, cls_logit = val_model(imgs)

                loss_main = seg_loss_fn(mask_logit, masks)

                loss_ds = 0.0
                for li in ds_logits:
                    up = F.interpolate(li, size=masks.shape[-2:], mode="bilinear", align_corners=False)
                    loss_ds += seg_loss_fn(up, masks)
                loss_ds = loss_ds / len(ds_logits)

                loss = loss_main + 0.25 * loss_ds

            iou, f1 = compute_iou_f1(mask_logit, masks, thr=0.5)
            ious.append(iou);
            f1s.append(f1)

            total += loss.item()
            n += 1
            pbar.set_postfix(val_loss=f"{loss.item():.4f}", f1=f"{f1:.4f}")

        return total / max(n, 1), float(np.mean(ious)), float(np.mean(f1s))

    # ----------------------------
    # Training
    # ----------------------------
    print(
        f"Start training: start_epoch={start_epoch}, epochs={Config.EPOCHS}, batch={Config.BATCH_SIZE}, img={Config.IMG_SIZE}")
    for epoch in range(start_epoch, Config.EPOCHS):
        t0 = time.time()

        train_loss = train_one_epoch(epoch)
        val_loss, val_iou, val_f1 = validate(epoch)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)

        dt = time.time() - t0
        print(f"Epoch {epoch + 1:02d}/{Config.EPOCHS} | "
              f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_iou={val_iou:.4f} | val_f1={val_f1:.4f} | time={dt:.1f}s")

        save_checkpoint(epoch, best_val)
        print(f"  [CKPT] saved -> {Config.CHECKPOINT_PATH}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = (ema.ema.state_dict() if ema is not None else model.state_dict())
            atomic_save(best_state, Config.BEST_PATH)
            print(f"  >>> Best saved -> {Config.BEST_PATH} (best_val={best_val:.4f})")

    print("Training done.")
    print("Best model path:", Config.BEST_PATH)
    print("Resume checkpoint:", Config.CHECKPOINT_PATH)

    import os
    import numpy as np
    import cv2
    import torch
    import matplotlib.pyplot as plt
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    # -----------------------------
    # 1) Paths: change if needed
    # -----------------------------
    BEST_PATH = "/kaggle/input/dual-branch-forensics-segmentation-transformer/best_model.pth"
    TRAIN_IMG_DIR = "/kaggle/input/recodai-luc-scientific-image-forgery-detection/train_images"
    TRAIN_MASK_DIR = "/kaggle/input/recodai-luc-scientific-image-forgery-detection/train_masks"

    assert os.path.exists(BEST_PATH), f"best model not found: {BEST_PATH}"
    assert "df" in globals(), "df not found. Please make sure you created df with columns: image_path, mask_path, label"
    assert "ForgeryNet" in globals(), "ForgeryNet class not found. Please run the training/model definition cells first"
    assert "Config" in globals(), "Config not found. Please run the config cell first"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Using BEST_PATH:", BEST_PATH)

    # -----------------------------
    # 2) Build model and load weights (robust)
    # -----------------------------
    model = ForgeryNet(Config.BACKBONE_NAME, Config.FPN_DIM).to(device)
    model.eval()

    sd = torch.load(BEST_PATH, map_location="cpu", weights_only=False)

    # Handle possible formats:
    # (a) pure state_dict
    # (b) checkpoint dict with key 'model' / 'state_dict'
    if isinstance(sd, dict) and ("model" in sd or "state_dict" in sd):
        sd = sd.get("model", sd.get("state_dict"))

    # Remove "module." if saved under DataParallel
    if isinstance(sd, dict):
        new_sd = {}
        for k, v in sd.items():
            if k.startswith("module."):
                new_sd[k[len("module."):]] = v
            else:
                new_sd[k] = v
        sd = new_sd

    missing, unexpected = model.load_state_dict(sd, strict=False)
    print("Loaded weights.")
    print("Missing keys:", len(missing), "Unexpected keys:", len(unexpected))

    # -----------------------------
    # 3) Val transform (Resize + Normalize + ToTensor)
    # -----------------------------
    val_tf = A.Compose([
        A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
        A.Normalize(),
        ToTensorV2(),
    ], is_check_shapes=False)

    def read_rgb(path):
        img = cv2.imread(path)
        if img is None:
            return np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def read_mask(mask_path, h, w):
        if mask_path is None or (isinstance(mask_path, float) and np.isnan(mask_path)):
            return np.zeros((h, w), np.float32)

        if str(mask_path).endswith(".npy"):
            try:
                m = np.load(mask_path)
            except Exception:
                m = np.zeros((h, w), np.float32)
        else:
            m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if m is None:
                m = np.zeros((h, w), np.float32)

        if m.ndim == 3:
            m = m.max(axis=2)

        if m.shape != (h, w):
            m = cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)

        return (m > 0).astype(np.float32)

    @torch.no_grad()
    def infer_one(img_np):
        # img_np: RGB uint8
        h, w = img_np.shape[:2]
        out = val_tf(image=img_np, mask=np.zeros((h, w), np.float32))
        x = out["image"].unsqueeze(0).to(device)  # [1,3,H,W]
        mask_logit, _, _ = model(x)
        prob = torch.sigmoid(mask_logit)[0, 0].detach().float().cpu().numpy()
        return prob

    def binarize(prob, thr=0.5):
        return (prob >= thr).astype(np.float32)

    # -----------------------------
    # 4) Pick 5 authentic + 5 forged (prefer forged with masks)
    # -----------------------------
    auth_df = df[df["label"] == "authentic"].sample(5, random_state=Config.SEED) if (
                df["label"] == "authentic").any() else df.iloc[:0]
    forg_df = df[(df["label"] == "forged") & (df["mask_path"].notnull())]
    if len(forg_df) >= 5:
        forg_df = forg_df.sample(5, random_state=Config.SEED + 1)
    else:
        # fallback: take forged even without mask if not enough
        forg_df = df[df["label"] == "forged"].sample(min(5, (df["label"] == "forged").sum()),
                                                     random_state=Config.SEED + 1)

    test_df = np.concatenate([auth_df.to_dict("records"), forg_df.to_dict("records")], axis=0)

    print(f"Selected: authentic={len(auth_df)} forged={len(forg_df)} total={len(test_df)}")

    # -----------------------------
    # 5) Visualize
    # -----------------------------
    thr = 0.3
    for idx, row in enumerate(test_df, start=1):
        img = read_rgb(row["image_path"])
        h, w = img.shape[:2]
        gt = read_mask(row.get("mask_path", None), h, w)

        # apply same resize to GT for fair compare with model output
        out = val_tf(image=img, mask=gt)
        img_t = out["image"]  # tensor [3,H,W]
        gt_t = out["mask"]  # tensor [H,W] float
        gt_np = gt_t.cpu().numpy()

        prob = infer_one(img)  # [H,W] after resize
        pred = binarize(prob, thr=thr)

        # metrics (for authentic GT is all-zeros -> interpret carefully)
        # convert to torch for compute_iou_f1 if available
        if "compute_iou_f1" in globals():
            with torch.no_grad():
                logits_t = torch.from_numpy(np.log(prob + 1e-7) - np.log(1 - prob + 1e-7)).unsqueeze(0).unsqueeze(0)
                target_t = torch.from_numpy(gt_np).unsqueeze(0).unsqueeze(0)
                iou, f1 = compute_iou_f1(logits_t, target_t, thr=thr)
        else:
            iou, f1 = np.nan, np.nan

        label = row["label"]
        title = f"[{idx}/10] {label} | IoU={iou:.4f} F1={f1:.4f}"

        plt.figure(figsize=(16, 4))
        plt.suptitle(title)

        plt.subplot(1, 4, 1)
        plt.title("Image")
        plt.imshow(img)
        plt.axis("off")

        plt.subplot(1, 4, 2)
        plt.title("GT Mask")
        plt.imshow(gt_np, cmap="gray", vmin=0, vmax=1)
        plt.axis("off")

        plt.subplot(1, 4, 3)
        plt.title("Pred Prob")
        plt.imshow(prob, cmap="gray", vmin=0, vmax=1)
        plt.axis("off")

        plt.subplot(1, 4, 4)
        plt.title(f"Pred Binary (thr={thr})")
        plt.imshow(pred, cmap="gray", vmin=0, vmax=1)
        plt.axis("off")

        plt.show()
