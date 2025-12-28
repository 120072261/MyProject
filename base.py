import os
import cv2
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler  # 混合精度训练
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from tqdm import tqdm  # 替换 tqdm.notebook 为 tqdm


# 全局配置
class Config:
    SEED = 42
    IMG_SIZE = 512  # 输入尺寸
    BATCH_SIZE = 16  # P100 显存适配 (EffNet-B4 比较大，设12或16)
    EPOCHS = 20  # 训练轮数
    LR = 2e-4  # 学习率
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 2  # 降低工作进程数，避免 Kaggle 资源溢出
    CHECKPOINT_PATH = "checkpoint.pth"  # 检查点保存路径


def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    # 新增：固定 cudnn 种子，确保结果可复现
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(Config.SEED)
print(f"Device: {Config.DEVICE}")  # 3. 智能路径搜索与匹配


# ==========================================
def find_dataset_root():
    possible_roots = [
        '/kaggle/input/recodai-luc-scientific-image-forgery-detection',
        '/kaggle/input/scientific-image-forgery-detection',
        './'
    ]
    for path in possible_roots:
        if os.path.exists(os.path.join(path, 'train_images')):
            return path
    return None


ROOT_DIR = find_dataset_root()
if not ROOT_DIR:
    raise ValueError("❌ 未找到数据集路径！请检查 Input 目录。")

print(f"✅ 数据集根目录: {ROOT_DIR}")

# 定义路径
TRAIN_IMG_DIR = os.path.join(ROOT_DIR, 'train_images')
TRAIN_MASK_DIR = os.path.join(ROOT_DIR, 'train_masks')
if not os.path.exists(TRAIN_MASK_DIR):
    TRAIN_MASK_DIR = os.path.join(TRAIN_IMG_DIR, 'train_masks')  # 备用路径


# 递归获取所有文件
def get_all_files(directory, extensions):
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(directory, '**', ext), recursive=True))
    return sorted(list(set(files)))


# 1. 收集所有图片
image_files = get_all_files(TRAIN_IMG_DIR, ['*.jpg', '*.png', '*.tif', '*.tiff'])

# 2. 收集所有 Mask (包括 .npy)
mask_files = get_all_files(TRAIN_MASK_DIR, ['*.jpg', '*.png', '*.npy'])
# 建立 Mask 索引字典: { 'case_id': 'full_path' }
mask_map = {os.path.basename(p).split('.')[0]: p for p in mask_files}

# 3. 构建 DataFrame
data = []
for img_path in image_files:
    case_id = os.path.basename(img_path).split('.')[0]

    # 判断是否 authentic (根据路径名)
    label = 'authentic' if 'authentic' in img_path.lower() else 'forged'

    # 尝试匹配 Mask
    mask_path = mask_map.get(case_id)

    # 修正逻辑：如果是 Authentic，强制 Mask 为 None
    if label == 'authentic':
        mask_path = None

    data.append({
        'case_id': case_id,
        'image_path': img_path,
        'mask_path': mask_path,
        'label': label
    })

df = pd.DataFrame(data)
print(f"\n📊 数据统计:")
print(f"总样本数: {len(df)}")
print(df['label'].value_counts())
print(f"成功匹配 Mask 的 Forged 样本: {df[df['label'] == 'forged']['mask_path'].notnull().sum()}")  # 数据集定义


class SIFDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # --- 1. Image ---
        image = cv2.imread(row['image_path'])
        if image is None:
            image = np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w = image.shape[:2]

        # --- 2. Mask ---
        mask_path = row['mask_path']

        if mask_path is None:
            mask = np.zeros((h, w), dtype=np.float32)

        else:
            if mask_path.endswith('.npy'):
                try:
                    mask = np.load(mask_path)
                except Exception:
                    mask = np.zeros((h, w), dtype=np.float32)
            else:
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    mask = np.zeros((h, w), dtype=np.float32)

            # ===== 第一次压维（numpy 阶段）=====
            if mask.ndim == 3:
                # HWC 或 CHW
                if mask.shape[0] == h and mask.shape[1] == w:
                    mask = mask.max(axis=2)
                else:
                    mask = mask.max(axis=0)

            # 对齐尺寸
            if mask.shape != (h, w):
                mask = cv2.resize(mask.astype(np.float32),
                                  (w, h),
                                  interpolation=cv2.INTER_NEAREST)

            mask = (mask > 0).astype(np.float32)

        # --- 3. Augmentation ---
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # ===== 第二次压维（tensor 阶段，关键！）=====
        if isinstance(mask, torch.Tensor):
            if mask.ndim == 3:
                mask = mask.max(dim=0).values  # [H, W]
            elif mask.ndim != 2:
                raise RuntimeError(f"Invalid mask tensor shape: {mask.shape}")

        return image, mask.unsqueeze(0)  # [1, H, W]# 数据增强


def get_transforms(phase):
    if phase == 'train':
        return A.Compose([
            A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0)),
                A.GaussianBlur(),
                A.MotionBlur(),
            ], p=0.3),
            A.Normalize(),
            ToTensorV2(),
        ], is_check_shapes=False)
    else:
        return A.Compose([
            A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
            A.Normalize(),
            ToTensorV2(),
        ], is_check_shapes=False)


# 划分数据集
train_df, val_df = train_test_split(df, test_size=0.15, stratify=df['label'], random_state=Config.SEED)

train_ds = SIFDataset(train_df, transform=get_transforms('train'))
val_ds = SIFDataset(val_df, transform=get_transforms('val'))

train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True,
                          num_workers=Config.NUM_WORKERS, pin_memory=True, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False,
                        num_workers=Config.NUM_WORKERS, pin_memory=True)

print("✅ DataLoader 准备就绪")  # 构建模型


def build_model():
    model = smp.UnetPlusPlus(
        encoder_name="efficientnet-b4",  # 强力 Backbone
        encoder_weights="imagenet",  # 预训练权重加速收敛
        in_channels=3,
        classes=1,
        activation=None,  # 输出 Logits，在 Loss 中做 Sigmoid
    )
    return model


model = build_model()
model.to(Config.DEVICE)
print("✅ U-Net++ 模型已加载 (EfficientNet-B4)")  # 构建模型


def build_model():
    model = smp.UnetPlusPlus(
        encoder_name="efficientnet-b4",  # 强力 Backbone
        encoder_weights="imagenet",  # 预训练权重加速收敛
        in_channels=3,
        classes=1,
        activation=None,  # 输出 Logits，在 Loss 中做 Sigmoid
    )
    return model


model = build_model()
model.to(Config.DEVICE)
print("✅ U-Net++ 模型已加载 (EfficientNet-B4)")

# 损失函数
dice_loss = smp.losses.DiceLoss(mode='binary')
pos_weight = torch.tensor([2.0]).to(Config.DEVICE)
bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def criterion(pred, target):
    return 0.5 * dice_loss(pred, target) + 0.5 * bce_loss(pred, target)


# 评估指标
def compute_scores(pred_logits, target):
    pred_probs = torch.sigmoid(pred_logits)
    pred_mask = (pred_probs > 0.5).float()

    # Intersection & Union
    intersection = (pred_mask * target).sum()
    union = pred_mask.sum() + target.sum()

    iou = (intersection + 1e-7) / (union - intersection + 1e-7)
    f1 = (2 * intersection + 1e-7) / (union + 1e-7)

    return iou.item(), f1.item()


optimizer = optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=1e-3)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS, eta_min=1e-6)
scaler = GradScaler()  # 混合精度

best_loss = float('inf')
history = {'train_loss': [], 'val_loss': [], 'val_f1': []}

print(f"🚀 开始训练... (Epochs: {Config.EPOCHS})")


# 加载检查点（如果存在）
def load_checkpoint(model, optimizer, scaler, checkpoint_path=Config.CHECKPOINT_PATH):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"✅ 加载模型从检查点恢复，Epoch {epoch}, Loss {loss}")
        return model, optimizer, scaler, epoch
    else:
        print("没有找到检查点，开始新的训练")
        return model, optimizer, scaler, 0


model, optimizer, scaler, start_epoch = load_checkpoint(model, optimizer, scaler)

for epoch in range(start_epoch, Config.EPOCHS):
    model.train()
    train_loss = 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{Config.EPOCHS} [Train]", leave=True)
    for imgs, masks in loop:
        imgs, masks = imgs.to(Config.DEVICE), masks.to(Config.DEVICE)

        optimizer.zero_grad()
        with autocast():
            outputs = model(imgs)
            loss = criterion(outputs, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        loop.set_postfix({"train_loss": f"{loss.item():.4f}"})

    # 验证步骤
    model.eval()
    val_loss = 0
    val_iou = 0
    val_f1 = 0

    val_loop = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{Config.EPOCHS} [Val]", leave=True)
    with torch.no_grad():
        for imgs, masks in val_loop:
            imgs, masks = imgs.to(Config.DEVICE), masks.to(Config.DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, masks)

            val_loss += loss.item()
            iou, f1 = compute_scores(outputs, masks)
            val_iou += iou
            val_f1 += f1
            val_loop.set_postfix({"val_loss": f"{loss.item():.4f}", "val_f1": f"{f1:.4f}"})

    # 计算平均值
    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    val_iou /= len(val_loader)
    val_f1 /= len(val_loader)

    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['val_f1'].append(val_f1)

    scheduler.step()

    print(
        f"📝 Epoch {epoch + 1} Summary | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")

    # 保存检查点
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'loss': val_loss
    }, Config.CHECKPOINT_PATH)

    # 保存最佳模型
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), "best_model.pth")
        print(f"    >>> 💾 Best Model Saved (Val Loss: {val_loss:.4f}, F1: {val_f1:.4f})")

# 绘制曲线
plt.figure(figsize=(10, 5))
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 绘制 F1 曲线
plt.figure(figsize=(10, 5))
plt.plot(history['val_f1'], label='Val F1 Score', color='green')
plt.title('Validation F1 Score')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.legend()
plt.show()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 512

# 自动寻找数据集路径
POSSIBLE_ROOTS = [
    '/kaggle/input/recodai-luc-scientific-image-forgery-detection',
    '/kaggle/input/scientific-image-forgery-detection',
    './'
]
ROOT_DIR = None
for path in POSSIBLE_ROOTS:
    if os.path.exists(path) and os.path.exists(os.path.join(path, 'train_images')):
        ROOT_DIR = path
        break

if ROOT_DIR is None:
    # 如果没找到，尝试硬编码 (根据你之前的报错信息)
    ROOT_DIR = '/kaggle/input/recodai-luc-scientific-image-forgery-detection'

TRAIN_IMG_DIR = os.path.join(ROOT_DIR, 'train_images')
AUTH_DIR = os.path.join(TRAIN_IMG_DIR, 'authentic')
FORGED_DIR = os.path.join(TRAIN_IMG_DIR, 'forged')
# 优先找 Input 里的 masks，如果没有再找 working 里的
MASK_DIR_OPT1 = os.path.join(ROOT_DIR, 'train_masks')
MASK_DIR = MASK_DIR_OPT1 if os.path.exists(MASK_DIR_OPT1) else os.path.join(TRAIN_IMG_DIR, 'train_masks')

print(f"Dataset Root: {ROOT_DIR}")
print(f"Mask Dir: {MASK_DIR}")


# ==========================================
# 2. 重新定义模型结构 (必须与训练时一致)
# ==========================================
def build_model():
    model = smp.UnetPlusPlus(
        encoder_name="efficientnet-b4",
        encoder_weights=None,  # 推理不需要下载 ImageNet 权重
        in_channels=3,
        classes=1,
    )
    return model


# ==========================================
# 3. 重新定义数据处理类 (修复版)
# ==========================================
class ScientificDatasetNPY(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = cv2.imread(row['image_path'])
        if image is None:
            image = np.zeros((512, 512, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        if row['mask_path'] is None:
            mask = np.zeros((h, w), dtype=np.float32)
        else:
            try:
                # 读取 .npy
                mask = np.load(row['mask_path'])
                mask = mask.astype(np.float32)
                if len(mask.shape) == 3: mask = mask.squeeze()  # 压扁 3D mask
                if mask.shape[:2] != (h, w):
                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            except:
                mask = np.zeros((h, w), dtype=np.float32)

        mask = np.where(mask > 0.5, 1.0, 0.0).astype(np.float32)
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        return image, mask.unsqueeze(0)


def get_transforms():
    return A.Compose([A.Resize(IMG_SIZE, IMG_SIZE), A.Normalize(), ToTensorV2()])


# ==========================================
# 4. 尝试加载模型权重
# ==========================================
MODEL_PATH = "best_model.pth"  # 默认在当前目录

if not os.path.exists(MODEL_PATH):
    # 尝试在 input 里找 (如果你是通过 Add Data 添加的)
    possible_paths = glob.glob('/kaggle/input/**/best_model.pth', recursive=True)
    if possible_paths:
        MODEL_PATH = possible_paths[0]
    else:
        print("❌ 错误：在当前目录或 Input 中找不到 best_model.pth！")
        print("请确认你之前是否训练并保存了模型。如果文件丢失，必须重新运行训练代码。")
        # 这里抛出异常停止运行，避免后面报错
        raise FileNotFoundError("Model file not found.")

print(f"正在加载模型: {MODEL_PATH}")
model = build_model()
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

# 自动修复 DataParallel 的 'module.' 前缀
if 'module.' in list(checkpoint.keys())[0]:
    new_state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}
    model.load_state_dict(new_state_dict)
else:
    model.load_state_dict(checkpoint)

model.to(DEVICE)
model.eval()
print("✅ 模型加载成功！准备测试...")

# ==========================================
# 5. 准备测试数据 (只找伪造图进行可视化)
# ==========================================
# 快速构建一个只包含 Forged 图片的临时 DataFrame
forged_img_files = sorted(glob.glob(os.path.join(FORGED_DIR, '*.*')))
mask_npy_files = glob.glob(os.path.join(MASK_DIR, '*.npy'))
mask_map = {os.path.basename(m).split('.')[0]: m for m in mask_npy_files}

data = []
for img_path in forged_img_files:
    case_id = os.path.basename(img_path).split('.')[0]
    if case_id in mask_map:
        data.append({'image_path': img_path, 'mask_path': mask_map[case_id]})

if not data:
    print("❌ 警告：未匹配到任何伪造数据，请检查路径。")
else:
    # 随机取 5 张
    val_df = pd.DataFrame(data).sample(n=min(5, len(data)), random_state=42)
    val_ds = ScientificDatasetNPY(val_df, transform=get_transforms())
    val_loader = DataLoader(val_ds, batch_size=5, shuffle=False)

    # ==========================================
    # 6. 可视化对比 (原图 vs 真值 vs 预测)
    # ==========================================
    print("🚀 正在生成对比图...")
    images, true_masks = next(iter(val_loader))
    images = images.to(DEVICE)

    with torch.no_grad():
        preds = model(images)
        preds = torch.sigmoid(preds)

    # 转 CPU
    images = images.cpu().numpy()
    true_masks = true_masks.numpy()
    preds = preds.cpu().numpy()

    # 反归一化
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    fig, axes = plt.subplots(len(images), 3, figsize=(12, 4 * len(images)))
    if len(images) == 1: axes = np.expand_dims(axes, axis=0)  # 兼容单张图

    for i in range(len(images)):
        # A. 原图
        img = images[i].transpose(1, 2, 0)
        img = img * std + mean
        img = np.clip(img, 0, 1)
        axes[i, 0].imshow(img)
        axes[i, 0].set_title("Original Image")
        axes[i, 0].axis('off')

        # B. 真值 (Ground Truth)
        axes[i, 1].imshow(true_masks[i].squeeze(), cmap='gray', vmin=0, vmax=1)
        axes[i, 1].set_title("Ground Truth (Real)")
        axes[i, 1].axis('off')

        # C. 预测 (Prediction)
        # 二值化显示
        p_mask = (preds[i].squeeze() > 0.5).astype(np.float32)
        axes[i, 2].imshow(p_mask, cmap='gray', vmin=0, vmax=1)
        axes[i, 2].set_title("Prediction (Model)")
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.show()