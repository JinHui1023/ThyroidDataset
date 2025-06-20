import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.quantization import QuantStub, DeQuantStub, prepare_qat, convert
from sklearn.metrics import roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import onnx
import tensorrt as trt
from PIL import Image

# =====================
# 数据预处理与增强
# =====================

class UltrasoundPreprocessor:
    def __init__(self, target_size=(256, 256), clip_limit=2.0, tile_grid_size=(8, 8)):
        self.target_size = target_size
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        
    def remove_probe_markings(self, img):
        """去除探头标记和水印"""
        # 阈值处理分离标记
        _, thresh = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 创建掩膜去除大块标记
        mask = np.ones_like(img) * 255
        for cnt in contours:
            if cv2.contourArea(cnt) > 50:
                cv2.drawContours(mask, [cnt], 0, 0, -1)
        
        return cv2.bitwise_and(img, mask)
    
    def apply_clahe(self, img):
        """应用CLAHE对比度增强"""
        return self.clahe.apply(img)
    
    def normalize(self, img):
        """归一化处理"""
        return (img - img.min()) / (img.max() - img.min() + 1e-8)
    
    def preprocess(self, img_path):
        """完整预处理流程"""
        # 读取并转换为灰度图
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"无法读取图像: {img_path}")
        
        # 预处理步骤
        img = self.remove_probe_markings(img)
        img = self.apply_clahe(img)
        img = self.normalize(img) * 255
        
        # 调整尺寸
        img = cv2.resize(img, self.target_size)
        return img.astype(np.uint8)

# GAN数据增强（简化版）
class DCGAN(nn.Module):
    """用于生成恶性结节的DCGAN模型"""
    def __init__(self, latent_dim=100):
        super().__init__()
        self.generator = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    
    def forward(self, z):
        return self.generator(z)
    
    def generate(self, num_samples, device):
        """生成恶性结节样本"""
        z = torch.randn(num_samples, 100, 1, 1, device=device)
        with torch.no_grad():
            generated = self(z)
            # 将(-1,1)映射到(0,255)
            generated = (generated + 1) * 127.5
            return generated.cpu().numpy().squeeze().astype(np.uint8)

# =====================
# 多任务模型架构
# =====================

class MultiTaskUNetPP(nn.Module):
    """多任务UNet++模型，同时进行分割和分类"""
    def __init__(self, num_classes=2):
        super().__init__()
        
        # 编码器部分
        self.encoder = nn.Sequential(
            self._conv_block(1, 64),
            nn.MaxPool2d(2),
            self._conv_block(64, 128),
            nn.MaxPool2d(2),
            self._conv_block(128, 256),
            nn.MaxPool2d(2),
            self._conv_block(256, 512),
            nn.MaxPool2d(2),
        )
        
        # 瓶颈层
        self.bottleneck = self._conv_block(512, 1024)
        
        # 分割解码器
        self.up1 = self._upsample_block(1024, 512)
        self.dec1 = self._conv_block(1024, 512)
        self.up2 = self._upsample_block(512, 256)
        self.dec2 = self._conv_block(512, 256)
        self.up3 = self._upsample_block(256, 128)
        self.dec3 = self._conv_block(256, 128)
        self.up4 = self._upsample_block(128, 64)
        self.dec4 = self._conv_block(128, 64)
        self.seg_out = nn.Conv2d(64, 1, 1)
        
        # 分类分支
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        
        # 量化支持
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
    
    def _conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )
    
    def _upsample_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )
    
    def forward(self, x):
        # 量化输入
        x = self.quant(x)
        
        # 编码
        e1 = self.encoder[0:2](x)
        e2 = self.encoder[2:4](e1)
        e3 = self.encoder[4:6](e2)
        e4 = self.encoder[6:8](e3)
        
        # 瓶颈
        b = self.bottleneck(e4)
        
        # 解码
        d1 = self.up1(b)
        d1 = torch.cat([d1, e4], dim=1)
        d1 = self.dec1(d1)
        
        d2 = self.up2(d1)
        d2 = torch.cat([d2, e3], dim=1)
        d2 = self.dec2(d2)
        
        d3 = self.up3(d2)
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.dec3(d3)
        
        d4 = self.up4(d3)
        d4 = torch.cat([d4, e1], dim=1)
        d4 = self.dec4(d4)
        
        # 分割输出
        seg_mask = torch.sigmoid(self.seg_out(d4))
        
        # 分类输出
        cls_logits = self.classifier(b)
        
        # 反量化
        seg_mask = self.dequant(seg_mask)
        cls_logits = self.dequant(cls_logits)
        
        return seg_mask, cls_logits

# =====================
# 训练与评估
# =====================

class ThyroidDataset(Dataset):
    """甲状腺超声图像数据集"""
    def __init__(self, image_dir, mask_dir, label_file, transform=None, preprocessor=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.labels = self._load_labels(label_file)
        self.transform = transform
        self.preprocessor = preprocessor or UltrasoundPreprocessor()
        
        # 收集有效样本
        self.samples = []
        for img_name in os.listdir(image_dir):
            if img_name.endswith('.png'):
                img_id = img_name.split('.')[0]
                mask_path = os.path.join(mask_dir, f"{img_id}_mask.png")
                if os.path.exists(mask_path) and img_id in self.labels:
                    self.samples.append((img_id, self.labels[img_id]))
    
    def _load_labels(self, label_file):
        """加载标签文件"""
        labels = {}
        with open(label_file, 'r') as f:
            for line in f:
                img_id, label = line.strip().split(',')
                labels[img_id] = int(label)
        return labels
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_id, label = self.samples[idx]
        img_path = os.path.join(self.image_dir, f"{img_id}.png")
        mask_path = os.path.join(self.mask_dir, f"{img_id}_mask.png")
        
        # 预处理图像
        img = self.preprocessor.preprocess(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, img.shape[::-1]) / 255.0
        
        # 应用变换
        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)
        
        return img, mask, torch.tensor(label, dtype=torch.long)

# 损失函数
class DiceBCELoss(nn.Module):
    """分割任务的Dice+BCE损失"""
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        dice_loss = 1 - (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        bce_loss = F.binary_cross_entropy(pred, target, reduction='mean')
        return dice_loss + bce_loss

class FocalLoss(nn.Module):
    """分类任务的Focal Loss"""
    def __init__(self, alpha=0.8, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

# 模型训练
def train_model(model, train_loader, val_loader, device, epochs=50):
    # 损失函数
    seg_criterion = DiceBCELoss()
    cls_criterion = FocalLoss()
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.5)
    
    # 训练记录
    history = {'train_loss': [], 'val_dice': [], 'val_auc': []}
    best_auc = 0.0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        # 训练阶段
        for images, masks, labels in train_loader:
            images, masks, labels = images.to(device), masks.to(device), labels.to(device)
            
            optimizer.zero_grad()
            seg_pred, cls_pred = model(images)
            
            # 计算多任务损失
            seg_loss = seg_criterion(seg_pred, masks.unsqueeze(1))
            cls_loss = cls_criterion(cls_pred, labels)
            loss = seg_loss + cls_loss
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        
        # 验证阶段
        val_dice, val_auc = evaluate_model(model, val_loader, device)
        history['val_dice'].append(val_dice)
        history['val_auc'].append(val_auc)
        
        # 学习率调度
        scheduler.step(val_auc)
        
        # 保存最佳模型
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), 'best_model.pth')
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, "
              f"Val Dice: {val_dice:.4f}, Val AUC: {val_auc:.4f}")
    
    # 保存训练曲线
    plot_training_history(history)
    return model

def evaluate_model(model, data_loader, device):
    model.eval()
    seg_preds, seg_targets, cls_preds, cls_targets = [], [], [], []
    
    with torch.no_grad():
        for images, masks, labels in data_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            seg_out, cls_out = model(images)
            seg_preds.append(seg_out.cpu())
            seg_targets.append(masks.cpu().unsqueeze(1))
            
            cls_probs = F.softmax(cls_out, dim=1)
            cls_preds.append(cls_probs[:, 1].cpu().numpy())
            cls_targets.append(labels.cpu().numpy())
    
    # 计算Dice系数
    seg_preds = torch.cat(seg_preds)
    seg_targets = torch.cat(seg_targets)
    dice_score = dice_coeff(seg_preds > 0.5, seg_targets)
    
    # 计算AUC
    cls_preds = np.concatenate(cls_preds)
    cls_targets = np.concatenate(cls_targets)
    auc_score = roc_auc_score(cls_targets, cls_preds)
    
    return dice_score.item(), auc_score

def dice_coeff(pred, target):
    smooth = 1e-5
    intersection = (pred & target).float().sum()
    return (2. * intersection + smooth) / (pred.float().sum() + target.float().sum() + smooth)

def plot_training_history(history):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_dice'], label='Val Dice')
    plt.plot(history['val_auc'], label='Val AUC')
    plt.title('Validation Metrics')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.savefig('training_history.png')
    plt.close()

# =====================
# 模型轻量化
# =====================

def knowledge_distillation(teacher, student, train_loader, device, epochs=20):
    teacher.eval()
    student.train()
    
    # 损失函数
    seg_criterion = DiceBCELoss()
    cls_criterion = nn.KLDivLoss()
    
    # 优化器
    optimizer = optim.Adam(student.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        for images, masks, labels in train_loader:
            images, masks, labels = images.to(device), masks.to(device), labels.to(device)
            
            # 教师预测
            with torch.no_grad():
                t_seg, t_cls = teacher(images)
                t_cls_probs = F.softmax(t_cls, dim=1)
            
            # 学生预测
            s_seg, s_cls = student(images)
            
            # 分割损失
            seg_loss = seg_criterion(s_seg, masks.unsqueeze(1))
            
            # 分类蒸馏损失
            s_cls_log_probs = F.log_softmax(s_cls, dim=1)
            cls_loss = cls_criterion(s_cls_log_probs, t_cls_probs)
            
            # 总损失
            loss = seg_loss + cls_loss
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Distillation Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    
    return student

def quantize_model(model):
    """应用量化感知训练"""
    model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    quant_model = prepare_qat(model.train())
    
    # 实际中需要额外训练步骤
    quant_model.eval()
    quant_model = convert(quant_model)
    return quant_model

def prune_model(model, amount=0.4):
    """结构化剪枝"""
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            parameters_to_prune.append((module, 'weight'))
    
    # 全局剪枝
    torch.nn.utils.prune.global_unstructured(
        parameters_to_prune,
        pruning_method=torch.nn.prune.L1Unstructured,
        amount=amount
    )
    
    # 移除剪枝掩码，使剪枝永久化
    for module, _ in parameters_to_prune:
        torch.nn.utils.prune.remove(module, 'weight')
    
    return model

# =====================
# 模型部署
# =====================

def export_to_onnx(model, input_size=(1, 256, 256), output_path="thyroid_model.onnx"):
    """导出模型到ONNX格式"""
    dummy_input = torch.randn(1, *input_size)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["seg_mask", "cls_logits"],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'seg_mask': {0: 'batch_size'},
            'cls_logits': {0: 'batch_size'}
        },
        opset_version=13
    )
    
    # 验证ONNX模型
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print(f"ONNX模型已成功导出至 {output_path}")

def build_tensorrt_engine(onnx_path, engine_path="thyroid_model.trt"):
    """使用TensorRT构建优化引擎"""
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    
    # 解析ONNX模型
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise ValueError("ONNX解析失败")
    
    # 配置优化参数
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
    config.set_flag(trt.BuilderFlag.FP16)
    
    # 构建引擎
    serialized_engine = builder.build_serialized_network(network, config)
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)
    
    print(f"TensorRT引擎已成功构建至 {engine_path}")
    return engine_path

class TRTInference:
    """TensorRT推理引擎"""
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        
        # 加载引擎
        with open(engine_path, 'rb') as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        self.input_shape = self.engine.get_binding_shape(0)
        
        # 分配输入/输出内存
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()
    
    def allocate_buffers(self):
        inputs, outputs, bindings = [], [], []
        stream = cuda.Stream()
        
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            
            # 分配内存
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            bindings.append(int(device_mem))
            
            if self.engine.binding_is_input(binding):
                inputs.append({'host': host_mem, 'device': device_mem})
            else:
                outputs.append({'host': host_mem, 'device': device_mem})
        
        return inputs, outputs, bindings, stream
    
    def preprocess(self, image):
        """预处理输入图像"""
        # 应用与训练相同的预处理
        preprocessor = UltrasoundPreprocessor()
        processed = preprocessor.preprocess(image)
        
        # 转换为Tensor
        tensor = torch.tensor(processed, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        return tensor.numpy()
    
    def infer(self, image_path):
        """执行推理"""
        # 预处理
        input_data = self.preprocess(image_path)
        
        # 复制数据到设备
        np.copyto(self.inputs[0]['host'], input_data.ravel())
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        
        # 执行推理
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle
        )
        
        # 复制结果回主机
        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        cuda.memcpy_dtoh_async(self.outputs[1]['host'], self.outputs[1]['device'], self.stream)
        self.stream.synchronize()
        
        # 获取输出
        seg_output = self.outputs[0]['host'].reshape(1, 256, 256)
        cls_output = self.outputs[1]['host'].reshape(1, 2)
        
        return seg_output, cls_output

# =====================
# 主执行流程
# =====================

def main():
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 数据预处理和增强
    preprocessor = UltrasoundPreprocessor()
    
    # 数据集加载
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    train_dataset = ThyroidDataset(
        image_dir='data/train/images',
        mask_dir='data/train/masks',
        label_file='data/train/labels.csv',
        transform=transform,
        preprocessor=preprocessor
    )
    
    val_dataset = ThyroidDataset(
        image_dir='data/val/images',
        mask_dir='data/val/masks',
        label_file='data/val/labels.csv',
        transform=transform,
        preprocessor=preprocessor
    )
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    # 初始化模型
    model = MultiTaskUNetPP().to(device)
    
    # 训练原始模型
    print("开始训练原始模型...")
    model = train_model(model, train_loader, val_loader, device, epochs=30)
    
    # 知识蒸馏
    print("\n开始知识蒸馏...")
    student_model = MultiTaskUNetPP().to(device)  # 更轻量的学生模型
    distilled_model = knowledge_distillation(model, student_model, train_loader, device)
    
    # 量化感知训练
    print("\n开始量化感知训练...")
    distilled_model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    quant_model = prepare_qat(distilled_model.train())
    
    # 需要额外的训练步骤
    quant_model = train_model(quant_model, train_loader, val_loader, device, epochs=10)
    quant_model = convert(quant_model.eval())
    
    # 结构化剪枝
    print("\n应用结构化剪枝...")
    pruned_model = prune_model(quant_model, amount=0.4)
    
    # 保存最终模型
    torch.save(pruned_model.state_dict(), 'final_model.pth')
    
    # 导出到ONNX
    print("\n导出ONNX模型...")
    export_to_onnx(pruned_model)
    
    # 构建TensorRT引擎（通常在Jetson上执行）
    print("\n构建TensorRT引擎...")
    build_tensorrt_engine("thyroid_model.onnx", "thyroid_model.trt")
    
    print("所有流程完成！")

if __name__ == "__main__":
    main()
