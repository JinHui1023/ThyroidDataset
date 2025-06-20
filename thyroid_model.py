import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from sklearn.metrics import roc_auc_score

# 模拟数据加载器
class ThyroidDataset(Dataset):
    def __init__(self, num_samples=100, img_size=(256, 256)):
        self.num_samples = num_samples
        self.img_size = img_size
        # 生成随机模拟数据
        self.images = np.random.rand(num_samples, 1, *img_size).astype(np.float32)
        self.masks = (np.random.rand(num_samples, 1, *img_size) > 0.7).astype(np.float32)
        self.labels = (np.random.rand(num_samples) > 0.8).astype(np.int64)  # 80%良性，20%恶性

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img = self.images[idx]
        mask = self.masks[idx]
        label = self.labels[idx]
        return torch.tensor(img), torch.tensor(mask), torch.tensor(label)

# 简化版UNet++模型
class MultiTaskUNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 编码器
        self.enc1 = self._block(1, 64)
        self.enc2 = self._block(64, 128)
        self.pool = nn.MaxPool2d(2)
        
        # 瓶颈层
        self.bottleneck = self._block(128, 256)
        
        # 分割解码器
        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec1 = self._block(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = self._block(128, 64)
        self.seg_out = nn.Conv2d(64, 1, 1)
        
        # 分类分支
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, 2)

    def _block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )

    def forward(self, x):
        # 编码
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        
        # 瓶颈
        b = self.bottleneck(self.pool(e2))
        
        # 分割解码
        d1 = self.up1(b)
        d1 = torch.cat([d1, e2], dim=1)
        d1 = self.dec1(d1)
        d2 = self.up2(d1)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2(d2)
        seg_mask = torch.sigmoid(self.seg_out(d2))
        
        # 分类
        cls_feat = self.avgpool(b).view(b.size(0), -1)
        cls_logits = self.fc(cls_feat)
        return seg_mask, cls_logits

# 知识蒸馏
def knowledge_distillation(teacher, student, dataloader, device, epochs=5):
    teacher.eval()
    student.train()
    optimizer = optim.Adam(student.parameters(), lr=0.001)
    cls_loss = nn.CrossEntropyLoss()
    kd_loss = nn.KLDivLoss()
    
    for epoch in range(epochs):
        for imgs, _, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            
            with torch.no_grad():
                _, t_logits = teacher(imgs)
            
            _, s_logits = student(imgs)
            
            # 计算知识蒸馏损失
            T = 5.0  # 温度参数
            soft_teacher = F.softmax(t_logits/T, dim=1)
            soft_student = F.log_softmax(s_logits/T, dim=1)
            loss_kd = kd_loss(soft_student, soft_teacher) * (T*T)
            
            # 计算学生分类损失
            loss_cls = cls_loss(s_logits, labels)
            
            # 总损失
            loss = 0.7*loss_kd + 0.3*loss_cls
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"蒸馏 Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# 量化感知训练
def quantize_model(model):
    model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    quantized_model = torch.quantization.prepare_qat(model.train())
    # 实际中这里需要额外的训练步骤
    quantized_model = torch.quantization.convert(quantized_model.eval())
    return quantized_model

# 主训练函数
def train_model():
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建数据集
    dataset = ThyroidDataset(num_samples=100)
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # 初始化模型
    model = MultiTaskUNet().to(device)
    
    # 损失函数
    def seg_loss(pred, target):
        bce = F.binary_cross_entropy(pred, target)
        dice = 1 - (2*torch.sum(pred*target) + 1e-8) / (torch.sum(pred) + torch.sum(target) + 1e-8)
        return bce + dice
    
    cls_loss = nn.CrossEntropyLoss()
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练循环
    for epoch in range(5):
        model.train()
        total_loss = 0
        
        for i, (imgs, masks, labels) in enumerate(train_loader):
            imgs, masks, labels = imgs.to(device), masks.to(device), labels.to(device)
            
            optimizer.zero_grad()
            seg_pred, cls_pred = model(imgs)
            
            # 计算多任务损失
            loss_seg = seg_loss(seg_pred, masks)
            loss_cls = cls_loss(cls_pred, labels)
            loss = loss_seg + loss_cls
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            if i % 5 == 0:
                print(f"Epoch {epoch+1}, Batch {i}, Loss: {loss.item():.4f}")
        
        # 验证指标
        model.eval()
        with torch.no_grad():
            _, cls_pred = model(imgs)
            auc = roc_auc_score(labels.cpu().numpy(), 
                               F.softmax(cls_pred, dim=1)[:,1].cpu().numpy())
            print(f"Epoch {epoch+1}完成, 平均损失: {total_loss/len(train_loader):.4f}, AUC: {auc:.4f}")
    
    # 保存原始模型
    torch.save(model.state_dict(), 'thyroid_model.pth')
    print("原始模型训练完成并保存")
    
    # 知识蒸馏
    print("\n开始知识蒸馏...")
    teacher = MultiTaskUNet().to(device).eval()
    teacher.load_state_dict(torch.load('thyroid_model.pth'))
    student = MultiTaskUNet().to(device)
    knowledge_distillation(teacher, student, train_loader, device)
    
    # 量化
    print("\n开始量化模型...")
    quantized_student = quantize_model(student)
    torch.save(quantized_student.state_dict(), 'compressed_model.pth')
    print("压缩模型保存完成")
    
    # ONNX导出
    print("\n导出ONNX模型...")
    dummy_input = torch.randn(1, 1, 256, 256).to(device)
    torch.onnx.export(quantized_student, 
                     dummy_input, 
                     "thyroid_model.onnx", 
                     input_names=["input"],
                     output_names=["seg_mask", "cls_logits"])
    print("ONNX模型导出成功")

if __name__ == "__main__":
    train_model()
