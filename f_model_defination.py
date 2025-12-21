import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionGate(nn.Module):
    """
    Attention Gate (AG) for spatial attention in U-Net skip connections.
    Computes attention coefficients for the skip connection features.
    Adapted from Attention U-Net paper (Oktay et al., 2018).
    """
    def __init__(self, F_g, F_l, F_int):  # F_g: gating signal channels (from up), F_l: skip channels, F_int: intermediate channels
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.GELU()   # 修正：使用 GELU 替代 ReLU，不使用 inplace=True

    def forward(self, g, x):  # g: gating signal (low-res from up), x: skip (high-res)
        g1 = self.W_g(g)  # Downsample g to match x size? No, we upsample g via conv, but actually resize g to x's spatial size
        # Resize g to match x's spatial dimensions (upsample if needed)
        if g1.shape[2:] != x.shape[2:]:
            g1 = F.interpolate(g1, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)  # Additive attention
        psi = self.psi(psi)
        # Resize attention map back if needed, but since we apply to x, it's already matching
        return x * psi  # Element-wise multiplication for attended skip

class DoubleConv(nn.Module):
    """双卷积块：不变"""
    def __init__(self, in_channels, out_channels, dropout_p=0.2):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),     # 修正：使用 GELU 替代 ReLU，不使用 inplace=True
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),     # 修正：使用 GELU 替代 ReLU，不使用 inplace=True
            nn.Dropout2d(p=dropout_p)  # 2D Dropout，针对空间场
        )

    def forward(self, x):
        return self.conv(x)
    

class ResidualBlock(nn.Module):
    """
    Residual Block = DoubleConv + identity shortcut
    - GELU (non-inplace)
    - Dropout2d
    - 当 stride 或通道不匹配时使用 1×1 投影
    """
    def __init__(self, in_channels, out_channels, stride=1, dropout_p=0.2):
        super().__init__()
        self.main_path = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),

            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Dropout2d(p=dropout_p)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.final_gelu = nn.GELU()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.main_path(x)
        out = out + identity
        out = self.final_gelu(out)
        return out

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p=0.2):
        super().__init__()
        self.res_block = ResidualBlock(in_channels, out_channels, stride=1, dropout_p=dropout_p)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        skip = self.res_block(x)      # 高分辨率 skip
        pooled = self.pool(skip)      # 低分辨率传给下一层
        return pooled, skip

class Up(nn.Module):
    """Up: 上采样 → Attention → cat → ResidualBlock → Dropout"""
    def __init__(self, in_channels, out_channels, dropout_p=0.1):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.attention = AttentionGate(F_g=out_channels, F_l=out_channels, F_int=out_channels // 2)
        self.res_block = ResidualBlock(out_channels * 2, out_channels, dropout_p=dropout_p)
        self.dropout = nn.Dropout2d(p=dropout_p)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # 精确尺寸对齐
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)

        if diffY != 0 or diffX != 0:
            pad_top = max(0, diffY // 2)
            pad_bottom = max(0, diffY - diffY // 2)
            pad_left = max(0, diffX // 2)
            pad_right = max(0, diffX - diffX // 2)
            if diffY < 0 or diffX < 0:
                # crop x1
                h_start = max(0, -diffY // 2)
                w_start = max(0, -diffX // 2)
                x1 = x1[:, :, h_start:h_start + x2.size(2), w_start:w_start + x2.size(3)]
            else:
                x1 = F.pad(x1, (pad_left, pad_right, pad_top, pad_bottom))

        x2_att = self.attention(x1, x2)
        x = torch.cat([x2_att, x1], dim=1)
        x = self.res_block(x)
        return self.dropout(x)

class UNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, features=[16, 32, 64, 128], dropout_p=0.2, env_type=[], is_residual=False):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.env_type = env_type
        self.is_residual = is_residual
        # print(f'UNet env_type: {self.env_type}')

        in_ch = in_channels
        for f in features:
            self.downs.append(Down(in_ch, f, dropout_p=dropout_p))
            in_ch = f

        # 瓶颈使用 ResidualBlock
        self.bottleneck = ResidualBlock(features[-1], features[-1] * 2, dropout_p=dropout_p)

        # 上采样路径
        for f in reversed(features):
            self.ups.append(Up(f * 2, f, dropout_p=dropout_p * 0.5))  # 上采样 dropout 可减半

        self.final_conv = nn.Conv2d(features[0], out_channels, 1)
        self.final_norm = nn.BatchNorm2d(out_channels)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            # GELU 初始化使用 ReLU 的 Kaiming（官方推荐）
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x, skip = down(x)
            skip_connections.append(skip)

        x = self.bottleneck(x)

        skip_connections = skip_connections[::-1]
        for up, skip in zip(self.ups, skip_connections):
            x = up(x, skip)

        x = self.final_conv(x)
        x = self.final_norm(x)

        # 是否属于残差训练阶段
        if self.is_residual:
            return x  # 残差阶段不激活

        if self.env_type == 'wave':
            return torch.sigmoid(x)  # 或 return x （若后续有归一化）
        else:
            return torch.tanh(x)



# 在U-Net模型定义后，训练循环前
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch.optim as optim
from torch.amp import autocast, GradScaler
from tqdm import tqdm

scaler = GradScaler("cuda")  # 混合精度缩放器

# 训练函数
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    num_batches = 0
    for batch_idx, (data, target) in enumerate(tqdm(loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        with autocast("cuda"):  # 关键！
            output = model(data)
            # 训练循环中，加权大值区域
            loss = criterion(output.float(), target.float())

        # NaN 监控
        if torch.isnan(loss):
            print(f"Warning: NaN loss at batch {batch_idx}! Check gradients/data.")
            return float('inf')
        
        # 在 scaler step 之前加入梯度裁剪 (Unscale first)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # 限制梯度最大范数
        
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        num_batches += 1
        
    return total_loss / num_batches if num_batches > 0 else float('inf')

# 验证函数
def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    num_batches = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # 训练循环中，加权大值区域
            loss = criterion(output, target)
            
            if torch.isnan(loss):
                print("Warning: NaN val loss!")
                return float('inf')
            total_loss += loss.item()
            num_batches += 1
    return total_loss / num_batches if num_batches > 0 else float('inf')

class UNetBoosting(nn.Module):
    def __init__(self, num_models=5, in_channels=2, out_channels=1, learning_rate=1e-3, shrinkage=0.1, features=[16, 32, 64, 128], device=[], env_type=[]):
        super(UNetBoosting, self).__init__()
        self.num_models = num_models
        self.shrinkage = shrinkage  # 收缩因子，控制后续模型贡献
        self.models = nn.ModuleList()  # 存储多个U-Net
        self.optimizers = []  # 每个U-Net独立优化器
        self.env_type = env_type
        self.device = device
        for i in range(num_models):
            is_res = (i > 0)  # 第一个模型非残差，后续为残差
            model = UNet(in_channels=in_channels, out_channels=out_channels, features=features, env_type=self.env_type, is_residual=is_res).to(self.device)
            # print(f'Env type for model {i}: {self.env_type}')
            model.apply(model.init_weights)  # 初始化
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, eps=1e-4)  # 后续模型lr不缩小
            self.models.append(model)
            self.optimizers.append(optimizer)
    
    def forward(self, x):
        # 预测：sum所有U-Net输出（第一个完整，后续乘shrinkage）
        pred = self.models[0](x)
        for model in self.models[1:]:
            pred += self.shrinkage * model(x)
            # pred += model(x)
        return pred
    
    def train_ensemble(self, train_loader, val_loader, criterion_1, scheduler_func, num_epochs=100, patience=7, device='cuda'):
        train_losses = [[] for _ in range(self.num_models)]
        val_losses = [[] for _ in range(self.num_models)]

        # === 1. 提取原始数据（只做一次）===
        train_subset = train_loader.dataset
        val_subset = val_loader.dataset
        full_dataset = train_subset.dataset  # 原始 NumpyDataset

        train_indices = train_subset.indices
        val_indices = val_subset.indices

        # 原始 numpy → Tensor（只转一次）
        wind_tensor = torch.from_numpy(full_dataset.wind_data.astype(np.float32)).to(device)
        # 根据维度判断是否需要 unsqueeze
        output_data_np = full_dataset.output_data.astype(np.float32)
        if output_data_np.ndim == 3: # 如果是 (N, H, W) -> (N, 1, H, W)
            output_tensor = torch.from_numpy(output_data_np).unsqueeze(1).to(device)
        else: # 如果已经是 (N, C, H, W) -> 保持原样
            output_tensor = torch.from_numpy(output_data_np).to(device)

        train_wind = wind_tensor[train_indices]
        val_wind = wind_tensor[val_indices]
        train_targets = output_tensor[train_indices]
        val_targets = output_tensor[val_indices]

        # === 2. 残差训练循环 ===
        current_train_res = train_targets.clone()
        current_val_res = val_targets.clone()

        for i in range(self.num_models):
            print(f"\nTraining U-Net {i+1}/{self.num_models}")
            counter = 0
            best_val_loss = float('inf')    # 源代码在循环外重置

            # 训练损失函数
            if i == 0:
                criterion = criterion_1  # 第一个模型使用指定损失函数
            else:
                print(f'env type: {self.env_type}')
                if self.env_type == 'current':
                    criterion = nn.MSELoss()  # 后续模型使用 MSE
                elif self.env_type == 'surge':
                    criterion = nn.L1Loss()  # 后续模型使用 MSE
                elif self.env_type == 'wave':
                    criterion = nn.L1Loss()   # 后续模型使用 MAE

            model = self.models[i]
            optimizer = self.optimizers[i]
            scheduler = scheduler_func(optimizer)

            # 使用当前残差创建临时 DataLoader（不修改原始 train_loader）
            if i == 0:
                train_res_dataset = TensorDataset(train_wind, current_train_res)
                val_res_dataset = TensorDataset(val_wind, current_val_res)
            else:
                train_res_dataset = TensorDataset(train_wind, current_train_res / self.shrinkage)
                val_res_dataset = TensorDataset(val_wind, current_val_res / self.shrinkage)
            '''train_res_dataset = TensorDataset(train_wind, current_train_res)
            val_res_dataset = TensorDataset(val_wind, current_val_res)'''

            train_loader_res = DataLoader(train_res_dataset, batch_size=train_loader.batch_size, shuffle=True)
            val_loader_res = DataLoader(val_res_dataset, batch_size=val_loader.batch_size, shuffle=False)

            # 训练当前 U-Net
            for epoch in range(num_epochs):
                train_loss = train_epoch(model, train_loader_res, criterion, optimizer, device)
                # 如果遇到 NaN，直接跳过当前模型的剩余训练，尝试下一个模型（或者直接终止）
                if train_loss == float('inf'):
                    print(f"Skipping Model {i+1} due to NaN.")
                    break
                val_loss = val_epoch(model, val_loader_res, criterion, device)
                scheduler.step(val_loss)

                if val_loss < best_val_loss - 5e-5:  # 小阈值防止频繁保存
                    best_val_loss = val_loss
                    counter = 0
                    torch.save(self.state_dict(), f'best_boosting_model_{i}.pth')
                else:
                    counter += 1
                if counter >= patience:
                    print("Early stopping.")
                    break

                print(f'U-Net {i+1} Epoch {epoch+1}: Train {train_loss:.4f}, Val {val_loss:.4f}')

                train_losses[i].append(train_loss)
                val_losses[i].append(val_loss)

            # === 3. 更新残差（为下一个 U-Net 准备）===
            if i < self.num_models:
                pred_train = self._predict_single_model(model, train_wind, train_loader.batch_size, device)
                pred_val = self._predict_single_model(model, val_wind, val_loader.batch_size, device)
                if i > 0:
                    pred_train = pred_train * self.shrinkage
                    pred_val = pred_val * self.shrinkage

                current_train_res = current_train_res - pred_train
                current_val_res = current_val_res - pred_val

        return train_losses, val_losses
    
    def _predict_single_model(self, model, wind_tensor, batch_size, device):
        model.eval()
        preds = []
        total = len(wind_tensor)
        with torch.no_grad():
            for i in range(0, total, batch_size):
                # 直接切片，不产生额外的大型数据拷贝
                batch_x = wind_tensor[i : i + batch_size].to(device)
                preds.append(model(batch_x))
        return torch.cat(preds, dim=0)
    

# 二叉树结构 U-Net 的定义
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast
from tqdm import tqdm
import numpy as np

# --- 保持原有的基础模块 (AttentionGate, DoubleConv, ResidualBlock, Down, Up, UNet) 不变 ---

# === 新增：门控模块 (Gate Block) ===
class GateBlock(nn.Module):
    """
    轻量级卷积网络，生成 0-1 之间的概率图 (Soft Routing)。
    用于决定输入应该主要由左子树处理还是右子树处理。
    """
    def __init__(self, in_channels):
        super(GateBlock, self).__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 1, kernel_size=1), # 输出单通道概率图
            nn.Sigmoid() # 关键：输出范围限制在 [0, 1]
        )

    def forward(self, x):
        return self.gate(x)

# === 修改：二叉树结构 U-Net (Tree Structured U-Net) ===
class TreeUNet(nn.Module):
    def __init__(self, depth, in_channels=2, out_channels=1, features=[16, 32, 64, 128], env_type=[], device='cuda'):
        """
        depth: 树的深度。depth=0 表示叶子节点（也就是一个 UNet）。
               depth=2 意味着根节点下有左右两个子树，子树下又是叶子，共 4 个 UNet。
        """
        super(TreeUNet, self).__init__()
        self.depth = depth
        self.device = device
        self.env_type = env_type
        
        # 如果深度为0，则是叶子节点（执行预测的 U-Net）
        if self.depth == 0:
            self.is_leaf = True
            self.model = UNet(in_channels=in_channels, out_channels=out_channels, 
                              features=features, env_type=env_type, is_residual=False).to(device)
            self.model.apply(self.model.init_weights)
        else:
            # 否则是内部节点（包含门控和左右子树）
            self.is_leaf = False
            self.gate = GateBlock(in_channels).to(device)
            
            # 递归创建左右子树
            self.left_child = TreeUNet(depth - 1, in_channels, out_channels, features, env_type, device)
            self.right_child = TreeUNet(depth - 1, in_channels, out_channels, features, env_type, device)

    def forward(self, x):
        if self.is_leaf:
            return self.model(x)
        else:
            # 计算路由概率图 (batch, 1, H, W)
            prob_map = self.gate(x)
            
            # 递归获取左右子树的输出
            left_out = self.left_child(x)
            right_out = self.right_child(x)
            
            # 软路由聚合： Prob * Left + (1 - Prob) * Right
            # 这种方式允许梯度在全树传播
            return prob_map * left_out + (1 - prob_map) * right_out

# === 训练逻辑适配 ===
# 树形结构通常不再适合使用“残差逼近”的线性Boosting训练方式。
# 它是端到端 (End-to-End) 的，所有专家网络和门控网络一起训练效果最好。

class TreeUNetManager:
    """
    用于管理 TreeUNet 的训练和预测，替代原来的 UNetBoosting
    """
    def __init__(self, tree_depth=2, in_channels=2, out_channels=1, learning_rate=1e-3, 
                 features=[16, 32, 64, 128], criterion = nn.MSELoss(),
                 device='cuda', env_type=[]):
        self.device = device
        self.env_type = env_type
        # depth=2 等同于 2^2 = 4 个 U-Net 叶子节点
        self.tree_model = TreeUNet(depth=tree_depth, in_channels=in_channels, out_channels=out_channels, 
                                   features=features, env_type=env_type, device=device).to(device)
        
        # 优化器优化整棵树的所有参数
        self.optimizer = optim.Adam(self.tree_model.parameters(), lr=learning_rate, eps=1e-4)
        
        # 定义损失函数
        self.criterion = criterion

    def __call__(self, x):
        return self.tree_model(x)

    def train_tree(self, train_loader, val_loader, scheduler_func, num_epochs=100, patience=7):
        """
        端到端训练整棵树
        """
        scheduler = scheduler_func(self.optimizer)
        best_val_loss = float('inf')
        counter = 0
        train_losses = []
        val_losses = []

        print(f"\nTraining Tree Structured U-Net (Depth={self.tree_model.depth})")
        
        for epoch in range(num_epochs):
            # 1. 训练一个 Epoch
            self.tree_model.train()
            total_loss = 0
            num_batches = 0
            
            for data, target in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()

                with autocast("cuda"):
                    output = self.tree_model(data)
                    loss = self.criterion(output, target)

                if torch.isnan(loss):
                    print("Error: NaN loss detected!")
                    return train_losses, val_losses
                
                # 反向传播
                scaler.scale(loss).backward()
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.tree_model.parameters(), max_norm=1.0)
                scaler.step(self.optimizer)
                scaler.update()

                total_loss += loss.item()
                num_batches += 1
            
            avg_train_loss = total_loss / num_batches
            
            # 2. 验证
            avg_val_loss = val_epoch(self.tree_model, val_loader, self.criterion, self.device)
            scheduler.step(avg_val_loss)

            print(f'Epoch {epoch+1}: Train Loss {avg_train_loss:.5f}, Val Loss {avg_val_loss:.5f}')
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)

            # 3. 早停策略
            if avg_val_loss < best_val_loss - 1e-5:
                best_val_loss = avg_val_loss
                counter = 0
                torch.save(self.tree_model.state_dict(), 'best_tree_unet_model.pth')
            else:
                counter += 1
                if counter >= patience:
                    print("Early stopping triggered.")
                    break
        
        return train_losses, val_losses

    def predict(self, x_tensor, batch_size=32):
        """
        批量预测辅助函数
        """
        self.tree_model.eval()
        preds = []
        total = len(x_tensor)
        with torch.no_grad():
            for i in range(0, total, batch_size):
                batch_x = x_tensor[i : i + batch_size].to(self.device)
                preds.append(self.tree_model(batch_x))
        return torch.cat(preds, dim=0)
    


# ==========================================
#  XGBoost-like Gradient Boosting U-Net
# ==========================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import gc  # 引入垃圾回收模块

class XGBoostUNet(nn.Module):
    def __init__(self, num_models=5, in_channels=2, out_channels=1, 
                 features=[16, 32, 64, 128], 
                 learning_rate=1e-3, 
                 shrinkage=0.1, 
                 subsample=0.8, 
                 reg_lambda=1e-4, 
                 device='cuda', 
                 env_type='current'):
        super(XGBoostUNet, self).__init__()
        self.num_models = num_models
        self.shrinkage = shrinkage
        self.subsample = subsample
        self.reg_lambda = reg_lambda
        self.learning_rate = learning_rate # 保存 LR 供后续使用
        self.features = features
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.env_type = env_type
        self.device = device
        
        self.models = nn.ModuleList()
        # 注意：这里不再预先创建 self.optimizers，因为优化器状态非常占显存
        
        # 初始化所有弱学习器 (Weak Learners)
        for i in range(num_models):
            is_res = (i > 0)
            # 初始化时默认放在 CPU
            model = UNet(in_channels=in_channels, out_channels=out_channels, 
                         features=features, env_type=self.env_type, is_residual=is_res)
            model.apply(model.init_weights)
            self.models.append(model)

    def forward(self, x):
        # 推理逻辑：Base + eta * (Model1 + Model2 + ...)
        # 自动获取输入数据所在的设备 (cuda 或 cpu)
        current_device = x.device 
        
        # 处理第一个模型
        model0 = self.models[0].to(current_device)
        pred = model0(x)
        # 如果不是在训练阶段，且为了极端节省显存，可以考虑 model0.to('cpu')
        # 但如果是频繁预测，可以保持在 GPU，或者只在 batch 结束后处理
        
        for i in range(1, self.num_models):
            model_i = self.models[i].to(current_device)
            pred += self.shrinkage * model_i(x)
            # 为了防止推理时显存溢出，计算完一个就放回 CPU
            model_i.to('cpu') 
            
        return pred
    
    def compute_grad_hess(self, pred, target):
        """
        计算损失函数的一阶导 (Gradient) 和二阶导 (Hessian)。
        XGBoost 的核心是拟合 -g/h。
        """
        if self.env_type == 'current' or self.env_type == 'default':
            # MSE Loss: L = 0.5 * (y - p)^2
            # Grad = p - y
            # Hess = 1
            grad = pred - target
            hess = torch.ones_like(grad)
            
        elif self.env_type == 'wave' or self.env_type == 'surge':
            # 为了数值稳定性，使用 Pseudo-Huber Loss 近似 L1
            # L = delta^2 * (sqrt(1 + ((y-p)/delta)^2) - 1)
            delta = 1.0 
            diff = pred - target
            scale = torch.sqrt(1 + (diff / delta) ** 2)
            
            grad = diff / scale
            hess = (1 / scale) - (diff / scale)**2 / scale # 近似二阶导
            
        else:
            # 默认回退到 MSE
            grad = pred - target
            hess = torch.ones_like(grad)
            
        return grad, hess

    def train_ensemble(self, train_loader, val_loader, criterion_base, scheduler_func, num_epochs=100, patience=7):
        train_losses = [[] for _ in range(self.num_models)]
        val_losses = [[] for _ in range(self.num_models)]

        # 1. 准备全量数据 Tensor (用于计算全局残差)
        full_train_dataset = train_loader.dataset
        full_val_dataset = val_loader.dataset
        
        # 假设 dataset 是 TensorDataset，直接获取；如果是自定义 Dataset，需修改此处获取逻辑
        # 这里沿用原代码逻辑，提取 data
        full_source_data = full_train_dataset.dataset # 原始 NumpyDataset
        
        # 提取训练集索引对应的数据
        X_train = torch.from_numpy(full_source_data.wind_data[full_train_dataset.indices].astype(np.float32)).to(self.device)
        y_train_np = full_source_data.output_data[full_train_dataset.indices].astype(np.float32)
        if y_train_np.ndim == 3: y_train_np = np.expand_dims(y_train_np, 1)
        y_train = torch.from_numpy(y_train_np).to(self.device)

        X_val = torch.from_numpy(full_source_data.wind_data[full_val_dataset.indices].astype(np.float32)).to(self.device)
        y_val_np = full_source_data.output_data[full_val_dataset.indices].astype(np.float32)
        if y_val_np.ndim == 3: y_val_np = np.expand_dims(y_val_np, 1)
        y_val = torch.from_numpy(y_val_np).to(self.device)

        # 当前的累积预测 (Current Prediction)
        pred_train_cumulative = torch.zeros_like(y_train)
        pred_val_cumulative = torch.zeros_like(y_val)

        # === 逐级训练 (Boosting Loop) ===
        for i in range(self.num_models):
            print(f"\n[XGBoost-UNet] Training Booster {i+1}/{self.num_models}")
            model = self.models[i].to(self.device)
            optimizer = optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=self.reg_lambda)
            scheduler = scheduler_func(optimizer)
            
            # --- 步骤 A: 准备当前阶段的目标 (Target) ---
            if i == 0:
                # 第一个模型直接拟合原始标签
                train_target_step = y_train
                val_target_step = y_val
                # 第一个模型通常不需要复杂的 loss gradient，直接用 MSE 或 MAE 预热
                criterion = criterion_base 
            else:
                # 后续模型拟合“负梯度” (Negative Gradient)
                # XGBoost update: new_model ~= -grad / (hess + lambda)
                # 在深度学习中，我们通常让网络输出去拟合这个目标
                
                with torch.no_grad():
                    grad, hess = self.compute_grad_hess(pred_train_cumulative, y_train)
                    # Newton Step: target = -grad / hess
                    # 注意：如果 hess 很小，除法不稳定，所以 MSE (hess=1) 时其实就是 residual
                    train_target_step = -grad / (hess + 1e-5) 
                    
                    # 验证集残差 (仅用于监控)
                    grad_val, hess_val = self.compute_grad_hess(pred_val_cumulative, y_val)
                    val_target_step = -grad_val / (hess_val + 1e-5)

                    # 【显存优化】：计算完 target 后，grad 和 hess 就不需要了
                    del grad, hess, grad_val, hess_val

                # 弱学习器使用 MSE Loss 来逼近这个计算出的“最优更新步长”
                criterion = nn.MSELoss()

            # --- 步骤 B: 随机采样 (Subsampling) ---
            # 创建本次训练的数据加载器
            if self.subsample < 1.0 and i > 0: # Base model 建议用全量数据
                num_samples = len(X_train)
                indices = torch.randperm(num_samples)[:int(num_samples * self.subsample)]
                train_sub_X = X_train[indices]
                train_sub_y = train_target_step[indices]
            else:
                train_sub_X = X_train
                train_sub_y = train_target_step

            # 封装为 Loader
            step_train_ds = TensorDataset(train_sub_X, train_sub_y)
            step_val_ds = TensorDataset(X_val, val_target_step) # 验证集不采样
            
            step_train_loader = DataLoader(step_train_ds, batch_size=train_loader.batch_size, shuffle=True)
            step_val_loader = DataLoader(step_val_ds, batch_size=val_loader.batch_size, shuffle=False)

            # --- 步骤 C: 训练单个 U-Net ---
            best_loss = float('inf')
            counter = 0
            
            for epoch in range(num_epochs):
                # 调用你现有的 train_epoch 函数
                # 注意：这里的 optimizer 已经包含了 L2 正则 (lambda)
                loss_train = train_epoch(model, step_train_loader, criterion, optimizer, self.device)
                loss_val = val_epoch(model, step_val_loader, criterion, self.device)
                
                scheduler.step(loss_val)
                train_losses[i].append(loss_train)
                val_losses[i].append(loss_val)
                
                print(f"  Booster {i+1} Epoch {epoch+1}: Train {loss_train:.5f}, Val {loss_val:.5f}")

                if loss_val < best_loss - 1e-5:
                    best_loss = loss_val
                    counter = 0
                    torch.save(model.state_dict(), f'xgboost_unet_booster_{i}.pth')
                else:
                    counter += 1
                    if counter >= patience:
                        print(f"  Booster {i+1} Early stopping.")
                        model.load_state_dict(torch.load(f'xgboost_unet_booster_{i}.pth'))
                        break
            
            # --- 步骤 D: 更新全局预测 (Update Accumulation) ---
            # 训练完后，使用该模型对全量数据进行预测，并更新累计预测值
            model.eval()
            with torch.no_grad():
                # 批量预测以节省显存
                new_pred_train = self._batch_predict(model, X_train, train_loader.batch_size)
                new_pred_val = self._batch_predict(model, X_val, val_loader.batch_size)
                
                if i == 0:
                    pred_train_cumulative = new_pred_train
                    pred_val_cumulative = new_pred_val
                else:
                    pred_train_cumulative += self.shrinkage * new_pred_train
                    pred_val_cumulative += self.shrinkage * new_pred_val

            self.models[i].to('cpu') 
            model.to('cpu') # 确保变量引用也指回 CPU
            # B. 删除占用显存的临时变量
            del optimizer        # 优化器状态（Momentum等）非常大，必须删
            del scheduler
            del step_train_loader
            del step_val_loader
            del train_target_step
            del val_target_step
            del new_pred_train
            del new_pred_val
            
            # C. 强制执行垃圾回收和显存清空
            gc.collect()                  # 清除 Python 层的无用引用
            torch.cuda.empty_cache()      # 释放 PyTorch 缓存分配器中的显存
            
            print(f"  Booster {i+1} finished. GPU cache cleared.")

        return train_losses, val_losses

    def _batch_predict(self, model, x, batch_size):
        preds = []
        for i in range(0, len(x), batch_size):
            batch = x[i:i+batch_size]
            preds.append(model(batch))
        return torch.cat(preds, dim=0)