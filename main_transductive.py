# =================================================================
#               请用以下全部内容覆盖您的 main_transductive.py
# =================================================================

import logging
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from itertools import chain
import dgl
from sklearn.model_selection import StratifiedKFold, train_test_split
import copy
from graphmae.models.gat import GAT
# 假设 fusion_model.py 与 main_transductive.py 在同一目录下
from fusion_model import FusionEncoder

# 从原始代码库中导入必要的工具函数和模型构建器
from graphmae.utils import build_args, set_random_seed, accuracy, create_norm
from graphmae.models import build_model

# ==================== VAE模型定义 ====================
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU())
        self.fc_mu, self.fc_log_var = nn.Linear(hidden_dim // 2, latent_dim), nn.Linear(hidden_dim // 2, latent_dim)
        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, input_dim), nn.Sigmoid())
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    def forward(self, x):
        h = self.encoder(x)
        mu, log_var = self.fc_mu(h), self.fc_log_var(h)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var

def loss_function_vae(recon_x, x, mu, log_var):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD

# ==================== 统一的融合模型封装 ====================
class SupervisedFusionModel(nn.Module):
    """
    一个封装了FusionEncoder和下游GNN的统一模型，方便在训练循环中管理。
    """
    def __init__(self, args, num_classes):
        super(SupervisedFusionModel, self).__init__()

    # 分支A和B的融合编码器
        self.fusion_encoder = FusionEncoder(
            h5_in_dim=args.h5_feature_dim,
            sc_in_dim=args.vae_latent_dim,
            hidden_dim=args.fusion_hidden_dim,
            fused_out_dim=args.fusion_out_dim
        )

    # ------------------- 关键修正代码 -------------------
    # 1. 调用辅助函数，将字符串norm名称转换为真正的层类
        norm_layer = create_norm(args.norm)

    # 2. 直接根据GAT类的实际定义来创建实例
        self.gnn = GAT(
            in_dim=args.fusion_out_dim,
            num_hidden=args.num_hidden,
            out_dim=num_classes,
            num_layers=args.num_layers,
            nhead=args.num_heads,
            nhead_out=args.num_out_heads,
            activation=args.activation,
            feat_drop=args.in_drop,
            attn_drop=args.attn_drop,
            negative_slope=args.negative_slope,
            residual=args.residual,
            norm=norm_layer,
            concat_out=True # <--- 修改为 True
        )
    # ----------------------------------------------------
    def forward(self, graph, h5_features, sc_features):
        fused_features = self.fusion_encoder(h5_features, sc_features)

        # --- 关键修正 ---
        # 调用GNN并处理其可能的元组输出
        gnn_output = self.gnn(graph, fused_features)

        # 检查GNN的返回类型，我们只取第一个元素（即最终的预测结果）
        if isinstance(gnn_output, tuple):
            output = gnn_output[0]
        else:
            output = gnn_output

        return output

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

# ==================== 全新的 main 函数 ====================
def main(args):
    device = args.device if args.device >= 0 else "cpu"
    set_random_seed(args.seeds[0] if args.seeds else 42)

    # ---- 1. Data Loading and Feature Engineering (Once before the loop) ----
    logging.info(f"--> Loading pre-processed data from cache: {args.cache_path}")
    cached_data = torch.load(args.cache_path, weights_only=False)
    
    # 移动部分数据到设备
    h5_features = cached_data['features'].to(device)
    labels = cached_data['y_train'].squeeze().long().to(device) # 确保为long类型
    network_numpy = cached_data['network'].cpu().numpy()
    sc_expression = cached_data['sc_expression'] # VAE在CPU上训练可能更快

    # ---- 1.1 Build DGL Graph ----
    logging.info("--> Building DGL graph...")
    src, dst = np.nonzero(network_numpy)
    num_nodes = network_numpy.shape[0]
    graph = dgl.graph((src, dst), num_nodes=num_nodes)
    graph = dgl.add_self_loop(graph).to(device)
    
    # ---- 1.2 VAE Feature Engineering ----
    logging.info("--> Generating single-cell features using VAE...")
    sc_expression_transposed = sc_expression.T
    min_val, max_val = sc_expression_transposed.min(), sc_expression_transposed.max()
    sc_data_normalized = (sc_expression_transposed - min_val) / (max_val - min_val)
    
    INPUT_DIM = sc_data_normalized.shape[1]
    vae_model = VAE(input_dim=INPUT_DIM, hidden_dim=128, latent_dim=args.vae_latent_dim)
    vae_optimizer = optim.Adam(vae_model.parameters(), lr=1e-3)
    
    vae_model.train()
    print("Training VAE...")
    for _ in tqdm(range(20)): # 简化训练
        recon, mu, log_var = vae_model(sc_data_normalized)
        loss = loss_function_vae(recon, sc_data_normalized, mu, log_var)
        vae_optimizer.zero_grad()
        loss.backward()
        vae_optimizer.step()
    
    vae_model.eval()
    with torch.no_grad():
        h = vae_model.encoder(sc_data_normalized)
        sc_features = vae_model.fc_mu(h)
    sc_features = sc_features.to(device)
    logging.info(f"--> Single-cell features generated with shape: {sc_features.shape}")
    # 将H5特征维度存入args，方便模型初始化
    args.h5_feature_dim = h5_features.shape[1]


    # ---- 2. K-Fold Cross-Validation Setup ----
    # 找出所有带标签的节点用于交叉验证
    # 这里我们假设y_train中的非零项为正样本，0为负样本，且都在一个子集上
    # 您的原始脚本逻辑更复杂，这里我们使用一个通用的方法
    all_labeled_indices = torch.arange(num_nodes, device=device)
    y_all_labeled = labels[all_labeled_indices].cpu().numpy()

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    test_accs = []

    logging.info("--> Starting 5-Fold Cross-Validation...")
    for i, (train_val_idx, test_idx) in enumerate(skf.split(all_labeled_indices.cpu().numpy(), y_all_labeled)):
        logging.info(f"======== Fold {i+1}/5 ========")
        
        # 划分训练集和验证集
        train_idx, val_idx, _, _ = train_test_split(train_val_idx, y_all_labeled[train_val_idx], test_size=0.1, shuffle=True, stratify=y_all_labeled[train_val_idx], random_state=42)

        # 创建当前fold的mask
        train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True

        # ---- 2.1 Model and Optimizer Initialization for current fold ----
        num_classes = labels.max().item() + 1
        model = SupervisedFusionModel(args, num_classes).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        criterion = nn.CrossEntropyLoss()
        
        # ---- 2.2 End-to-End Training Loop for current fold ----
        best_val_acc = 0
        best_epoch = 0
        
        epoch_iter = tqdm(range(args.max_epoch))
        for epoch in epoch_iter:
            model.train()
            output = model(graph, h5_features, sc_features)
            if i == 0 and epoch == 0:
                print("\n" + "="*30 + " DEBUGGING INFO " + "="*30)
                print(f"--- Checking tensors right before loss calculation ---")
                print(f"Type of output:       {type(output)}")
                print(f"Type of train_mask:   {type(train_mask)}")
                print(f"Type of labels:       {type(labels)}")
                print("-" * 20)
                print(f"output.shape:         {output.shape}")
                print(f"train_mask.shape:     {train_mask.shape}")
                print(f"labels.shape:         {labels.shape}")
                print("-" * 20)
                print(f"output.dtype:         {output.dtype}")
                print(f"train_mask.dtype:     {train_mask.dtype}")
                print(f"labels.dtype:         {labels.dtype}")
                print("-" * 20)
                print(f"output.device:        {output.device}")
                print(f"train_mask.device:    {train_mask.device}")
                print(f"labels.device:        {labels.device}")
                print("-" * 20)
                print(f"Number of True values in train_mask: {train_mask.sum().item()}")
                print("="*70 + "\n")
    # ======================================================

    # 我们依然保留上次的 .bool() 修正，并用try-except包裹
            try:
                loss_train = criterion(output[train_mask.bool()], labels[train_mask.bool()])
            except TypeError as e:
                print("\n" + "!"*60)
                print("!!! CRITICAL ERROR: Indexing failed with TypeError. !!!")
                print(f"!!! Error message: {e}")
                print("!!! Please provide all the DEBUGGING INFO above to the assistant. !!!")
                print("!"*60)
                # 退出程序，以便我们分析调试信息
                exit()



            loss_train = criterion(output[train_mask.bool()], labels[train_mask.bool()])
            
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
            
            model.eval()
            with torch.no_grad():
                output = model(graph, h5_features, sc_features)
                loss_val = criterion(output[val_mask.bool()], labels[val_mask.bool()])
                preds_val = output[val_mask].max(1)[1]
                acc_val = preds_val.eq(labels[val_mask]).double().mean().item()

            if acc_val > best_val_acc:
                best_val_acc = acc_val
                best_epoch = epoch
                # 保存当前fold的最佳模型状态
                best_model_state = copy.deepcopy(model.state_dict())

            epoch_iter.set_description(f"Fold {i+1} | Epoch {epoch:03d} | Val Acc: {acc_val:.4f} (Best: {best_val_acc:.4f} at Epoch {best_epoch})")
            
        # ---- 2.3 Test on current fold using the best model ----
        logging.info(f"Fold {i+1} finished. Best validation accuracy: {best_val_acc:.4f}")
        model.load_state_dict(best_model_state)
        model.eval()
        with torch.no_grad():
            output = model(graph, h5_features, sc_features)
            loss_test = criterion(output[test_mask.bool()], labels[test_mask.bool()])
            preds_test = output[test_mask].max(1)[1]
            acc_test = preds_test.eq(labels[test_mask]).double().mean().item()
        
        logging.info(f"Fold {i+1} Test Accuracy: {acc_test:.4f}")
        test_accs.append(acc_test)

    # ---- 3. Final Report ----
    logging.info("\n" + "="*30 + " Cross-Validation Finished " + "="*30)
    mean_acc = np.mean(test_accs)
    std_acc = np.std(test_accs)
    logging.info(f"Average Test Accuracy across 5 folds: {mean_acc:.4f} ± {std_acc:.4f}")
    
    return

# ==================== 主执行块 ====================
if __name__ == "__main__":
    args = build_args()
    # 我们不再使用configs.yml，所有参数来自命令行或build_args的默认值
    # if args.use_cfg:
    #     args = load_best_configs(args, "configs.yml")
    print(args)
    main(args)