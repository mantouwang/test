import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AttentionFusion(nn.Module):
    """
    一个基于注意力的融合模块。
    它使用宏观特征作为Query，微观特征作为Key和Value，
    让宏观信息指导模型应该关注哪些微观细节。
    """

    def __init__(self, hidden_dim, n_heads=4):
        super(AttentionFusion, self).__init__()
        assert hidden_dim % n_heads == 0, "hidden_dim must be divisible by n_heads"

        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        # 定义Q, K, V的线性映射层
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)

        # 输出层
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, macro_features, micro_features):
        """
        macro_features: 宏观特征 (作为Query), shape: (N, D_hidden)
        micro_features: 微观特征 (作为Key和Value), shape: (N, D_hidden)
        """
        N = macro_features.shape[0]  # N = 基因数

        # 1. 线性映射
        Q = self.query_proj(macro_features)  # (N, D_hidden)
        K = self.key_proj(micro_features)  # (N, D_hidden)
        V = self.value_proj(micro_features)  # (N, D_hidden)

        # 2. 拆分成多头 (multi-head)
        # view and transpose: (N, D_hidden) -> (N, n_heads, head_dim) -> (n_heads, N, head_dim)
        Q = Q.view(N, self.n_heads, self.head_dim).transpose(0, 1)
        K = K.view(N, self.n_heads, self.head_dim).transpose(0, 1)
        V = V.view(N, self.n_heads, self.head_dim).transpose(0, 1)

        # 3. 计算注意力分数
        # (n_heads, N, head_dim) @ (n_heads, head_dim, N) -> (n_heads, N, N)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(attention_scores, dim=-1)

        # 4. 使用注意力权重加权Value
        # (n_heads, N, N) @ (n_heads, N, head_dim) -> (n_heads, N, head_dim)
        attended_values = torch.matmul(attention_weights, V)

        # 5. 合并多头
        # (n_heads, N, head_dim) -> (N, n_heads, head_dim) -> (N, D_hidden)
        attended_values = attended_values.transpose(0, 1).contiguous().view(N, self.hidden_dim)

        # 6. 通过输出层
        output = self.out_proj(attended_values)  # (N, D_hidden)

        return output


class FusionEncoder(nn.Module):
    """
    升级版“Y型”编码器，使用注意力机制进行融合。
    """

    def __init__(self, h5_in_dim, sc_in_dim, hidden_dim, fused_out_dim):
        super(FusionEncoder, self).__init__()

        # --- 分支A: 宏观特征学习分支 ---
        self.h5_stream = nn.Sequential(
            nn.Linear(h5_in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # --- 分支B: 微观特征学习分支 ---
        self.sc_stream = nn.Sequential(
            nn.Linear(sc_in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # --- 模块C: 基于注意力的特征融合模块 ---
        self.attention_fusion = AttentionFusion(hidden_dim=hidden_dim, n_heads=4)

        # --- 最后的处理层 ---
        # 将原始的宏观特征与经过注意力加权后的微观特征进行拼接
        self.fusion_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, fused_out_dim),
            # 输入维度是 hidden_dim (来自h5_output) + hidden_dim (来自attention_output)
            nn.ReLU()
        )

    def forward(self, x_h5, x_sc):
        # 数据分别流经两个分支，得到初步处理的特征
        h5_output = self.h5_stream(x_h5)
        sc_output = self.sc_stream(x_sc)

        # 使用注意力模块进行融合
        # h5_output作为Query, sc_output作为Key和Value
        attention_output = self.attention_fusion(h5_output, sc_output)

        # 将原始的宏观特征与注意力加权后的特征进行拼接
        # 这种残差连接式的设计通常更稳定
        final_concatenated = torch.cat((h5_output, attention_output), dim=1)

        # 通过最后的融合头得到最终输出
        fused_features = self.fusion_head(final_concatenated)

        return fused_features