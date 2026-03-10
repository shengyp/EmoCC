"""EmoCC Model: Emoji-Enhanced Multi-view Temporal Graph Model
整合了组件和主模型代码
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 时间注意力机制 已经检查，看的论文觉得没问题
class TemporalAttention(nn.Module):
    """
    时间注意力机制（Temporal Attention）
    参考论文: Towards Ordinal Suicide Ideation Detection on Social Media
    公式: α_t^(i) = c^T tanh(W_x h_t^(i) + b_x)
    
    学习用户历史帖子的自适应权重，突出包含自杀风险指标的帖子
    """
    def __init__(self, input_dim):
        super(TemporalAttention, self).__init__()
        self.input_dim = input_dim
        
        self.W_x = nn.Linear(input_dim, input_dim)
        self.c = nn.Parameter(torch.Tensor(input_dim), requires_grad=True)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.W_x.weight)
        nn.init.zeros_(self.W_x.bias)
        stdv = 1.0 / np.sqrt(self.input_dim)
        nn.init.uniform_(self.c, -stdv, stdv)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (B, T, d) 用户历史帖子的上下文表示
            mask: (B, T) 有效帖子的mask
        Returns:
            a_i: (B, d) 用户历史帖子的时间注意力加权表示
        """
        B, T, d = x.size()
        
        attn_input = torch.tanh(self.W_x(x))
        attn_scores = torch.matmul(attn_input, self.c)
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = attn_weights.unsqueeze(-1)
        
        a_i = (x * attn_weights).sum(dim=1)
        
        return a_i


class GlobalSelfAttention(nn.Module):
    """
    使用自注意力机制建模用户所有历史帖子之间的全局关系，提取全局情感共性 V_common
    """
    def __init__(self, input_dim, hidden_dim=None):
        super(GlobalSelfAttention, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim or input_dim

        self.W_Q = nn.Linear(input_dim, self.hidden_dim)
        self.W_K = nn.Linear(input_dim, self.hidden_dim)
        self.W_V = nn.Linear(input_dim, self.hidden_dim)
        self.scale = torch.sqrt(torch.tensor(self.hidden_dim, dtype=torch.float32))

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W_Q.weight)
        nn.init.xavier_uniform_(self.W_K.weight)
        nn.init.xavier_uniform_(self.W_V.weight)
        nn.init.zeros_(self.W_Q.bias)
        nn.init.zeros_(self.W_K.bias)
        nn.init.zeros_(self.W_V.bias)

    def forward(self, x, mask=None):
        B, T, d = x.size()

        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if mask is not None:
            attn_mask = mask.unsqueeze(1)
            attn_scores = attn_scores.masked_fill(attn_mask == 0, -1e9)

        attn_weights = F.softmax(attn_scores, dim=-1)
        V_common = torch.matmul(attn_weights, V)

        if mask is not None:
            mask_float = mask.unsqueeze(-1).type_as(V_common)
            V_common = V_common * mask_float

        V_common = V_common.sum(dim=1)
        return V_common


# 信号层面门控融合 已经检查，看的论文觉得没问题
class AdaptiveGateFusion(nn.Module):
    """
    Adaptive Multi-view Graph Fusion (参考论文公式13-14)
    λ = Sigmoid(W_f·V_f + W_t·V_t)
    V_fuse = (1 - λ)·V_f + λ·V_t
    
    这是真正的信号层面门控融合，而非简单拼接
    """
    def __init__(self, hidden_size):
        super(AdaptiveGateFusion, self).__init__()
        self.W_f = nn.Linear(hidden_size, hidden_size)
        self.W_t = nn.Linear(hidden_size, hidden_size)
        self.sigmoid = nn.Sigmoid()
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.W_f.weight)
        nn.init.xavier_uniform_(self.W_t.weight)
        nn.init.zeros_(self.W_f.bias)
        nn.init.zeros_(self.W_t.bias)
    
    def forward(self, V_f, V_t):
        """
        Args:
            V_f: 第一个特征向量 (B, H)
            V_t: 第二个特征向量 (B, H)
        Returns:
            V_fuse: 融合后的特征 (B, H)
        """
        assert V_f.shape == V_t.shape, f"AdaptiveGateFusion输入维度不匹配: {V_f.shape} vs {V_t.shape}"
        lambda_gate = self.sigmoid(self.W_f(V_f) + self.W_t(V_t))
        V_fuse = (1 - lambda_gate) * V_f + lambda_gate * V_t
        return V_fuse

class MultiLayerPerceptron(nn.Module):
    """多层感知机分类器"""
    def __init__(self, input_dim, embed_dims, dropout, class_num, output_layer=True):
        super().__init__()
        layers = []
        for embed_dim in embed_dims:
            layers.append(nn.Linear(input_dim, embed_dim))
            layers.append(nn.BatchNorm1d(embed_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(nn.Linear(input_dim, class_num))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class BertEmojiModel(nn.Module):
    """
    BERT + Emoji双模态层次融合模型
    
    架构流程：
    1. Text Encoding Layer (GRU1):
       - BERT嵌入(768维) → Post-level BiGRU1(768 → 2H)
    
    2. Emoji Fusion Layer:
       - GRU1输出(2H) + Emoji均值池化(300维) → 拼接(2H+300维)
    
    3. Feature Extraction:
       - 分支1: V_short = BiGRU1最后一个有效帖子向量 (B, 2H)
       - 分支2: V_common = GlobalSelfAttention(拼接特征) (B, 2H+300) → (B, 2H)
    
    4. Fusion Layer:
       - V_final = AdaptiveGateFusion(V_short, V_common) (B, 2H)
    
    5. Output:
       - Logits = MLP(V_final) (B, C)
    
    输入:
        - bert_embeddings: (B, T, 768) 预训练BERT帖子嵌入
        - emoji_ids: (B, T, E) 每个帖子的emoji ID序列
        - post_masks: (B, T) 有效帖子mask
        - emoji_masks: (B, T, E) 每个帖子的emoji有效mask
    
    输出:
        - logits: (B, C) 分类logits
    """
    def __init__(self, args, emoji_vocab_size, device, 
                 emoji_weights=None, is_pretrain_emoji=False):
        super(BertEmojiModel, self).__init__()
        self.args = args
        self.device = device
        self.dropout = args.dropout
        self.gru_size = args.gru_size
        self.class_num = args.class_num
        self.bert_dim = 768
        self.emoji_dim = 300
        
        if is_pretrain_emoji and emoji_weights is not None:
            self.emoji_embed = nn.Embedding.from_pretrained(emoji_weights, freeze=False)
        else:
            self.emoji_embed = nn.Embedding(emoji_vocab_size, self.emoji_dim)
        
        self.emoji_embed_drop = nn.Dropout(args.dropout)
        
        self.gru1 = nn.GRU(
            input_size=self.bert_dim,
            hidden_size=self.gru_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        self.gru1_drop = nn.Dropout(args.dropout)
        
        self.global_attention = GlobalSelfAttention(
            input_dim=2 * self.gru_size + self.emoji_dim,
        #   input_dim=2 * self.gru_size,#消融
            hidden_dim=2 * self.gru_size
        )

        #self.temporal_attention = TemporalAttention(input_dim=2 * self.gru_size)
        self.adaptive_gate_fusion = AdaptiveGateFusion(2 * self.gru_size)
        
        self.class_fc = MultiLayerPerceptron(
            input_dim=2 * self.gru_size,
            embed_dims=[2 * self.gru_size, self.gru_size, self.gru_size // 2],
            dropout=self.dropout,
            class_num=self.class_num
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.gru1.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
    
    def forward(self, bert_embeddings, emoji_ids, post_masks, emoji_masks):
        """
        Args:
            bert_embeddings: (B, T, 768) BERT嵌入
            emoji_ids: (B, T, E) Emoji ID序列
            post_masks: (B, T) 帖子有效mask
            emoji_masks: (B, T, E) Emoji有效mask
        Returns:
            logits: (B, C) 分类logits
        """
        B, T, _ = bert_embeddings.size()
        
        post_lengths = post_masks.sum(dim=1).cpu()
        non_zero_indices = (post_lengths > 0).nonzero(as_tuple=True)[0]
        if len(non_zero_indices) == 0:
            return torch.zeros(B, self.class_num, device=self.device)
        
        valid_bert = bert_embeddings[non_zero_indices]
        valid_lengths = post_lengths[non_zero_indices]
        
        packed_bert = nn.utils.rnn.pack_padded_sequence(
            valid_bert, valid_lengths, batch_first=True, enforce_sorted=False
        )
        gru1_output, _ = self.gru1(packed_bert)
        gru1_output, _ = nn.utils.rnn.pad_packed_sequence(gru1_output, batch_first=True)
        gru1_output = self.gru1_drop(gru1_output)
        
        emoji_embeds = self.emoji_embed(emoji_ids)
        emoji_embeds = self.emoji_embed_drop(emoji_embeds)
        
        emoji_mask_expanded = emoji_masks.unsqueeze(-1).float()
        emoji_embeds_masked = emoji_embeds * emoji_mask_expanded
        emoji_counts = emoji_masks.sum(dim=-1, keepdim=True).clamp(min=1)
        emoji_mean = emoji_embeds_masked.sum(dim=2) / emoji_counts
        
        max_len = int(valid_lengths.max().item())
        if max_len <= 0:
            max_len = 1
        valid_emoji = emoji_mean[non_zero_indices][:, :max_len, :]
        
        fused_features = torch.cat([gru1_output, valid_emoji], dim=-1)
        
        valid_post_masks = post_masks[non_zero_indices][:, :max_len]
        
        last_indices = (valid_post_masks.cumsum(dim=1) == valid_post_masks.sum(dim=1, keepdim=True)).float()
        V_short_valid = (gru1_output * last_indices.unsqueeze(-1)).sum(dim=1)
        
        V_common_valid = self.global_attention(fused_features, mask=valid_post_masks)
    #   V_common_valid = self.global_attention(gru1_output, mask=valid_post_masks) #消融
        V_short = torch.zeros(B, 2 * self.gru_size, device=self.device)
        V_short[non_zero_indices] = V_short_valid
        
        V_common = torch.zeros(B, 2 * self.gru_size, device=self.device)
        V_common[non_zero_indices] = V_common_valid
        
        V_final = self.adaptive_gate_fusion(V_short, V_common)
        
        logits = self.class_fc(V_final)
        
        return logits
