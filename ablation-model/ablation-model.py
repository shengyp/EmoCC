"""
消融实验代码 - EmoCC模型
按照架构图进行5个消融实验，每个实验都有详细的变更说明
"""
import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import time
import random
import argparse
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

import sys
sys.path.append('..')
from Emocc_model.model import GlobalSelfAttention, AdaptiveGateFusion, MultiLayerPerceptron
from Emocc_model.utils.dataset import BertEmojiDataset, collate_fn_bert_emoji
from Emocc_model.utils.emoji_utils import get_emoji_vocabulary, load_pretrained_emoji_weights
from Emocc_model.utils.loss import loss_function, gr_metrics


def set_seed(seed=42):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class BertEmojiModelAblation(nn.Module):
    """
    消融实验模型 - 支持5种消融配置
    
    完整模型架构：
    1. Text Encoding: BERT(768) → BiGRU → (B, T, 2H)
    2. Emoji Encoding: Emoji Embedding → Mean Pooling → (B, T, 300)
    3. Feature Fusion: Cat[BiGRU, Emoji] → (B, T, 2H+300)
    4. Global Self-Attention: (B, T, 2H+300) → V_common (B, 2H)
    5. Last Valid Post: BiGRU最后有效帖子 → V_short (B, 2H)
    6. Gated Fusion: V_short + V_common → V_final (B, 2H)
    7. MLP Classifier: V_final → Logits
    """
    def __init__(self, args, emoji_vocab_size, device, 
                 emoji_weights=None, is_pretrain_emoji=False,
                 ablation_type='full'):
        super(BertEmojiModelAblation, self).__init__()
        self.args = args
        self.device = device
        self.dropout = args.dropout
        self.gru_size = args.gru_size
        self.class_num = args.class_num
        self.bert_dim = 768
        self.emoji_dim = 300
        self.ablation_type = ablation_type
        
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
        
        if ablation_type == 'no_emoji_feature':
            self.global_attention = GlobalSelfAttention(
                input_dim=2 * self.gru_size,
                hidden_dim=2 * self.gru_size
            )
        elif ablation_type == 'no_text_feature':
            self.global_attention = GlobalSelfAttention(
                input_dim=self.emoji_dim,
                hidden_dim=2 * self.gru_size
            )
        else:
            self.global_attention = GlobalSelfAttention(
                input_dim=2 * self.gru_size + self.emoji_dim,
                hidden_dim=2 * self.gru_size
            )
        
        if ablation_type == 'concat_fusion':
            self.adaptive_gate_fusion = None
        else:
            self.adaptive_gate_fusion = AdaptiveGateFusion(2 * self.gru_size)
        
        if ablation_type == 'concat_fusion':
            classifier_input_dim = 4 * self.gru_size
        else:
            classifier_input_dim = 2 * self.gru_size
        
        self.class_fc = MultiLayerPerceptron(
            input_dim=classifier_input_dim,
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
        前向传播 - 根据ablation_type执行不同的消融逻辑
        
        ablation_type说明:
        - 'full': 完整模型
        - 'no_global_attention': V_final = V_short (只用最后帖子)
        - 'no_last_post': V_final = V_common (只用全局注意力)
        - 'no_emoji_feature': Global Attention输入只有BiGRU输出
        - 'no_text_feature': 只用Emoji特征，移除V_short
        - 'concat_fusion': 用Concat替换Gated Fusion
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
        
        valid_post_masks = post_masks[non_zero_indices][:, :max_len]
        
        last_indices = (valid_post_masks.cumsum(dim=1) == valid_post_masks.sum(dim=1, keepdim=True)).float()
        V_short_valid = (gru1_output * last_indices.unsqueeze(-1)).sum(dim=1)
        
        if self.ablation_type == 'no_emoji_feature':
            V_common_valid = self.global_attention(gru1_output, mask=valid_post_masks)
        elif self.ablation_type == 'no_text_feature':
            V_common_valid = self.global_attention(valid_emoji, mask=valid_post_masks)
        else:
            fused_features = torch.cat([gru1_output, valid_emoji], dim=-1)
            V_common_valid = self.global_attention(fused_features, mask=valid_post_masks)
        
        V_short = torch.zeros(B, 2 * self.gru_size, device=self.device)
        V_short[non_zero_indices] = V_short_valid
        
        V_common = torch.zeros(B, 2 * self.gru_size, device=self.device)
        V_common[non_zero_indices] = V_common_valid
        
        if self.ablation_type == 'no_global_attention':
            V_final = V_short
        elif self.ablation_type == 'no_last_post':
            V_final = V_common
        elif self.ablation_type == 'no_text_feature':
            V_final = V_common
        elif self.ablation_type == 'concat_fusion':
            V_final = torch.cat([V_short, V_common], dim=-1)
        else:
            V_final = self.adaptive_gate_fusion(V_short, V_common)
        
        logits = self.class_fc(V_final)
        return logits


def train_epoch(model, dataloader, optimizer, device, args):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    out_result = []
    label_result = []
    
    for batch in tqdm(dataloader, desc="训练中", leave=False):
        bert_emb, emoji_ids, post_masks, emoji_masks, labels = batch
        bert_emb = bert_emb.to(device)
        emoji_ids = emoji_ids.to(device)
        post_masks = post_masks.to(device)
        emoji_masks = emoji_masks.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        output = model(bert_emb, emoji_ids, post_masks, emoji_masks)
        loss = loss_function(output, labels, loss_type="ordered", expt_type=args.class_num, scale=1.4)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        out_result.extend(predicted.cpu().numpy().tolist())
        label_result.extend(labels.cpu().numpy().tolist())
    
    acc = accuracy_score(label_result, out_result)
    f1 = f1_score(label_result, out_result, average='macro')
    GP, GR, FS, _ = gr_metrics(out_result, label_result)
    return total_loss / len(dataloader), acc, GP, GR, FS


def eval_epoch(model, dataloader, device, args):
    """评估一个epoch"""
    model.eval()
    total_loss = 0.0
    out_result = []
    label_result = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="验证中", leave=False):
            bert_emb, emoji_ids, post_masks, emoji_masks, labels = batch
            bert_emb = bert_emb.to(device)
            emoji_ids = emoji_ids.to(device)
            post_masks = post_masks.to(device)
            emoji_masks = emoji_masks.to(device)
            labels = labels.to(device)
            
            output = model(bert_emb, emoji_ids, post_masks, emoji_masks)
            loss = loss_function(output, labels, loss_type="ordered", expt_type=args.class_num, scale=1.4)
            total_loss += loss.item()
            
            _, predicted = torch.max(output.data, 1)
            out_result.extend(predicted.cpu().numpy().tolist())
            label_result.extend(labels.cpu().numpy().tolist())
    
    acc = accuracy_score(label_result, out_result)
    f1 = f1_score(label_result, out_result, average='macro')
    GP, GR, FS, _ = gr_metrics(out_result, label_result)
    return total_loss / len(dataloader), acc, GP, GR, FS


def test_model(model, dataloader, device):
    """测试模型"""
    model.eval()
    out_result = []
    label_result = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="测试中", leave=False):
            bert_emb, emoji_ids, post_masks, emoji_masks, labels = batch
            bert_emb = bert_emb.to(device)
            emoji_ids = emoji_ids.to(device)
            post_masks = post_masks.to(device)
            emoji_masks = emoji_masks.to(device)
            labels = labels.to(device)
            
            output = model(bert_emb, emoji_ids, post_masks, emoji_masks)
            _, predicted = torch.max(output.data, 1)
            
            out_result.extend(predicted.cpu().numpy().tolist())
            label_result.extend(labels.cpu().numpy().tolist())
    
    acc = accuracy_score(label_result, out_result)
    f1 = f1_score(label_result, out_result, average='macro')
    GP, GR, FS, OE = gr_metrics(out_result, label_result)
    
    return acc, GP, GR, FS, OE, out_result, label_result


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pth'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_fs_max = -np.inf
        self.delta = delta
        self.path = path

    def __call__(self, val_fs, model):
        score = val_fs
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_fs, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"早停计数: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_fs, model)
            self.counter = 0

    def save_checkpoint(self, val_fs, model):
        if self.verbose:
            print(f"FScore 提升 ({self.val_fs_max:.6f} --> {val_fs:.6f})，保存模型...")
        torch.save({'model_state_dict': model.state_dict()}, self.path)
        self.val_fs_max = val_fs


def run_single_ablation(args, ablation_config, train_loader, val_loader, test_loader, 
                        emoji_vocab_size, emoji_weights, device):
    """
    运行单个消融实验
    
    Args:
        ablation_config: dict包含 'name', 'type', 'description'
    """
    print(f"\n{'='*80}")
    print(f"开始实验: {ablation_config['name']}")
    print(f"描述: {ablation_config['description']}")
    print(f"{'='*80}")
    
    set_seed(args.seed)
    
    model = BertEmojiModelAblation(
        args=args,
        emoji_vocab_size=emoji_vocab_size,
        device=device,
        emoji_weights=emoji_weights,
        is_pretrain_emoji=args.use_pretrained_emoji,
        ablation_type=ablation_config['type']
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    checkpoint_path = f"./ablation_checkpoints/{ablation_config['type']}_model.pth"
    os.makedirs('./ablation_checkpoints', exist_ok=True)
    early_stopping = EarlyStopping(patience=args.patience, verbose=False, path=checkpoint_path)
    
    for epoch in range(args.epochs):
        print(f"\nEpoch [{epoch + 1}/{args.epochs}]")
        
        train_loss, train_acc, train_GP, train_GR, train_FS = train_epoch(
            model, train_loader, optimizer, device, args)
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | GP: {train_GP:.4f} | GR: {train_GR:.4f} | FS: {train_FS:.4f}")
        
        val_loss, val_acc, val_GP, val_GR, val_FS = eval_epoch(
            model, val_loader, device, args)
        print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | GP: {val_GP:.4f} | GR: {val_GR:.4f} | FS: {val_FS:.4f}")
        
        scheduler.step()
        early_stopping(val_FS, model)
        if early_stopping.early_stop:
            print(f"早停触发于epoch {epoch+1}")
            break
    
    print(f"\n训练完成，最佳FScore: {early_stopping.val_fs_max:.4f}")
    
    set_seed(args.seed)
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"已加载最佳模型: {checkpoint_path}")
    else:
        print(f"警告: 模型文件不存在，使用当前模型参数")
    
    test_acc, test_GP, test_GR, test_FS, test_OE, preds, labels_test = test_model(model, test_loader, device)
    test_f1 = f1_score(labels_test, preds, average='macro')
    
    print(f"\n{ablation_config['name']} 测试结果:")
    print(f"  Acc: {test_acc:.4f} | GP: {test_GP:.4f} | GR: {test_GR:.4f} | F-score: {test_FS:.4f} | OE: {test_OE:.4f} | Macro-F1: {test_f1:.4f}")
    
    return {
        'Experiment': ablation_config['name'],
        'Description': ablation_config['description'],
        'Acc (%)': round(test_acc * 100, 2),
        'F1 (%)': round(test_f1 * 100, 2),
        'GP': round(test_GP, 2),
        'GR': round(test_GR, 2),
        'F-score': round(test_FS, 2),
        'OE': round(test_OE, 2)
    }


def run_ablation_study(args):
    """运行完整的消融实验"""
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*80}")
    print("EmoCC模型消融实验")
    print(f"设备: {device}")
    print(f"随机种子: {args.seed}")
    print(f"{'='*80}")
    
    print("\n===== 构建Emoji词汇表 =====")
    emoji_vocab, emoji_to_id = get_emoji_vocabulary(
        args.emoji_csv, 
        max_vocab_size=500
    )
    
    emoji_weights = None
    if args.use_pretrained_emoji:
        print("\n===== 加载预训练Emoji权重 =====")
        emoji_weights_np = load_pretrained_emoji_weights(
            emoji_vocab, 
            args.emoji2vec_path, 
            embedding_dim=300
        )
        emoji_weights = torch.FloatTensor(emoji_weights_np)
    
    print("\n===== 构建数据集 =====")
    dataset = BertEmojiDataset(
        bert_pkl_path=args.bert_pkl,
        emoji_csv_path=args.emoji_csv,
        emoji_to_id=emoji_to_id,
        max_posts=args.max_posts,
        max_emojis_per_post=args.max_emojis_per_post
    )
    
    print(f"数据集大小: {len(dataset)}")
    
    print("\n===== 数据划分（8:1:1）=====")
    indices = list(range(len(dataset)))
    labels_list = [dataset[i]['label'] for i in indices]
    
    train_indices, test_indices = train_test_split(
        indices, test_size=0.2, stratify=labels_list, random_state=args.seed
    )
    test_labels = [labels_list[i] for i in test_indices]
    test_indices, val_indices = train_test_split(
        test_indices, test_size=0.5, stratify=test_labels, random_state=args.seed
    )
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    
    print(f"训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}, 测试集: {len(test_dataset)}")
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, 
        collate_fn=collate_fn_bert_emoji, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, 
        collate_fn=collate_fn_bert_emoji, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, 
        collate_fn=collate_fn_bert_emoji, num_workers=0
    )
    
    ablation_experiments = [
        {
            'name': 'Full Model',
            'type': 'full',
            'description': '完整模型: BiGRU + Emoji特征 + Global Self-Attention + Last Post + Gated Fusion'
        },
        {
            'name': 'w/o Global Self-Attention',
            'type': 'no_global_attention',
            'description': '移除全局自注意力: V_final = V_short (只使用最后有效帖子特征)'
        },
        {
            'name': 'w/o Last Post',
            'type': 'no_last_post',
            'description': '移除最后帖子特征: V_final = V_common (只使用全局注意力输出)'
        },
        {
            'name': 'w/o Emoji Feature',
            'type': 'no_emoji_feature',
            'description': '移除Emoji特征: Global Self-Attention输入只有BiGRU输出(B,T,2H)，不拼接Emoji'
        },
        {
            'name': 'w/o Text Feature',
            'type': 'no_text_feature',
            'description': '移除文本特征: 去掉V_short，Global Self-Attention输入只有Emoji特征(B,T,300)'
        },
        {
            'name': 'Concat Fusion',
            'type': 'concat_fusion',
            'description': '用拼接替换门控融合: V_final = Concat(V_short, V_common)而非Gated Fusion'
        }
    ]
    
    results = []
    for ablation_config in ablation_experiments:
        result = run_single_ablation(
            args, ablation_config, train_loader, val_loader, test_loader,
            len(emoji_vocab), emoji_weights, device
        )
        results.append(result)
    
    results_df = pd.DataFrame(results)
    results_df.to_csv('ablation_results.csv', index=False, encoding='utf-8')
    
    print(f"\n{'='*80}")
    print("消融实验完成！结果已保存至: ablation_results.csv")
    print(f"{'='*80}")
    print("\n实验结果汇总:")
    print(results_df.to_string(index=False))
    
    return results_df


def parse_args():
    """参数解析"""
    parser = argparse.ArgumentParser(description='EmoCC Ablation Study')
    parser.add_argument("--seed", type=int, default=24)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gru_size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--class_num", type=int, default=5)
    parser.add_argument("--patience", type=int, default=15)
    
    parser.add_argument("--bert_pkl", type=str, default='../data/bert_embeddings.pkl')
    parser.add_argument("--emoji_csv", type=str, default='../data/reddit_500_emoji_batch.csv')
    parser.add_argument("--emoji2vec_path", type=str, default='../pre-trained/emoji2vec.bin')
    
    parser.add_argument("--use_pretrained_emoji", action='store_true', 
                        help='使用预训练的emoji2vec初始化emoji嵌入层')
    parser.add_argument("--max_posts", type=int, default=50, help='最大帖子序列长度')
    parser.add_argument("--max_emojis_per_post", type=int, default=10, help='每个帖子最大emoji数')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_ablation_study(args)
