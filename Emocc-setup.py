"""
EmoCC Training Script: 支持双模态BERT+Emoji训练
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
from collections import Counter
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from Emocc_model.model import BertEmojiModel
from Emocc_model.utils.dataset import BertEmojiDataset, collate_fn_bert_emoji
from Emocc_model.utils.emoji_utils import get_emoji_vocabulary, load_pretrained_emoji_weights
from Emocc_model.utils.loss import loss_function, gr_metrics


def print_class_statistics(preds, labels, class_num=5):
    """打印分类统计信息和混淆矩阵"""
    preds = np.array(preds)
    labels = np.array(labels)
    cm = confusion_matrix(labels, preds, labels=list(range(class_num)))
    
    print("\n" + "=" * 60)
    print("分类统计:")
    print("=" * 60)
    print(f"{'类别':<8} {'样本数':<10} {'正确':<10} {'错误':<10} {'准确率':<10}")
    print("-" * 60)
    
    for cls in range(class_num):
        total = (labels == cls).sum()
        correct = cm[cls, cls]
        wrong = total - correct
        acc = correct / total if total > 0 else 0.0
        print(f"类别{cls:<5} {total:<10} {correct:<10} {wrong:<10} {acc:.4f}")
    
    overall_acc = (preds == labels).sum() / len(labels)
    print("-" * 60)
    print(f"{'总计':<8} {len(labels):<10} {(preds == labels).sum():<10} {(preds != labels).sum():<10} {overall_acc:.4f}")
    print("=" * 60)
    
    print("\n混淆矩阵 (行=真实, 列=预测):")
    print("=" * 60)
    header = "真实\\预测" + "".join([f"  类{i:>3}" for i in range(class_num)])
    print(header)
    print("-" * 60)
    for i in range(class_num):
        row = f"  类别{i}  " + "".join([f"  {cm[i,j]:>5}" for j in range(class_num)])
        print(row)
    print("=" * 60)


def train_epoch_bert_emoji(model, dataloader, optimizer, device, args):
    """BERT+Emoji模型训练一个epoch"""
    model.train()
    total_loss = 0.0
    out_result = []
    label_result = []
    
    for batch in tqdm(dataloader, desc="训练中"):
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


def eval_epoch_bert_emoji(model, dataloader, device, args):
    """BERT+Emoji模型评估一个epoch"""
    model.eval()
    total_loss = 0.0
    out_result = []
    label_result = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="验证中"):
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


def test_model_bert_emoji(model, dataloader, device):
    """BERT+Emoji模型测试"""
    model.eval()
    out_result = []
    label_result = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="测试中"):
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


def set_seed(seed=2024):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_reddit_data(file_path='data/reddit_clean.pkl'):
    """读取 Reddit 数据"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据文件未找到: {file_path}")
    print(f"正在读取数据: {file_path}")
    reddit_data = pd.read_pickle(file_path)
    labels = [user['label'] for user in reddit_data]
    users = [user['user'] for user in reddit_data]
    return reddit_data, labels, users


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


def parse_args():
    """参数解析"""
    parser = argparse.ArgumentParser(description='EmoCC Training')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--embedding_dim", type=int, default=512)
    parser.add_argument("--emoji_embedding_dim", type=int, default=300)
    parser.add_argument("--gru_size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--class_num", type=int, default=5)
    parser.add_argument("--patience", type=int, default=15)
    
    parser.add_argument("--bert_pkl", type=str, default='data/reddit_500_bert_embeddings.pkl')
    parser.add_argument("--emoji_csv", type=str, default='data/reddit_500_emoji.csv')
    parser.add_argument("--emoji2vec_path", type=str, default='pre-trained/emoji2vec.bin')
    parser.add_argument("--save_path", type=str, default='./Emocc_model/checkpoints/emocc_model.pth')
    
    parser.add_argument("--use_pretrained_emoji", action='store_true', 
                        help='使用预训练的emoji2vec初始化emoji嵌入层')
    parser.add_argument("--mode", type=str, choices=['train', 'test', 'train_test'], default='train_test')
    parser.add_argument("--max_posts", type=int, default=50, help='最大帖子序列长度')
    parser.add_argument("--max_emojis_per_post", type=int, default=10, help='每个帖子最大emoji数')

    return parser.parse_args()


def train_bert_emoji():
    """训练BERT + Emoji双模态模型"""
    args = parse_args()
    set_seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs('Emocc_model/checkpoints', exist_ok=True)
    
    print(f"[{time.asctime()}] 开始训练 BERT+Emoji模型...")
    print(f"设备: {device}")
    print(f"模式: {args.mode}")
    print(f"使用预训练emoji: {args.use_pretrained_emoji}")
    print(f"参数: {args}")
    
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
    
    # 数据划分 8:1:1
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
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn_bert_emoji,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn_bert_emoji,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn_bert_emoji,
        num_workers=0
    )
    
    print("\n===== 构建模型 =====")
    model = BertEmojiModel(
        args=args,
        emoji_vocab_size=len(emoji_vocab),
        device=device,
        emoji_weights=emoji_weights,
        is_pretrain_emoji=args.use_pretrained_emoji
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数总数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    early_stopping = EarlyStopping(patience=args.patience, verbose=True, path=args.save_path)
    
    print("\n===== 开始训练 =====")
    for epoch in range(args.epochs):
        print(f"\nEpoch [{epoch + 1}/{args.epochs}]")
        
        train_loss, train_acc, train_GP, train_GR, train_FS = train_epoch_bert_emoji(
            model, train_loader, optimizer, device, args)
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | GP: {train_GP:.4f} | GR: {train_GR:.4f} | FS: {train_FS:.4f}")
        
        val_loss, val_acc, val_GP, val_GR, val_FS = eval_epoch_bert_emoji(
            model, val_loader, device, args)
        print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | GP: {val_GP:.4f} | GR: {val_GR:.4f} | FS: {val_FS:.4f}")
        
        scheduler.step()
        early_stopping(val_FS, model)
        if early_stopping.early_stop:
            print("早停触发")
            break
    
    print(f"\n训练完成，最佳FScore: {early_stopping.val_fs_max:.4f}")
    
    if args.mode in ['test', 'train_test']:
        print(f"\n===== 开始测试 =====")
        set_seed(args.seed)
        
        if os.path.exists(args.save_path):
            checkpoint = torch.load(args.save_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"已加载最佳模型: {args.save_path}")
        else:
            print(f"警告: 模型文件不存在，使用当前模型参数")
        
        test_acc, test_GP, test_GR, test_FS, test_OE, preds, labels_test = test_model_bert_emoji(
            model, test_loader, device)
        
        print("\n" + "=" * 50)
        print("最终测试结果:")
        print("=" * 50)
        print(f"Accuracy: {test_acc:.4f}")
        print(f"GP: {test_GP:.4f}")
        print(f"GR: {test_GR:.4f}")
        print(f"FScore: {test_FS:.4f}")
        print(f"OE: {test_OE:.4f}")
        print(f"Macro-F1: {f1_score(labels_test, preds, average='macro'):.4f}")
        print("=" * 50)
        
        print_class_statistics(preds, labels_test, class_num=args.class_num)
    
    print("\n所有任务完成。")


if __name__ == '__main__':
    train_bert_emoji()
