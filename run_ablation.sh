#!/bin/bash
# 消融实验运行脚本

echo "========================================"
echo "EmoCC模型消融实验启动"
echo "========================================"

cd ablation-model

echo ""
echo "运行参数 (与主模型Emocc-setup.py保持一致):"
echo "  - 随机种子: 24"
echo "  - Batch Size: 16"
echo "  - Epochs: 50"
echo "  - Learning Rate: 0.0005"
echo "  - GRU Hidden Size: 128"
echo "  - Dropout: 0.5"
echo "  - Weight Decay: 1e-6"
echo "  - Patience: 15"
echo ""

python ablation-model.py \
  --seed 24 \
  --batch_size 16 \
  --epochs 50 \
  --lr 0.0005 \
  --gru_size 128 \
  --dropout 0.5 \
  --weight_decay 1e-6 \
  --class_num 5 \
  --patience 15 \
  --max_posts 50 \
  --max_emojis_per_post 10 \
  --use_pretrained_emoji \
  --bert_pkl ../data/bert_embeddings.pkl \
  --emoji_csv ../data/reddit_500_emoji_batch.csv \
  --emoji2vec_path ../pre-trained/emoji2vec.bin

echo ""
echo "========================================"
echo "消融实验完成！"
echo "结果已保存至: ablation-model/ablation_results.csv"
echo "========================================"
