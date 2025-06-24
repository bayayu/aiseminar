import subprocess
import os
import re
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, pipeline
from transformers import DataCollatorWithPadding
from datasets import Dataset
from evaluate import load
from transformers import set_seed
import torch
import random


# シードの固定
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)

seed_everything(42)  # 任意のシード値を設定
# データセットの準備
if not os.path.exists('wrime-ver1.tsv'):
    subprocess.run(["wget", "https://github.com/ids-cv/wrime/raw/master/wrime-ver1.tsv"])

df_wrime = pd.read_table('wrime-ver1.tsv')

# オノマトペの読み込み
def load_onomatope_sentences(file_name):
    with open(file_name, 'r', encoding='utf-8') as f:
        sentences = f.read().splitlines()
    return sentences

onomatope_sentences = load_onomatope_sentences('onomatope_sentences.txt')

# 前処理
emotion_names = ['Joy', 'Sadness', 'Anticipation', 'Surprise', 'Anger', 'Fear', 'Disgust', 'Trust']
emotion_names_jp = ['喜び', '悲しみ', '期待', '驚き', '怒り', '恐れ', '嫌悪', '信頼']
num_labels = len(emotion_names)

# 感情強度が最大2以上かつオノマトペに含まれていない文章を抽出
df_wrime['readers_emotion_intensities'] = df_wrime.apply(lambda x: [x['Avg. Readers_' + name] for name in emotion_names], axis=1)
is_target = df_wrime['readers_emotion_intensities'].map(lambda x: max(x) >= 2)
not_in_onomatope = ~df_wrime['Sentence'].isin(onomatope_sentences)
df_wrime_target = df_wrime[is_target & not_in_onomatope]

# train / test に分割
df_groups = df_wrime_target.groupby('Train/Dev/Test')
df_train = df_groups.get_group('train')
df_test = pd.concat([df_groups.get_group('dev'), df_groups.get_group('test')])

# モデルとトークナイザの読み込み
checkpoint = 'cl-tohoku/bert-base-japanese-v2'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=num_labels)

def tokenize_function(batch):
    tokenized_batch = tokenizer(batch['Sentence'], truncation=True, padding='max_length')
    tokenized_batch['labels'] = [x / np.sum(x) for x in batch['readers_emotion_intensities']]
    return tokenized_batch

# Transformers用のデータセット形式に変換
target_columns = ['Sentence', 'readers_emotion_intensities']
train_dataset = Dataset.from_pandas(df_train[target_columns])
test_dataset = Dataset.from_pandas(df_test[target_columns])

# 前処理の適用
train_tokenized_dataset = train_dataset.map(tokenize_function, batched=True)
test_tokenized_dataset = test_dataset.map(tokenize_function, batched=True)

# DataCollatorを使用して自動的にパディング
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 評価指標の定義
metric = load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    label_ids = np.argmax(labels, axis=-1)
    return metric.compute(predictions=predictions, references=label_ids)

# 訓練時の設定
training_args = TrainingArguments(
    output_dir="test_trainer",
    per_device_train_batch_size=8,
    num_train_epochs=3.0,
    eval_strategy="steps",
    eval_steps=200
)

# Trainerを生成
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized_dataset,
    eval_dataset=test_tokenized_dataset,
    compute_metrics=compute_metrics,
    data_collator=data_collator
)

# 訓練を実行
trainer.train()

# 訓練済みモデルの保存
model.save_pretrained('finetuned_model_kai1')
tokenizer.save_pretrained('finetuned_model_kai1')

print(f"学習データ数: {len(df_train)}")
print(f"検証データ数: {len(df_test)}")
