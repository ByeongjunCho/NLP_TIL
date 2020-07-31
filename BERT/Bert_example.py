import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from torch.optim import Adam
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from IPython.display import clear_output


# dataset load
train_df = pd.read_csv('./nsmc/ratings_train.txt', sep='\t') # 15만개
test_df = pd.read_csv('./nsmc/ratings_test.txt', sep='\t') # 5만
train_df.dropna(inplace=True)
test_df.dropna(inplace=True)

# 랜덤하게 데이터 추출
# train_df = train_df.sample(frac=0.4, random_state=999)
# test_df = test_df.sample(frac=0.4, random_state=999)
train_df.tail()

# Dataset Class 선언
class NsmcDataset(Dataset):
    '''Naver Sentiment Movie Corpus Dataset'''

    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.iloc[idx, 1]
        label = self.df.iloc[idx, 2]
        return text, label

# dataloader 생성
nsmc_train_dataset = NsmcDataset(train_df)
train_loader = DataLoader(nsmc_train_dataset, batch_size=2, shuffle=True, num_workers=0) # window에서는 num_workers=0으로 해야한다.

# tokenizer 로드
device = torch.device('cuda')
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

# Bert load
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased')
model.to(device)

# trainer
nsmc_eval_dataset = NsmcDataset(test_df)

from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total # of training epochs
    per_device_train_batch_size=4,  # batch size per device during training
    per_device_eval_batch_size=4,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=nsmc_train_dataset,
    eval_dataset=nsmc_eval_dataset
)

trainer.train()