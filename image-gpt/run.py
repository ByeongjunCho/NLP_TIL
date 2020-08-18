import gluonnlp as nlp
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from imageGPT import GPT2Config, GPT2LMHeadModel
import datetime
import numpy as np
import time
# from transformers import GPT2Config, GPT2LMHeadModel

# model 설정
kogpt2_config = {
    "initializer_range": 0.02,
    "layer_norm_epsilon": 1e-05,
    "n_ctx": 1024,
    "n_embd": 512,
    "n_head": 8,
    "n_layer": 12,
    "n_positions": 28*28,
    "vocab_size": 256,
    'embd_pdrop': 0.1,
    'attn_pdrop': 0.1,
    'resid_pdrop': 0.1,
    'output_past': False,
    'output_attentions': True
}
batch_size = 128
INF = 100000000

# Load model
model = GPT2LMHeadModel(config=GPT2Config.from_dict(kogpt2_config))
cuda = torch.device('cuda: 0')
model.to(cuda)

# Load datasets
ds = lambda x, y: TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
train_x = np.load('./data/train_x.npy')
train_y = np.load('./data/train_y.npy')
test_x = np.load('./data/test_x.npy')
test_y = np.load('./data/test_y.npy')

# train/valid 배분
train_ds = ds(train_x, train_y)
train_size = int(0.9*len(train_ds))
train_ds, valid_ds = random_split(
    train_ds, [train_size, len(train_ds) - train_size]
)
# test
test_ds = ds(test_x, test_y)


train_datasets = DataLoader(train_ds, shuffle=True, batch_size=batch_size)
valid_datasets = DataLoader(valid_ds, shuffle=True, batch_size=batch_size)
test_datasets  = DataLoader(test_ds, shuffle=False, batch_size=batch_size)

# optimizer 설정
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'ln']   # no decay for bias and LayerNorm (ln)
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer
                if not any(nd in n for nd in no_decay)],
     'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer
                if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

train_logger = open(join(log_dir, 'train_log.txt'), 'a+', buffering=1)
eval_logger = open(join(log_dir, 'eval_log.txt'), 'a+', buffering=1)
print('epoch,global_step,step,mean_loss,mean_ppl,n_token_real,'
      'n_token_total,epoch_time', file=train_logger)
print('epoch,global_step,step,eval_loss,eval_ppl', file=eval_logger)

# optimizer
optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=1e-3)

# training params
global_step = 0
step = 0
epoch = 0

while True:
    model.train()
    (tr_loss, tr_ppl, mean_ppl, nb_tr_examples, nb_tr_steps) = 0.0, 0.0, 0.0, 0, 0
    n_token_real, n_token_total = 0, 0
    train_start_time_epoch = time.time()
    for inputs, _ in train_datasets:
        # activate new training mode
        inputs.to(cuda)
        loss, ppl = model(inputs, None, None, inputs)

        loss.backward()

        tr_loss += float(loss.item()) * (batch_size / inputs.shape[0])
        nb_tr_examples += inputs.size(0)
        nb_tr_steps += 1
        mean_loss = tr_loss / nb_tr_steps
        if ppl.item() < INF:
            tr_ppl += ppl.item()
        else:
            tr_ppl += mean_ppl
        mean_ppl = tr_ppl / nb_tr_steps

        n_token_total += inputs.shape[0] * inputs.shape[1]
        n_token_real += (inputs != 0).sum().item()

        # gradient update
        step += 1
