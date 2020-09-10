# BERT

모델 구조는 [참조](https://github.com/dhlee347/pytorchic-bert/blob/master/models.py#L123-L133)를 보았습니다.

## 구조

![archi2](.\img\archi2.png)

Transformer 구조에서 왼쪽(Encoder)을 사용

## model.py



### 1. LayerNorm

```python
class LayerNorm(nn.Module):
    "Tensorflow style => NHWC style??"
    def __init__(self, cfg, variance_epsilon=1e-12):
        # trainable parameters(pytorch document 참조)
        self.gamma = nn.Parameter(torch.ones(cfg.dim))
        self.beta = nn.Parameter(torch.zeros(cfg.dim))
        self.variance_epsilon = variance_epsilon
    def forward(self, x):
        u = x.mean(-1, keepdim=True)  # Expectation
        s = (x-u).pow(2).mean(-1, keepdim=True) # Variance
        x = (x-u) / torch.sqrt(s+self.variance_epsilon)
        return x*self.gamma + self.beta
```

* Encoder `Add&Norm`을 위한 Module class
* [LayerNorm in pytorch](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html)에 있는 수식과 같다.

### 2. Embeddings

![img](https://user-images.githubusercontent.com/1250095/50039788-8e4e8a00-007b-11e9-9747-8e29fbbea0b3.png)

```python
class Embeddings(nn.Module):
    "The embedding module from word, position and token_type embeddings"
    ""
    def __init__(self, cfg):
        super().__init__()
        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.dim) # token embedding
        self.pos_embed = nn.Embedding(cfg.max_len, cfg.dim) # position embedding
        self.seg_embed = nn.Embedding(cfg.n_segments, cfg.dim) # segment(token type) embedding

        self.norm = LayerNorm(cfg)
        self.drop = nn.Dropout(cfg.p_drop_hidden)

    def forward(self, x, seg): # x: (Batch, Seq_len)
        seq_len = x.size(1) #
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device)
        pos = pos.unsqueeze(0).expand_as(x) # (S,) -> (B, S)

        e = self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)
        return self.drop(self.norm(e))
```

* `token_embed` + `pos_embed` + `seg_embed`을 반환하는 class
* Embedding class return인 layerNorm + dropout는 해당 구성을 [여기에서](https://github.com/google-research/bert/blob/eedf5716ce1268e56f0a50264a88cafad334ac61/modeling.py#L520)찾을 수 있었음.

### MultiHeadSelfAttention

![multi_head_attention](.\img\multi_head_attention.png)

```python
class MultiHeaderSelfAttention(nn.Module):
    ''' Multi-Head Dot Product Attention '''
    def __init__(self, cfg):
        super().__init__()
        self.proj_q = nn.Linear(cfg.dim, cfg.dim)
        self.proj_k = nn.Linear(cfg.dim, cfg.dim)
        self.proj_v = nn.Linear(cfg.dim, cfg.dim)
        self.drop = nn.Dropout(cfg.p_drop_attn)
        self.scores = None # for visualization
        self.n_heads = cfg.n_heads

    def forward(self, x, mask):
        """
        :param x, q, k, v: (Batch_size, Seq_len, Dim)
        :param mask: (Batch_size, Seq_len)
        * split Dim into (H(n_heads), W(width of head)) ; Dim = H * W
        :return: (Batch, seq_len, Dim)
        """

        # (B, S, D)  -proj->  (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        # 그림에서 1,2,3 번
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        # n_head 수만큼 쪼개서 사용 q, k, v (batch, n_head, seq_length, head_features)
        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2) for x in [q, k, v])
        # Scale Dot Product Attention 부분(multi head인 경우 고려)
        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        if mask is not None:
            mask = mask[:, None, None, :].float()
            scores -= 10000.0 * (1.0 - mask)
        scores = self.drop(F.softmax(scores, dim=-1))
        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        h = (scores @ v).transpose(1, 2).contiguous()
        # -merge-> (B, S, D=H*W)
        h = merge_last(h, 2)
        self.scores = scores
        return h
```



