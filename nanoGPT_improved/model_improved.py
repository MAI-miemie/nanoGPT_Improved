"""
改进版 GPT 语言模型定义 - 针对 RTX 3060 显存优化
主要改进：
1. 轻量化模型结构，适合小显存训练
2. 混合精度训练支持
3. 梯度累积优化
4. 中文注释便于理解
5. 可配置的模型大小

参考实现：
1) OpenAI GPT-2 官方 TensorFlow 实现
2) HuggingFace transformers PyTorch 实现
"""

import math
import inspect
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    """ 层归一化，支持可选偏置项 """

    def __init__(self, ndim: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):
    """ 因果自注意力机制 - 支持 Flash Attention 加速 """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # 多头注意力的 key, query, value 投影
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # 输出投影
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # 正则化
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        
        # Flash Attention 支持（PyTorch >= 2.0）
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("警告: 使用较慢的注意力实现。Flash Attention 需要 PyTorch >= 2.0")
            # 因果掩码：确保注意力只关注左侧序列
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality

        # 计算所有头的 query, key, values 并重组维度
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # 因果自注意力计算
        if self.flash:
            # 使用 Flash Attention CUDA 内核（更快）
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True
            )
        else:
            # 手动实现注意力机制
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # 重组所有头的输出

        # 输出投影
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    """ 多层感知机 - Transformer 块的前馈网络 """

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    """ Transformer 块：注意力 + 前馈网络 """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))  # 残差连接
        x = x + self.mlp(self.ln_2(x))   # 残差连接
        return x

@dataclass
class GPTConfig:
    """ GPT 模型配置类 """
    block_size: int = 1024      # 序列最大长度
    vocab_size: int = 50304     # 词汇表大小（GPT-2 的 50257，向上取整到 64 的倍数）
    n_layer: int = 12           # Transformer 层数
    n_head: int = 12            # 注意力头数
    n_embd: int = 768           # 嵌入维度
    dropout: float = 0.0        # Dropout 率
    bias: bool = True           # 是否使用偏置项
    # 新增配置项
    use_flash_attention: bool = True  # 是否使用 Flash Attention
    use_mixed_precision: bool = True  # 是否使用混合精度训练
    gradient_accumulation_steps: int = 1  # 梯度累积步数

class GPT(nn.Module):
    """ GPT 语言模型主类 """

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        # Transformer 架构
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),  # 词嵌入
            wpe = nn.Embedding(config.block_size, config.n_embd),  # 位置嵌入
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),  # Transformer 层
            ln_f = LayerNorm(config.n_embd, bias=config.bias),  # 最终层归一化
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # 权重绑定：词嵌入和输出层共享权重（提高效率）
        self.transformer.wte.weight = self.lm_head.weight

        # 初始化权重
        self.apply(self._init_weights)
        # 对残差投影应用特殊的缩放初始化（GPT-2 论文）
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # 报告参数数量
        print(f"模型参数数量: {self.get_num_params()/1e6:.2f}M")

    def get_num_params(self, non_embedding=True):
        """
        返回模型参数数量
        non_embedding=True 时，位置嵌入参数不计入（因为它们是固定的）
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        """ 权重初始化 """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        前向传播
        idx: 输入 token 索引 (batch_size, seq_len)
        targets: 目标 token 索引，用于计算损失
        """
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"序列长度 {t} 超过最大块大小 {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # 位置索引

        # GPT 模型前向传播
        tok_emb = self.transformer.wte(idx)  # 词嵌入 (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # 位置嵌入 (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        
        # 通过所有 Transformer 层
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # 训练模式：计算损失
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # 推理模式：只计算最后一个位置的 logits（优化）
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        """
        模型手术：减小块大小（用于加载预训练模型但使用更小的序列长度）
        """
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        """
        从预训练模型加载权重
        支持: gpt2, gpt2-medium, gpt2-large, gpt2-xl
        """
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {}
        assert all(k == 'dropout' for k in override_args)
        
        try:
            from transformers import GPT2LMHeadModel
        except ImportError:
            raise ImportError("需要安装 transformers 库: pip install transformers")
            
        print(f"正在加载预训练 GPT 权重: {model_type}")

        # 根据模型类型确定配置
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),   # 124M 参数
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024),  # 350M 参数
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280),  # 774M 参数
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M 参数
        }[model_type]
        
        print("强制设置: vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257  # GPT 模型检查点总是 50257
        config_args['block_size'] = 1024   # GPT 模型检查点总是 1024
        config_args['bias'] = True         # GPT 模型检查点总是 True
        
        # 可以覆盖 dropout 率
        if 'dropout' in override_args:
            print(f"覆盖 dropout 率到 {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
            
        # 创建模型并加载权重
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = [k for k in sd.keys() if not k.endswith('.attn.bias')]  # 丢弃掩码

        # 初始化 HuggingFace transformers 模型
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # 复制权重，确保参数对齐
        sd_keys_hf = [k for k in sd_hf.keys() if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        
        assert len(sd_keys_hf) == len(sd_keys), f"键不匹配: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # 特殊处理需要转置的权重
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # 直接复制其他参数
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        配置优化器 - 支持权重衰减分组和融合 AdamW
        """
        # 收集所有需要梯度的参数
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        
        # 创建优化组：2D 参数使用权重衰减，1D 参数不使用
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"权重衰减参数张量数量: {len(decay_params)}, 参数数量: {num_decay_params:,}")
        print(f"非权重衰减参数张量数量: {len(nodecay_params)}, 参数数量: {num_nodecay_params:,}")
        
        # 创建 AdamW 优化器，在 CUDA 上使用融合版本
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"使用融合 AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """
        估算模型 FLOPS 利用率 (MFU)，以 A100 bfloat16 峰值 FLOPS 为单位
        参考 PaLM 论文附录 B
        """
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # 以 A100 bfloat16 峰值 FLOPS 的比率表示我们的 FLOPS 吞吐量
        flops_achieved = flops_per_iter * (1.0/dt)  # 每秒
        flops_promised = 312e12  # A100 GPU bfloat16 峰值 FLOPS 是 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
        """
        文本生成
        idx: 条件序列的 token 索引
        max_new_tokens: 最大生成 token 数
        temperature: 温度参数（控制随机性）
        top_k: top-k 采样
        top_p: nucleus 采样
        """
        for _ in range(max_new_tokens):
            # 如果序列上下文太长，必须在 block_size 处裁剪
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # 前向传播获取 logits
            logits, _ = self(idx_cond)
            # 取最后一步的 logits 并按温度缩放
            logits = logits[:, -1, :] / temperature
            
            # 可选的 top-k 裁剪
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # 可选的 nucleus 采样 (top-p)
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')
            
            # 应用 softmax 转换为概率
            probs = F.softmax(logits, dim=-1)
            # 从分布中采样
            idx_next = torch.multinomial(probs, num_samples=1)
            # 将采样的索引附加到运行序列并继续
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

# 预定义的模型配置（针对不同显存大小优化）
def get_model_config(model_size='small', device_type='cuda'):
    """
    获取预定义的模型配置
    model_size: 'tiny', 'small', 'medium', 'large'
    device_type: 'cuda' 或 'cpu'
    """
    configs = {
        'tiny': {
            'n_layer': 4, 'n_head': 4, 'n_embd': 256, 'block_size': 128,
            'batch_size': 32, 'gradient_accumulation_steps': 4
        },
        'small': {
            'n_layer': 6, 'n_head': 6, 'n_embd': 384, 'block_size': 256,
            'batch_size': 16, 'gradient_accumulation_steps': 8
        },
        'medium': {
            'n_layer': 12, 'n_head': 12, 'n_embd': 768, 'block_size': 512,
            'batch_size': 8, 'gradient_accumulation_steps': 16
        },
        'large': {
            'n_layer': 24, 'n_head': 16, 'n_embd': 1024, 'block_size': 1024,
            'batch_size': 4, 'gradient_accumulation_steps': 32
        }
    }
    
    if model_size not in configs:
        raise ValueError(f"不支持的模型大小: {model_size}")
    
    config = configs[model_size].copy()
    
    # 根据设备类型调整
    if device_type == 'cpu':
        config['batch_size'] = max(1, config['batch_size'] // 4)
        config['gradient_accumulation_steps'] = max(1, config['gradient_accumulation_steps'] // 2)
    
    return config

# 创建模型的便捷函数
def create_model(model_size='small', device_type='cuda', **kwargs):
    """
    创建模型的便捷函数
    """
    config_dict = get_model_config(model_size, device_type)
    config_dict.update(kwargs)
    
    config = GPTConfig(**config_dict)
    model = GPT(config)
    
    print(f"创建了 {model_size} 模型:")
    print(f"  层数: {config.n_layer}")
    print(f"  头数: {config.n_head}")
    print(f"  嵌入维度: {config.n_embd}")
    print(f"  块大小: {config.block_size}")
    print(f"  参数数量: {model.get_num_params()/1e6:.2f}M")
    
    return model
