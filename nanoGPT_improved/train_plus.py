"""
train_plus.py - nanoGPT-Plus 训练脚本
- 线性预热 + 余弦衰减
- 混合精度 FP16/BF16
- 梯度累积
- 梯度检查点开关
- 显存实时监控与 TensorBoard 日志
"""

import os, math, time, pickle, argparse
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from model_plus import GPTPlus, GPTConfig

# 数据批次（与 nanoGPT 一致的二进制切片）
def load_meta(data_dir):
    with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    return meta['stoi'], meta['itos']

class BinDataset:
    def __init__(self, data_dir, split, block_size):
        path = os.path.join(data_dir, f'{split}.bin')
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.data = torch.tensor(data, dtype=torch.long)
        self.block_size = block_size
    def get_batch(self, batch_size, device):
        ix = torch.randint(0, len(self.data) - self.block_size, (batch_size,))
        x = torch.stack([self.data[i:i+self.block_size] for i in ix])
        y = torch.stack([self.data[i+1:i+self.block_size+1] for i in ix])
        return x.to(device), y.to(device)

# 学习率调度：线性预热 + 余弦衰减
def get_lr(it, base_lr, warmup_iters, lr_decay_iters, min_lr):
    if it < warmup_iters:
        return base_lr * it / max(1, warmup_iters)
    if it >= lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (base_lr - min_lr)

# 显存监控
def mem_gb():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated()/1024**3, torch.cuda.memory_reserved()/1024**3
    return 0.0, 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_size', type=str, default='small', choices=['tiny','small','medium','large'])
    parser.add_argument('--dataset', type=str, default='shakespeare_char')
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--out_dir', type=str, default='out-plus')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda','cpu'])
    parser.add_argument('--dtype', type=str, default='float16', choices=['float16','bfloat16','float32'])
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--max_iters', type=int, default=5000)
    parser.add_argument('--eval_interval', type=int, default=200)
    parser.add_argument('--eval_iters', type=int, default=20)
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=6e-4)
    parser.add_argument('--min_lr', type=float, default=6e-5)
    parser.add_argument('--warmup_iters', type=int, default=100)
    parser.add_argument('--lr_decay_iters', type=int, default=5000)
    parser.add_argument('--weight_decay', type=float, default=1e-1)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.95)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--grad_accum_steps', type=int, default=8)
    parser.add_argument('--use_ckpt', action='store_true', help='启用梯度检查点以省显存')
    args = parser.parse_args()

    torch.manual_seed(1337)
    device = 'cuda' if args.device=='cuda' and torch.cuda.is_available() else 'cpu'
    dtype_map = {'float16': torch.float16, 'bfloat16': torch.bfloat16, 'float32': torch.float32}
    dtype = dtype_map[args.dtype]

    data_dir = args.data_dir or os.path.join('data', args.dataset)
    os.makedirs(args.out_dir, exist_ok=True)

    # 配置模型尺寸（预设）
    presets = {
        'tiny':   dict(n_layer=4,  n_head=4,  n_embd=256, block_size=128, batch_size=32),
        'small':  dict(n_layer=6,  n_head=6,  n_embd=384, block_size=256, batch_size=16),
        'medium': dict(n_layer=12, n_head=12, n_embd=768, block_size=512, batch_size=8),
        'large':  dict(n_layer=24, n_head=16, n_embd=1024,block_size=1024,batch_size=4),
    }
    cfgd = presets[args.model_size].copy()
    if args.batch_size is not None:
        cfgd['batch_size'] = args.batch_size

    # 构建模型
    model_cfg = GPTConfig(
        block_size=cfgd['block_size'], vocab_size=50304,
        n_layer=cfgd['n_layer'], n_head=cfgd['n_head'], n_embd=cfgd['n_embd'],
        dropout=0.0, bias=True, use_gradient_checkpointing=args.use_ckpt
    )
    model = GPTPlus(model_cfg).to(device)

    # 优化器（CUDA 上启用 fused AdamW）
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters if hasattr(torch.optim, 'AdamW') else False
    use_fused = fused_available and device=='cuda'
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(args.beta1,args.beta2), weight_decay=args.weight_decay, fused=use_fused) if use_fused else torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(args.beta1,args.beta2), weight_decay=args.weight_decay)

    # 数据
    stoi, itos = load_meta(data_dir)
    train_ds = BinDataset(data_dir, 'train', cfgd['block_size'])
    val_ds   = BinDataset(data_dir, 'val',   cfgd['block_size'])

    # AMP
    scaler = torch.cuda.amp.GradScaler(enabled=(device=='cuda' and dtype==torch.float16))

    # 日志
    writer = SummaryWriter(log_dir=os.path.join(args.out_dir, 'logs'))

    iter_num = 0
    best_val = 1e9
    t0 = time.time()

    while iter_num < args.max_iters:
        # 学习率调度
        lr = get_lr(iter_num, args.learning_rate, args.warmup_iters, args.lr_decay_iters, args.min_lr)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        # 梯度累积
        optimizer.zero_grad(set_to_none=True)
        for micro in range(args.grad_accum_steps):
            X, Y = train_ds.get_batch(cfgd['batch_size'], device)
            with torch.cuda.amp.autocast(enabled=(device=='cuda' and dtype in (torch.float16, torch.bfloat16))):
                logits, loss = model(X, Y)
                loss = loss / args.grad_accum_steps
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
        # 梯度裁剪
        if args.grad_clip>0:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        # 步进
        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        iter_num += 1

        # 日志输出
        if iter_num % args.log_interval == 0:
            mem_a, mem_r = mem_gb()
            dt = time.time() - t0
            tokens_per_sec = cfgd['batch_size'] * cfgd['block_size'] * args.grad_accum_steps / max(1e-6, dt)
            print(f"it {iter_num} | lr {lr:.2e} | loss {loss.item()*args.grad_accum_steps:.4f} | tok/s {tokens_per_sec:.1f} | mem {mem_a:.2f}/{mem_r:.2f}GB")
            writer.add_scalar('loss/train', loss.item()*args.grad_accum_steps, iter_num)
            writer.add_scalar('lr', lr, iter_num)
            writer.add_scalar('memory/alloc_gb', mem_a, iter_num)
            writer.add_scalar('memory/resv_gb', mem_r, iter_num)
            t0 = time.time()

        # 评估与保存
        if iter_num % args.eval_interval == 0:
            model.eval()
            with torch.no_grad():
                losses = []
                for _ in range(args.eval_iters):
                    X, Y = val_ds.get_batch(cfgd['batch_size'], device)
                    _, l = model(X, Y)
                    losses.append(l.item())
                val_loss = sum(losses)/len(losses)
            model.train()
            print(f"eval@{iter_num} | val_loss {val_loss:.4f}")
            writer.add_scalar('loss/val', val_loss, iter_num)
            if val_loss < best_val:
                best_val = val_loss
            # 按需保存（也可始终保存最新）
            ckpt = {
                'model': model.state_dict(),
                'model_args': model_cfg.__dict__,
                'optimizer': optimizer.state_dict(),
                'iter_num': iter_num,
                'best_val': best_val,
            }
            torch.save(ckpt, os.path.join(args.out_dir, 'ckpt.pt'))

    # 最终保存
    torch.save(ckpt, os.path.join(args.out_dir, 'ckpt_final.pt'))
    # 词表也复制过去方便 sample
    with open(os.path.join(args.out_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump({'stoi': stoi, 'itos': itos}, f)

if __name__ == '__main__':
    main()
