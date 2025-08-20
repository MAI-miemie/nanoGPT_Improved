"""
改进版 nanoGPT 训练脚本 - 针对 RTX 3060 显存优化
主要特性：
1. 混合精度训练 (FP16)
2. 梯度累积
3. 显存监控
4. 自动学习率调度
5. 更好的日志记录
6. 中文注释
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model_improved import GPT, GPTConfig, create_model, get_model_config

# 设置默认张量类型为 float16（节省显存）
torch.set_default_dtype(torch.float16)

# 数据加载器
class DataLoaderLazy:
    """ 懒加载数据加载器，节省内存 """
    
    def __init__(self, filename, block_size, batch_size, device_type='cuda', device='cuda', num_workers=0):
        self.filename = filename
        self.block_size = block_size
        self.batch_size = batch_size
        self.device_type = device_type
        self.device = device
        self.num_workers = num_workers
        
        # 加载数据
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        self.data = torch.tensor(data, dtype=torch.long)
        print(f"加载数据: {filename}, 形状: {self.data.shape}")
        
    def get_batch(self, split):
        """ 获取一个批次的数据 """
        data = self.data if split == 'train' else self.data
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([data[i:i+self.block_size] for i in ix])
        y = torch.stack([data[i+1:i+self.block_size+1] for i in ix])
        if self.device_type == 'cuda':
            x, y = x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
        return x, y

def get_batch(data, block_size, batch_size, device_type='cuda', device='cuda'):
    """ 获取批次数据的便捷函数 """
    return data.get_batch('train')

# 损失估算
def estimate_loss(model, data, eval_iters, block_size, batch_size, device_type, device):
    """ 估算验证损失 """
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = get_batch(data, block_size, batch_size, device_type, device)
        with torch.no_grad():
            logits, loss = model(X, Y)
        losses[k] = loss.item()
    model.train()
    return losses.mean()

# 学习率调度器
def get_lr(it, learning_rate, warmup_iters, lr_decay_iters, min_lr):
    """ 学习率调度：线性预热 + 余弦衰减 """
    # 线性预热
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 余弦衰减到 min_lr
    if it > lr_decay_iters:
        return min_lr
    # 余弦衰减
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

# 显存监控
def get_memory_usage():
    """ 获取当前显存使用情况 """
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3, torch.cuda.memory_reserved() / 1024**3
    return 0, 0

def main():
    """ 主训练函数 """
    
    # 训练配置
    config = {
        # 模型配置
        'model_size': 'small',  # 'tiny', 'small', 'medium', 'large'
        'device_type': 'cuda',  # 'cuda' 或 'cpu'
        
        # 训练配置
        'max_iters': 5000,
        'eval_interval': 100,
        'eval_iters': 20,
        'log_interval': 10,
        'always_save_checkpoint': True,
        
        # 优化器配置
        'learning_rate': 6e-4,
        'weight_decay': 1e-1,
        'beta1': 0.9,
        'beta2': 0.95,
        'grad_clip': 1.0,
        
        # 学习率调度
        'warmup_iters': 100,
        'lr_decay_iters': 5000,
        'min_lr': 6e-5,
        
        # 混合精度训练
        'use_mixed_precision': True,
        'dtype': 'float16',  # 'float16', 'bfloat16', 'float32'
        
        # 数据配置
        'dataset': 'shakespeare_char',
        'out_dir': 'out-shakespeare-improved',
        
        # 其他
        'seed': 1337,
        'compile': False,  # 避免 Windows 上的编译器问题
    }
    
    # 根据模型大小获取配置
    model_config = get_model_config(config['model_size'], config['device_type'])
    config.update(model_config)
    
    # 设置随机种子
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])
    
    # 设备设置
    device_type = config['device_type']
    device = 'cuda' if device_type == 'cuda' and torch.cuda.is_available() else 'cpu'
    dtype = {'float16': torch.float16, 'bfloat16': torch.bfloat16, 'float32': torch.float32}[config['dtype']]
    
    print(f"使用设备: {device}")
    print(f"数据类型: {config['dtype']}")
    
    # 创建输出目录
    os.makedirs(config['out_dir'], exist_ok=True)
    
    # 创建模型
    print("创建模型...")
    model = create_model(config['model_size'], device_type, **model_config)
    model.to(device)
    
    # 打印模型信息
    print(f"模型参数数量: {model.get_num_params()/1e6:.2f}M")
    
    # 加载数据
    print("加载数据...")
    data_dir = os.path.join('data', config['dataset'])
    train_data = DataLoaderLazy(
        os.path.join(data_dir, 'train.bin'),
        config['block_size'],
        config['batch_size'],
        device_type,
        device
    )
    val_data = DataLoaderLazy(
        os.path.join(data_dir, 'val.bin'),
        config['block_size'],
        config['batch_size'],
        device_type,
        device
    )
    
    # 加载词汇表
    meta_path = os.path.join(data_dir, 'meta.pkl')
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    
    # 配置优化器
    print("配置优化器...")
    optimizer = model.configure_optimizers(
        config['weight_decay'],
        config['learning_rate'],
        (config['beta1'], config['beta2']),
        device_type
    )
    
    # 混合精度训练设置
    if config['use_mixed_precision'] and device_type == 'cuda':
        scaler = torch.cuda.amp.GradScaler()
        print("启用混合精度训练")
    else:
        scaler = None
        print("使用标准精度训练")
    
    # 学习率调度器
    def get_lr_scheduler():
        return lambda it: get_lr(
            it, config['learning_rate'], config['warmup_iters'],
            config['lr_decay_iters'], config['min_lr']
        )
    
    # 初始化训练状态
    iter_num = 0
    best_val_loss = 1e9
    
    # 设置日志
    writer = SummaryWriter(log_dir=os.path.join(config['out_dir'], 'logs'))
    
    print("开始训练...")
    print(f"目标迭代次数: {config['max_iters']}")
    print(f"批次大小: {config['batch_size']}")
    print(f"梯度累积步数: {config['gradient_accumulation_steps']}")
    print(f"有效批次大小: {config['batch_size'] * config['gradient_accumulation_steps']}")
    
    # 训练循环
    t0 = time.time()
    local_iter_num = 0
    running_mfu = -1.0
    
    while True:
        # 学习率调度
        lr = get_lr_scheduler()(iter_num)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        # 获取批次数据
        X, Y = get_batch(train_data, config['block_size'], config['batch_size'], device_type, device)
        
        # 前向传播
        if config['use_mixed_precision'] and device_type == 'cuda':
            with torch.cuda.amp.autocast():
                logits, loss = model(X, Y)
                loss = loss / config['gradient_accumulation_steps']
        else:
            logits, loss = model(X, Y)
            loss = loss / config['gradient_accumulation_steps']
        
        # 反向传播
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
            
        # 梯度累积
        if (iter_num + 1) % config['gradient_accumulation_steps'] == 0:
            # 梯度裁剪
            if config['grad_clip'] != 0.0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
            
            # 优化器步进
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            
            # 清零梯度
            optimizer.zero_grad(set_to_none=True)
            
            # 更新迭代计数
            iter_num += 1
            local_iter_num += 1
            
            # 计算 MFU
            if iter_num % 20 == 0:
                mfu = model.estimate_mfu(config['batch_size'] * config['gradient_accumulation_steps'], 1.0)
                running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        
        # 定期评估
        if iter_num % config['eval_interval'] == 0:
            losses = estimate_loss(model, val_data, config['eval_iters'], 
                                 config['block_size'], config['batch_size'], device_type, device)
            print(f"迭代 {iter_num}: 验证损失 {losses:.4f}")
            
            # 记录到 TensorBoard
            writer.add_scalar('Loss/val', losses, iter_num)
            writer.add_scalar('LR', lr, iter_num)
            if running_mfu != -1.0:
                writer.add_scalar('MFU', running_mfu, iter_num)
            
            # 保存最佳模型
            if losses < best_val_loss or config['always_save_checkpoint']:
                best_val_loss = losses
                if iter_num > 0:
                    checkpoint = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': model_config,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'config': config,
                    }
                    print(f"保存检查点到 {config['out_dir']}")
                    torch.save(checkpoint, os.path.join(config['out_dir'], 'ckpt.pt'))
        
        # 定期日志
        if iter_num % config['log_interval'] == 0:
            # 获取显存使用情况
            mem_allocated, mem_reserved = get_memory_usage()
            
            # 计算训练损失
            train_loss = loss.item() * config['gradient_accumulation_steps']
            
            # 计算时间
            dt = time.time() - t0
            tokens_per_sec = config['batch_size'] * config['block_size'] * config['gradient_accumulation_steps'] / dt
            
            print(f"迭代 {iter_num}: 训练损失 {train_loss:.4f}, "
                  f"学习率 {lr:.2e}, "
                  f"速度 {tokens_per_sec:.2f} tokens/sec, "
                  f"MFU {running_mfu:.2f}, "
                  f"显存 {mem_allocated:.2f}GB")
            
            # 记录到 TensorBoard
            writer.add_scalar('Loss/train', train_loss, iter_num)
            writer.add_scalar('Speed/tokens_per_sec', tokens_per_sec, iter_num)
            writer.add_scalar('Memory/allocated_gb', mem_allocated, iter_num)
            writer.add_scalar('Memory/reserved_gb', mem_reserved, iter_num)
        
        # 检查是否完成训练
        if iter_num >= config['max_iters']:
            break
    
    # 保存最终模型
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'model_args': model_config,
        'iter_num': iter_num,
        'best_val_loss': best_val_loss,
        'config': config,
    }
    torch.save(checkpoint, os.path.join(config['out_dir'], 'ckpt_final.pt'))
    
    # 保存词汇表
    with open(os.path.join(config['out_dir'], 'meta.pkl'), 'wb') as f:
        pickle.dump({
            'stoi': stoi,
            'itos': itos,
            'vocab_size': len(stoi),
        }, f)
    
    print(f"训练完成！总迭代次数: {iter_num}")
    print(f"最佳验证损失: {best_val_loss:.4f}")
    print(f"模型保存在: {config['out_dir']}")
    
    # 关闭 TensorBoard 写入器
    writer.close()

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='改进版 nanoGPT 训练脚本')
    parser.add_argument('--model_size', type=str, default='small', 
                       choices=['tiny', 'small', 'medium', 'large'],
                       help='模型大小')
    parser.add_argument('--max_iters', type=int, default=5000,
                       help='最大训练迭代次数')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='批次大小（覆盖默认值）')
    parser.add_argument('--learning_rate', type=float, default=6e-4,
                       help='学习率')
    parser.add_argument('--out_dir', type=str, default=None,
                       help='输出目录')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='训练设备')
    parser.add_argument('--dtype', type=str, default='float16',
                       choices=['float16', 'bfloat16', 'float32'],
                       help='数据类型')
    parser.add_argument('--compile', action='store_true',
                       help='是否使用 torch.compile（需要 MSVC）')
    
    args = parser.parse_args()
    
    # 更新配置
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.out_dir is not None:
        config['out_dir'] = args.out_dir
    if args.device != 'cuda':
        config['device_type'] = args.device
    if args.dtype != 'float16':
        config['dtype'] = args.dtype
    if args.compile:
        config['compile'] = True
    
    main()
