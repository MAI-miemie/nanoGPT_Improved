# 改进版 nanoGPT - RTX 3060 优化版本

这是一个针对 RTX 3060 显存限制优化的 nanoGPT 实现，包含完整的中文注释和多种改进功能。

# 主要改进

### 1. 显存优化
- **混合精度训练 (FP16)**: 节省约 50% 显存
- **梯度累积**: 小批次累积实现大批次效果
- **预定义模型配置**: 针对不同显存大小优化
- **显存监控**: 实时监控显存使用情况

### 2. 训练优化
- **自动学习率调度**: 线性预热 + 余弦衰减
- **更好的日志记录**: TensorBoard 支持
- **多种采样策略**: Top-k, Top-p, 温度采样
- **可控生成**: 情感和风格控制

### 3. 易用性改进
- **完整中文注释**: 便于理解和修改
- **便捷函数**: 一键创建不同大小的模型
- **交互式生成**: 实时对话式文本生成
- **批量生成**: 支持多个提示同时生成

## 文件结构

```
nanoGPT/
├── model_improved.py      # 改进版模型定义
├── train_improved.py      # 改进版训练脚本
├── sample_improved.py     # 改进版采样脚本
├── README_IMPROVED.md     # 本说明文档
└── data/
    └── shakespeare_char/  # 示例数据集
```

##  安装依赖

```bash
# 激活虚拟环境
.venv\Scripts\activate

# 安装基础依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy tiktoken datasets tqdm wandb

# 安装 TensorBoard（用于训练监控）
pip install tensorboard

# 安装 transformers（用于预训练模型加载）
pip install transformers
```

##  快速开始

### 1. 准备数据

```bash
# 准备莎士比亚字符级数据
python data\shakespeare_char\prepare.py
```

### 2. 训练模型

#### 使用预定义配置（推荐）

```bash
# 小模型（适合快速验证）
python train_improved.py --model_size tiny --max_iters 1000

# 中等模型（平衡性能和速度）
python train_improved.py --model_size small --max_iters 5000

# 大模型（需要更多显存）
python train_improved.py --model_size medium --max_iters 10000
```

#### 自定义参数

```bash
# 自定义批次大小和学习率
python train_improved.py \
    --model_size small \
    --batch_size 8 \
    --learning_rate 1e-3 \
    --max_iters 3000 \
    --out_dir out-custom
```

### 3. 生成文本

#### 基础生成

```bash
# 使用训练好的模型生成文本
python sample_improved.py --out_dir out-shakespeare-improved --start "To be, or not to be"
```

#### 交互式生成

```bash
# 进入交互模式
python sample_improved.py --out_dir out-shakespeare-improved --interactive
```

#### 风格控制生成

```bash
# 在交互模式中使用风格控制
# 输入: style:充满诗意的语言，| 春天的花朵
```

#### 情感控制生成

```bash
# 在交互模式中使用情感控制
# 输入: emotion:happy| 今天天气很好
```

##  模型配置说明

### 预定义模型大小

| 模型大小 | 层数 | 头数 | 嵌入维度 | 块大小 | 参数数量 | 推荐显存 |
|---------|------|------|----------|--------|----------|----------|
| tiny    | 4    | 4    | 256      | 128    | ~1M      | 4GB      |
| small   | 6    | 6    | 384      | 256    | ~3M      | 6GB      |
| medium  | 12   | 12   | 768      | 512    | ~15M     | 8GB      |
| large   | 24   | 16   | 1024     | 1024   | ~50M     | 12GB+    |

### 针对 RTX 3060 的建议

- **快速验证**: 使用 `tiny` 模型，几分钟内完成训练
- **平衡训练**: 使用 `small` 模型，1-2 小时内完成训练
- **深度训练**: 使用 `medium` 模型，需要数小时训练

##  高级功能

### 1. 混合精度训练

自动启用 FP16 训练，显著节省显存：

```python
# 在 train_improved.py 中自动启用
config['use_mixed_precision'] = True
config['dtype'] = 'float16'
```

### 2. 梯度累积

小批次累积实现大批次效果：

```python
# 例如：batch_size=8, gradient_accumulation_steps=4
# 等效于 batch_size=32
```

### 3. 学习率调度

自动学习率调度，提高训练稳定性：

```python
# 线性预热 + 余弦衰减
warmup_iters = 100
lr_decay_iters = 5000
```

### 4. 显存监控

实时监控显存使用情况：

```bash
# 训练时会显示显存使用情况
迭代 100: 训练损失 2.3456, 学习率 6.00e-04, 速度 1250.50 tokens/sec, MFU 0.85, 显存 4.23GB
```

##  训练监控

### TensorBoard 可视化

```bash
# 启动 TensorBoard
tensorboard --logdir out-shakespeare-improved/logs

# 在浏览器中访问 http://localhost:6006
```

### 监控指标

- **损失曲线**: 训练和验证损失
- **学习率**: 学习率变化曲线
- **MFU**: 模型 FLOPS 利用率
- **显存使用**: 显存分配和保留情况
- **训练速度**: tokens/sec

##  生成策略

### 1. 温度采样

```bash
# 低温度：更确定性
python sample_improved.py --temperature 0.5

# 高温度：更随机
python sample_improved.py --temperature 1.2
```

### 2. Top-k 采样

```bash
# 限制候选 token 数量
python sample_improved.py --top_k 50
```

### 3. Top-p (Nucleus) 采样

```bash
# 累积概率阈值
python sample_improved.py --top_p 0.9
```

### 4. 贪婪解码

```bash
# 确定性生成
python sample_improved.py --greedy
```

##  故障排除

### 1. 显存不足

```bash
# 减小批次大小
python train_improved.py --batch_size 4

# 使用更小的模型
python train_improved.py --model_size tiny

# 使用 CPU 训练（慢但可行）
python train_improved.py --device cpu
```

### 2. 训练速度慢

```bash
# 启用混合精度
config['use_mixed_precision'] = True

# 使用 GPU 训练
python train_improved.py --device cuda
```

### 3. 生成质量差

```bash
# 增加训练迭代次数
python train_improved.py --max_iters 10000

# 调整生成参数
python sample_improved.py --temperature 0.8 --top_p 0.9
```

##  自定义扩展

### 1. 添加新的采样策略

在 `sample_improved.py` 中添加新的生成函数：

```python
def generate_with_custom_strategy(model, encode, decode, prompt, **kwargs):
    # 实现自定义生成策略
    pass
```

### 2. 修改模型架构

在 `model_improved.py` 中修改模型定义：

```python
class CustomGPT(GPT):
    def __init__(self, config):
        super().__init__(config)
        # 添加自定义层
        self.custom_layer = nn.Linear(config.n_embd, config.n_embd)
```

### 3. 添加新的数据集

创建新的数据准备脚本：

```python
# data/custom_dataset/prepare.py
def prepare_custom_data():
    # 实现数据预处理
    pass
```

##  贡献指南

1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

##  许可证

本项目基于 MIT 许可证开源。

##  致谢

- 原始 nanoGPT 项目：Andrej Karpathy
- PyTorch 团队
- HuggingFace transformers 库

---
