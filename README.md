##  nanoGPT_Plus

面向学习与小规模实验的 GPT 训练/生成项目，基于 Karpathy 的 nanoGPT 做了工程与易用性增强：显存优化、两卡并行、交互生成、中文注释与即开即用脚本。适合 10 分钟内完成的快速演示，也支持进一步扩展。

### 相比原版 nanoGPT 的改进
- 模型框架
  - **GPT-Plus 架构**（`model_plus.py`）：保持 GPT-2 风格，新增可选的梯度检查点、可插拔注意力接口、更明确的权重初始化与参数统计。
  - **更清晰的配置**：`GPTConfig` 聚合必要超参，采样/训练共享，便于保存与复现。
- 训练策略与工程
  - **混合精度 (FP16/BF16/FP32)** 可选；默认 FP16 以节省显存。
  - **梯度累积** 与 **余弦退火 + 线性预热** 学习率调度。
  - **两卡自动并行**：检测到多 GPU 时自动启用 `torch.nn.DataParallel`（无需额外指令），保存权重对采样脚本兼容。
  - **fused AdamW（可用时）** 与 TensorBoard 日志。
  - **数据读取修正**：`.bin` 使用 `numpy.fromfile(uint16)` 读取，更稳健更通用。
- 交互生成
  - `sample_plus.py` 支持交互、单次与批量生成；带温度、Top-k/Top-p；提供简单“风格/情感”提示工程入口。

---

## 代码结构与脚本对齐
- `nanoGPT_improved/train_plus.py` ↔ `nanoGPT_improved/sample_plus.py`
- `nanoGPT_improved/train_improved.py` ↔ `nanoGPT_improved/sample_improved.py`

建议优先使用 Plus 组（更易上手，和本 README 示例一致）。

---

## 环境安装（Windows 示例）
```cmd
python -m venv .venv
.\.venv\Scripts\activate
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy tqdm tiktoken datasets tensorboard
```

---

## 数据准备（字符级莎士比亚）
```cmd
cd nanoGPT_improved\data\shakespeare_char
python prepare.py
cd ..\..\..
```

---

## 按设备选择模型与训练命令

下表给出不同设备/显存的快速选择（均为 Plus 训练脚本，Windows/cmd）：

| 设备/显存 | 模型建议 | 典型参数 | 示例命令（两卡可选） |
|---|---|---|---|
| CPU 或 ≤4GB | tiny | `--max_iters 300~800`，`--grad_accum_steps 4` | `python nanoGPT_improved\train_plus.py --model_size tiny --dataset shakespeare_char --data_dir nanoGPT_improved\data\shakespeare_char --out_dir out-plus-char --max_iters 500 --grad_accum_steps 4 --device cpu --dtype float32` |
| 6~8GB (RTX 2060/3060) | small | `--max_iters 800~2000`，`--grad_accum_steps 4` | `set CUDA_VISIBLE_DEVICES=0,1 & python nanoGPT_improved\train_plus.py --model_size small --dataset shakespeare_char --data_dir nanoGPT_improved\data\shakespeare_char --out_dir out-plus-char --max_iters 1200 --eval_interval 200 --eval_iters 20 --log_interval 50 --grad_accum_steps 4 --device cuda --dtype float16` |
| 10~12GB | medium（或 small 更快） | 适当增大 `--max_iters` | 同上，改 `--model_size medium`，必要时调小 `--batch_size` 或 `--grad_accum_steps` |
| ≥16GB | medium/large | 适当放宽 `block_size/batch_size` | 同上，注意显存与速度平衡 |

说明：
- 两卡并行：先 `set CUDA_VISIBLE_DEVICES=0,1`，脚本会自动启用 `DataParallel`；无需改代码。
- 数据目录：Plus 训练需显式指定 `--data_dir nanoGPT_improved\data\shakespeare_char`。
- 10 分钟内演示：用 `tiny` 或 `small`，将 `--max_iters` 设为 300~1200，`--eval_interval` 稍大以减少评估开销。

---

## 训练（Plus 方案）
极简演示（更快）：
```cmd
set CUDA_VISIBLE_DEVICES=0,1
python nanoGPT_improved\train_plus.py --model_size tiny --dataset shakespeare_char --data_dir nanoGPT_improved\data\shakespeare_char --out_dir out-plus-char --max_iters 500 --eval_interval 100 --eval_iters 10 --log_interval 20 --grad_accum_steps 4 --device cuda --dtype float16
```

标准演示（平衡速度/效果）：
```cmd
set CUDA_VISIBLE_DEVICES=0,1
python nanoGPT_improved\train_plus.py --model_size small --dataset shakespeare_char --data_dir nanoGPT_improved\data\shakespeare_char --out_dir out-plus-char --max_iters 1200 --eval_interval 200 --eval_iters 20 --log_interval 50 --grad_accum_steps 4 --device cuda --dtype float16
```

TensorBoard：
```cmd
tensorboard --logdir out-plus-char\logs
```

---

## 交互与生成（Plus）
- 交互模式：
```cmd
python nanoGPT_improved\sample_plus.py --out_dir out-plus-char --interactive --device cuda
```
- 单次生成：
```cmd
python nanoGPT_improved\sample_plus.py --out_dir out-plus-char --start "To be, or not to be" --device cuda
```
- 批量生成（每行一个提示）：
```cmd
(echo 今晚的月亮真圆& echo 写一首五言绝句& echo Explain overfitting in 1 sentence) > prompts.txt
python nanoGPT_improved\sample_plus.py --out_dir out-plus-char --batch_file prompts.txt --device cuda
```
- 可选参数：`--max_new_tokens 200 --temperature 0.8 --top_k 200 --top_p 0.9`
- 风格/情感（交互里输入）：
  - `style:充满诗意的语言，| 今晚的月亮`
  - `emotion:happy| 今天真好`

若目录只有 `ckpt_final.pt`：
```cmd
copy out-plus-char\ckpt_final.pt out-plus-char\ckpt.pt
```



---

## 常见问题
- 找不到 `meta.pkl` 或 `.bin`：确认已运行 `nanoGPT_improved\data\shakespeare_char\prepare.py`，并在 Plus 训练中带上 `--data_dir`。
- 找不到 `ckpt.pt`：训练中期保存的是 `ckpt.pt`；若只有 `ckpt_final.pt`，可复制为 `ckpt.pt` 后再采样。
- 显存不足：降低 `--batch_size` 或增大 `--grad_accum_steps`；改用更小模型；或切换 BF16/FP32。
- 训练太慢：减少 `--max_iters`、降低评估频率；或用两卡（自动并行）。

---

## 许可证与致谢
- 许可证：MIT
- 致谢：Andrej Karpathy 的 nanoGPT、PyTorch 团队、HuggingFace 生态

