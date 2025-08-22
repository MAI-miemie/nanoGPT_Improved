"""
sample_plus.py - nanoGPT-Plus 采样脚本
- 支持交互模式/批量文件/基础生成
- 支持温度/Top-k/Top-p 采样
- 实验性：风格/情感控制（通过提示工程）
"""

import os, argparse, pickle
import torch
from model_plus import GPTPlus, GPTConfig

@torch.no_grad()
def load_model(out_dir, device='cuda'):
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'未找到检查点: {ckpt_path}')
    ckpt = torch.load(ckpt_path, map_location=device)
    config = GPTConfig(**ckpt['model_args'])
    model = GPTPlus(config).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    meta_path = os.path.join(out_dir, 'meta.pkl')
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi.get(c, 0) for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    return model, encode, decode

@torch.no_grad()
def generate_once(model, encode, decode, prompt, device='cuda', max_new_tokens=200, temperature=0.8, top_k=200, top_p=0.9):
    x = torch.tensor(encode(prompt), dtype=torch.long, device=device)[None, :]
    y = model.generate(x, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, top_p=top_p)
    return decode(y[0].tolist())

@torch.no_grad()
def interactive(out_dir, device='cuda'):
    model, encode, decode = load_model(out_dir, device)
    print("进入交互模式，输入 exit 退出。示例：style:充满诗意的语言，| 今晚的月亮\n示例：emotion:happy| 今天真好")
    while True:
        s = input('> ').strip()
        if s.lower() in ('exit', 'quit'): break
        if s.startswith('style:'):
            # 风格控制：style:风格示例|提示
            try:
                style, prompt = s[6:].split('|', 1)
                s = style + "\n" + prompt
            except Exception:
                pass
        if s.startswith('emotion:'):
            # 情感控制：emotion:happy|提示
            try:
                emo, prompt = s[8:].split('|', 1)
                emo_map = {
                    'happy': '充满欢乐和积极的情感，', 'sad': '充满悲伤的情感，', 'angry': '充满愤怒的情感，',
                    'calm': '平静祥和的情感，', 'romantic': '浪漫温馨的情感，', 'mysterious': '神秘莫测的情感，',
                }
                prefix = emo_map.get(emo.strip(), '')
                s = prefix + prompt
            except Exception:
                pass
        out = generate_once(model, encode, decode, s, device)
        print(out)

@torch.no_grad()
def batch_from_file(out_dir, batch_file, device='cuda'):
    model, encode, decode = load_model(out_dir, device)
    with open(batch_file, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]
    print(f'批量生成 {len(prompts)} 条...')
    for i, p in enumerate(prompts, 1):
        out = generate_once(model, encode, decode, p, device)
        print(f'[{i}] {p}\n{out}\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--start', default='To be, or not to be')
    ap.add_argument('--interactive', action='store_true')
    ap.add_argument('--batch_file', type=str, default=None)
    ap.add_argument('--device', type=str, default='cuda', choices=['cuda','cpu'])
    args = ap.parse_args()

    if args.interactive:
        interactive(args.out_dir, args.device)
    elif args.batch_file:
        batch_from_file(args.out_dir, args.batch_file, args.device)
    else:
        model, encode, decode = load_model(args.out_dir, args.device)
        print(generate_once(model, encode, decode, args.start, args.device))

if __name__ == '__main__':
    main()
