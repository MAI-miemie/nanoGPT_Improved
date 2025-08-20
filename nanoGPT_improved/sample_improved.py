"""
改进版 nanoGPT 采样脚本 - 支持多种生成策略
主要特性：
1. 多种采样策略（Top-k, Top-p, 温度采样）
2. 可控生成（情感、风格控制）
3. 批量生成
4. 更好的输出格式
5. 中文注释
"""

import os
import pickle
import torch
import torch.nn.functional as F
from contextlib import nullcontext

from model_improved import GPT, GPTConfig

# 设置默认张量类型
torch.set_default_dtype(torch.float16)

def load_model(checkpoint_path, device='cuda'):
    """
    加载训练好的模型
    """
    print(f"正在加载模型: {checkpoint_path}")
    
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 获取模型配置
    model_args = checkpoint['model_args']
    config = GPTConfig(**model_args)
    
    # 创建模型
    model = GPT(config)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    
    print(f"模型加载完成，参数数量: {model.get_num_params()/1e6:.2f}M")
    return model

def load_vocab(meta_path):
    """
    加载词汇表
    """
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    
    return encode, decode

def generate_text(model, encode, decode, prompt, max_new_tokens=200, 
                 temperature=1.0, top_k=200, top_p=0.9, 
                 do_sample=True, num_samples=1, device='cuda'):
    """
    生成文本
    
    参数:
    - model: 训练好的模型
    - encode/decode: 编码/解码函数
    - prompt: 输入提示
    - max_new_tokens: 最大生成 token 数
    - temperature: 温度参数（控制随机性）
    - top_k: top-k 采样参数
    - top_p: nucleus 采样参数
    - do_sample: 是否使用采样（False 为贪婪解码）
    - num_samples: 生成样本数量
    - device: 设备
    """
    
    # 编码输入
    context = torch.tensor(encode(prompt), dtype=torch.long, device=device)
    context = context.unsqueeze(0).repeat(num_samples, 1)
    
    print(f"输入提示: {prompt}")
    print(f"生成参数: temperature={temperature}, top_k={top_k}, top_p={top_p}")
    print(f"最大生成长度: {max_new_tokens}")
    print("-" * 50)
    
    # 生成文本
    with torch.no_grad():
        if do_sample:
            # 采样生成
            y = model.generate(context, max_new_tokens, temperature=temperature, 
                             top_k=top_k, top_p=top_p)
        else:
            # 贪婪解码
            y = model.generate(context, max_new_tokens, temperature=1.0, 
                             top_k=None, top_p=None)
    
    # 解码输出
    generated_texts = []
    for i in range(num_samples):
        generated = decode(y[i].tolist())
        generated_texts.append(generated)
        print(f"样本 {i+1}:")
        print(generated)
        print("-" * 50)
    
    return generated_texts

def generate_with_style_control(model, encode, decode, prompt, style_prompt, 
                               max_new_tokens=200, temperature=0.8, device='cuda'):
    """
    风格控制生成（简单实现）
    通过提供风格示例来控制生成风格
    """
    print("使用风格控制生成...")
    print(f"风格示例: {style_prompt}")
    
    # 将风格示例和输入提示结合
    combined_prompt = style_prompt + "\n" + prompt
    
    return generate_text(model, encode, decode, combined_prompt, 
                        max_new_tokens, temperature, device=device)

def generate_with_emotion_control(model, encode, decode, prompt, emotion, 
                                 max_new_tokens=200, temperature=0.8, device='cuda'):
    """
    情感控制生成（简单实现）
    通过情感关键词控制生成的情感倾向
    """
    emotion_prompts = {
        'happy': '充满欢乐和积极的情感，',
        'sad': '充满悲伤和忧郁的情感，',
        'angry': '充满愤怒和激烈的情感，',
        'calm': '平静祥和的情感，',
        'romantic': '浪漫温馨的情感，',
        'mysterious': '神秘莫测的情感，',
    }
    
    if emotion in emotion_prompts:
        emotion_prompt = emotion_prompts[emotion]
        controlled_prompt = emotion_prompt + prompt
        print(f"情感控制: {emotion}")
        return generate_text(model, encode, decode, controlled_prompt, 
                            max_new_tokens, temperature, device=device)
    else:
        print(f"不支持的情感类型: {emotion}")
        return generate_text(model, encode, decode, prompt, 
                            max_new_tokens, temperature, device=device)

def batch_generate(model, encode, decode, prompts, max_new_tokens=200, 
                  temperature=1.0, top_k=200, top_p=0.9, device='cuda'):
    """
    批量生成文本
    """
    print(f"批量生成 {len(prompts)} 个样本...")
    
    results = []
    for i, prompt in enumerate(prompts):
        print(f"\n处理提示 {i+1}/{len(prompts)}: {prompt}")
        result = generate_text(model, encode, decode, prompt, max_new_tokens, 
                             temperature, top_k, top_p, device=device)
        results.append(result)
    
    return results

def interactive_generation(model, encode, decode, device='cuda'):
    """
    交互式生成模式
    """
    print("进入交互式生成模式...")
    print("输入 'quit' 退出，输入 'help' 查看帮助")
    
    while True:
        try:
            user_input = input("\n请输入提示文本: ").strip()
            
            if user_input.lower() == 'quit':
                print("退出交互模式")
                break
            elif user_input.lower() == 'help':
                print("""
帮助信息:
- 输入普通文本进行生成
- 输入 'quit' 退出
- 输入 'help' 显示此帮助
- 输入 'style:风格示例|提示文本' 进行风格控制
- 输入 'emotion:情感类型|提示文本' 进行情感控制
                """)
                continue
            elif user_input.startswith('style:'):
                # 风格控制
                parts = user_input[6:].split('|', 1)
                if len(parts) == 2:
                    style_prompt, prompt = parts
                    generate_with_style_control(model, encode, decode, prompt, 
                                              style_prompt, device=device)
                else:
                    print("格式错误，正确格式: style:风格示例|提示文本")
                continue
            elif user_input.startswith('emotion:'):
                # 情感控制
                parts = user_input[8:].split('|', 1)
                if len(parts) == 2:
                    emotion, prompt = parts
                    generate_with_emotion_control(model, encode, decode, prompt, 
                                                emotion, device=device)
                else:
                    print("格式错误，正确格式: emotion:情感类型|提示文本")
                continue
            elif user_input:
                # 普通生成
                generate_text(model, encode, decode, user_input, device=device)
            
        except KeyboardInterrupt:
            print("\n退出交互模式")
            break
        except Exception as e:
            print(f"生成出错: {e}")

def main():
    """ 主函数 """
    import argparse
    
    parser = argparse.ArgumentParser(description='改进版 nanoGPT 采样脚本')
    parser.add_argument('--out_dir', type=str, required=True,
                       help='模型输出目录')
    parser.add_argument('--start', type=str, default="To be, or not to be",
                       help='生成起始文本')
    parser.add_argument('--max_new_tokens', type=int, default=200,
                       help='最大生成 token 数')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='温度参数')
    parser.add_argument('--top_k', type=int, default=200,
                       help='Top-k 采样参数')
    parser.add_argument('--top_p', type=float, default=0.9,
                       help='Top-p 采样参数')
    parser.add_argument('--num_samples', type=int, default=1,
                       help='生成样本数量')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='设备')
    parser.add_argument('--interactive', action='store_true',
                       help='交互式生成模式')
    parser.add_argument('--greedy', action='store_true',
                       help='使用贪婪解码')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    checkpoint_path = os.path.join(args.out_dir, 'ckpt.pt')
    meta_path = os.path.join(args.out_dir, 'meta.pkl')
    
    if not os.path.exists(checkpoint_path):
        print(f"错误: 找不到检查点文件 {checkpoint_path}")
        return
    
    if not os.path.exists(meta_path):
        print(f"错误: 找不到词汇表文件 {meta_path}")
        return
    
    # 加载模型和词汇表
    model = load_model(checkpoint_path, args.device)
    encode, decode = load_vocab(meta_path)
    
    # 生成文本
    if args.interactive:
        # 交互式模式
        interactive_generation(model, encode, decode, args.device)
    else:
        # 单次生成
        do_sample = not args.greedy
        generate_text(
            model, encode, decode, args.start,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            do_sample=do_sample,
            num_samples=args.num_samples,
            device=args.device
        )

if __name__ == '__main__':
    main()
