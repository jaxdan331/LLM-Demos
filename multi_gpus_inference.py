import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os
from os import path
import argparse

import torch.distributed as dist
from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler

import logging
import warnings

# 把 [WARNING] 级别的日志过滤掉
warnings.simplefilter('ignore')


def get_logger():
    # 创建Logger实例
    logger = logging.getLogger()
    logger.setLevel(logging.ERROR)

    # 创建文件Handler并设置级别为ERROR
    file_handler = logging.FileHandler('error.log')
    file_handler.setLevel(logging.ERROR)

    # 创建Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # 将Handler添加到Logger中
    logger.addHandler(file_handler)
    return logger


class JsonlDataset(Dataset):
    def __init__(self, data_dir: str):
        # 读取 jsonl 数据
        with open(data_dir, 'r') as f:
            self.data = [json.loads(line) for line in f.readlines()]
        print(f"Dataset scale: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# 多卡推理
def inference_with_multi_gpus(
        model_path: str,
        data_path: str,
        save_path: str,
        batch_size: int = 4,
):
    # 初始化进程组
    dist.init_process_group(backend='nccl')
    # 查看当前进程的编号和进程总数，有几个 GPU 就有几个进程
    local_rank, world_size = dist.get_rank(), dist.get_world_size()
    print(f"local_rank: {local_rank}, world_size: {world_size}")

    # 设置 device 很重要！！要不然后面的 inputs.to(device) 就不知道 device 是哪个了！
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # 数据并行（数据端）：给每个进程分配不同的数据
    dataset = JsonlDataset(data_path)

    sampler = DistributedSampler(dataset, shuffle=False)
    data_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        # 如果 world_size > 1 则为分布式推理，采用 auto 方式分配模型切片，否则为单卡推理，需要将模型切片分配到当前卡
        device_map='auto' if world_size > 1 else {"": local_rank},
    ).eval()

    # 有的 tokenizer 没有 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 数据并行（模型端）：在模型最后一层添一个数据并行 layer
    if torch.cuda.device_count() > 1:
        model = DataParallel(model)
        # 因为在模型最后一层添了一个数据并行 layer，所以要加上下面这个命令，模型才能做 .generate()
        model = model.module

    results = []
    # 每次取 batch_size 条数据
    # for i, queries in tqdm(enumerate(data_loader), desc='Inference', total=len(data_loader)):
    for i, queries in enumerate(data_loader):
        for j, question in enumerate(queries):
            print(f"Question-{i * batch_size + j + 1}: {question}")

        if tokenizer.chat_template is None:
            # tokenizer 没有实现聊天模板
            inputs = tokenizer(
                queries,
                return_tensors="pt",
                return_dict=True,  # 有的模型的 tokenizer 识别不出 return_dict 参数，但也不会抛出异常，只是在命令行中提醒一下
                padding=True,
            )
        else:
            messages = [[
                {"role": "system", "content": '你是一个人工智能助手。'},
                {"role": "user", "content": q}
            ] for q in queries]

            # tokenizer 实现了聊天模板
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                return_dict=True,
                padding=True,
            )

        # 把数据放到当前进程所在设备上
        inputs = inputs.to(device)

        gen_kwargs = {"max_new_tokens": 512, "do_sample": True, "top_k": 1, "pad_token_id": tokenizer.eos_token_id}

        try:
            with torch.no_grad():
                responses = model.generate(**inputs, **gen_kwargs)
                batch_results = []
                for response in responses:
                    response = response[inputs['input_ids'].shape[1]:]
                    response = tokenizer.decode(response, skip_special_tokens=True)
                    response = response.strip()
                    print(f'Response: {response}')
                    batch_results.append(response)

            results.extend({
                               'id': len(results) + 1,
                               'question': question,
                               'response': response
                           } for question, response in zip(queries, batch_results))

        except ValueError as e:
            print(f"Error: {e}")

        # 及时释放显存，防止后期 CUDA out of memory 了
        del inputs
        del responses
        torch.cuda.empty_cache()  # 必须得带上这句话

    # 最好把 encoding='utf-8' 带上，否则可能报 UnicodeEncodeError 错误
    # 这里在并行时又会有问题，因为每个进程都会写这个文件，最后我们看到的是最后一个进程写入的版本
    with open(save_path, 'a', encoding='utf-8') as f:
        for line in results:
            print(line)
            f.write(json.dumps(line, ensure_ascii=False) + '\n')


def main():
    # 初始化日志
    logger = get_logger()

    # 参数解析
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--data_path', type=str, default='data/seval_dev.jsonl')
    argparser.add_argument('--batch_size', type=int, default=8)
    argparser.add_argument('--batch_infer', action='store_true', help='Batch inference.')
    argparser.add_argument('--parallel_infer', action='store_true', help='Parallel inference.')
    argparser.add_argument('--models_dir', type=str, default='/mnt/data/djx/tfs_models')
    argparser.add_argument('--model_name', type=str, default='Qwen2-0.5B-Instruct')
    # argparser.add_argument('--models_dir', type=str, default='/mnt/data/model')
    # argparser.add_argument('--model_name', type=str, default='Qwen2-72B-Instruct')
    argparser.add_argument('--load_in_4bit', action='store_true', help='Load model in 4-bit.')
    argparser.add_argument('--load_in_8bit', action='store_true', help='Load model in 8-bit.')
    argparser.add_argument('--auto_device_map', action='store_true', help='Auto device map.')
    argparser.add_argument('--accelerate', action='store_true', help='Accelerate inference.')
    argparser.add_argument('--save_dir', type=str, default='gen_results')
    args = argparser.parse_args()
    # print("Parallel inference: {}".format(args.parallel_infer))
    print(args)

    if not path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    save_path = path.join(args.save_dir, f'{args.model_name}.jsonl')
    if path.exists(save_path):
        return

    model_path = path.join(args.models_dir, args.model_name)
    print(f"Model: {args.model_name}")

    # 读取数据
    with open(args.data_path, 'r') as f:
        data = [json.loads(line) for line in f.readlines()]
    print(f"Dataset scale: {len(data)}")

    # 加载模型
    try:
        # 正常情况下都是多卡批量推理的
        inference_with_multi_gpus(
            model_path=model_path,
            data_path=args.data_path,
            save_path=save_path,
            batch_size=args.batch_size,
        )
    except Exception as e:
        err_msg = f"Error: {e} When performing {args.model_name}"
        print(f"Error: {e}")
        logger.error(f"{err_msg}")


if __name__ == '__main__':
    # torchrun --standalone --nnodes=1 --nproc_per_node 8 ...
    # 上述命令的问题是就算模型并行和数据并行同时进行，一个模型还是会复制到所有卡上，这样就会占用很多显存
    # torchrun --standalone --nnodes=1 --nproc_per_node 1 ... 并且程序中加载模型时设置 device_map='auto'
    # 这样只把模型加载一份，放在多张卡上（这样严格来说根本没有做任何并行，只是模型分片放 GPU，做 GPU 加速），显存够的话把 batch_size 设置大一点，速度一样会快
    main()

    # 最后，我们的结论是：
    # 1.对于小模型，可以做并行计算（并行推理）—— 把模型复制 N 份，同时推理
    # 2.对于大模型，没法并行推理，因为一个模型就占满了显存，只能把模型拆开放在多张卡上，然后把 batch_size 设置大一点，这样相比不使用 GPU 来说速度会快一点
