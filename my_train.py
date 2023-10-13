import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT

eval_interval = 200
# adamw optimizer
out_dir = 'out_question_metric'
eval_iters = 200
device = 'cuda:3' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
batch_size = 64 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 256
dataset = 'shakespeare_char'
model_args = {}

ckpt_path = os.path.join(out_dir, 'ckpt.pt')
print(f"Loading checkpoint from {ckpt_path}...")
checkpoint = torch.load(ckpt_path, map_location=device)
checkpoint_model_args = checkpoint['model_args']
# force these config attributes to be equal otherwise we can't even resume training
# the rest of the attributes (e.g. dropout) can stay as desired from command line
for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
    model_args[k] = checkpoint_model_args[k]
print(model_args)
# create the model
gptconf = GPTConfig(**model_args)
model = GPT(gptconf)
state_dict = checkpoint['model']
model.to(device)

data_dir = os.path.join('data', dataset)
print(f"Loading data from {data_dir}...")
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss_and_perplexity():
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = get_batch('val')
        # with ctx:
        logits, loss = model(X, Y)
        losses[k] = loss.item()
        # print(f"step {k}: val loss {losses[k]:.4f}, avg val loss {avg_loss:.4f}")
    print(f"losses: {losses}")
    avg_loss = losses.mean()
    perplexity = math.exp(avg_loss)
    model.train()
    return perplexity

# while True:
#     if iter_num % eval_interval == 0:
#         losses = estimate_loss_and_perplexity()
#         print(f"step {iter_num}: train loss {losses['train']['loss']:.4f}, train perplexity {losses['train']['perplexity']:.4f}, val loss {losses['val']['loss']:.4f}, val perplexity {losses['val']['perplexity']:.4f}")

est_loss = estimate_loss_and_perplexity()
print(f"val perplexity {est_loss:.4f}")