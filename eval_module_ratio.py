import torch
from torch.fx import symbolic_trace
from fvcore.nn import FlopCountAnalysis
from transformers import LlamaForCausalLM, AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, GPT2Model, GPT2Tokenizer
from huggingface_hub import login
from collections import defaultdict
from torch.autograd.graph import saved_tensors_hooks
from torch.utils.data import DataLoader
import torch.nn as nn
import time
import datasets
import matplotlib.pyplot as plt
import numpy as np

def load_wikitext(tokenizer, collator, max_length=None, split_size=None):
    def mask_tokens(x):
        input_ids, labels = collator.torch_mask_tokens(
            x['input_ids'], special_tokens_mask=x['special_tokens_mask']
        )
        return {"input_ids": input_ids, "labels": labels}

    ds = datasets.load_dataset("wikitext", "wikitext-2-v1")["train"]
    # tokenization
    ds = ds.map(lambda x: tokenizer(
                        x["text"],
                        max_length=max_length,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt",
                        return_special_tokens_mask=True),
                batched=True)
    ds.set_format(type="torch", columns=["input_ids", "special_tokens_mask"])
    if collator.mlm:
        ds = ds.map(mask_tokens, remove_columns=["special_tokens_mask"])
    else:
        ds = ds.map(lambda x: {"input_ids": x["input_ids"], "labels": x["input_ids"]})
    if split_size is not None:
        ds = ds.select(range(split_size))
    return ds

def main():
    # 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "meta-llama/Llama-3.2-1B"
    opt_model = "facebook/opt-1.3b"
    gpt_model = "openai-community/gpt2-large"

    # 모델 및 토크나이저 로드
    print(f"Loading model {model_name}...")
    login("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")
    # model = AutoModelForCausalLM.from_pretrained(opt_model, torch_dtype=torch.float16).to(device).eval()
    model = GPT2Model.from_pretrained(gpt_model, torch_dtype=torch.float16).to(device).eval()
    model.model = model
    # model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device).eval()
    # Disable cache to skip internal arange calls during tracing
    model.config.use_cache = False
    if hasattr(model, "model") and hasattr(model.model, "config"):
        model.model.config.use_cache = False
    # tokenizer = AutoTokenizer.from_pretrained(opt_model, use_fast=False)
    tokenizer = GPT2Tokenizer.from_pretrained(gpt_model, use_fast=False)
    # tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    train_dataset = load_wikitext(tokenizer, collator, max_length=512, split_size=64)
    batch_size = 4
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=1
    )

    # Activation size accumulator and state
    activation_sizes = {}          # module object -> total activation bytes
    tmp_act_size = 0               # bytes accumulator for current module
    current_module = None             # previous module in forward sequence
    last_module = None
    model_len = 0

    # Timing accumulators per module
    exec_time = {}      # module object -> total elapsed time
    start_times = {}    # module object -> start time

    # ── Activation offload via saved_tensors with per-module accumulation ──
    def pack_hook(saved):
        nonlocal tmp_act_size
        # normalize saved to iterable
        if isinstance(saved, torch.Tensor):
            saved_list = (saved,)
        else:
            try:
                iter(saved)
                saved_list = tuple(saved)
            except TypeError:
                saved_list = (saved,)
        # accumulate activation bytes
        for t in saved_list:
            tmp_act_size += t.numel() * t.element_size()
        return saved

    def pre_forward_hook(module, inp):
        nonlocal activation_sizes, tmp_act_size, current_module, model_len
        model_len+=1

        current_module = module
        # start timer for this module
        start_times[module] = time.time()

    def post_forward_hook(module, inp, out):
        # stop timer and accumulate exec_time
        start = start_times.pop(module, None)
        if start is not None:
            elapsed = time.time() - start
            exec_time[module] = exec_time.get(module, 0.0) + elapsed

        nonlocal activation_sizes, tmp_act_size, current_module, last_module, model
        # record accumulated size for previous module
        if current_module is not None:
            activation_sizes[current_module] = tmp_act_size
        tmp_act_size = 0
        current_module = None

        name = None
        for n, m in model.model.named_modules():
            if m is module:
                name = n
                break

        if name is not None:
            last_module = name

    # register hooks on leaf modules
    leaf_modules = [mod for name, mod in model.model.named_modules() if len(list(mod.children())) == 0]
    for mod in leaf_modules:
        mod.register_forward_pre_hook(pre_forward_hook)
        mod.register_forward_hook(post_forward_hook)

    # register saved_tensors_hooks for pack
    ctx = saved_tensors_hooks(pack_hook, lambda x: x)
    ctx.__enter__()

    # Run forward under hooks
    print("Running forward to collect activations with saved_tensors_hooks...")
    with ctx:
        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            labels   = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, labels=labels)

            if step == 0:
                break
        ctx.__exit__(None, None, None)

    # record last module's activation
    if current_module is not None:
        activation_sizes[current_module] = tmp_act_size
    else:
        activation_sizes["None"] = tmp_act_size

    # Compute execution-time to activation-byte ratio per module
    ratio_dict = {}
    for mod, act_b in activation_sizes.items():
        name = None
        # find module name
        for n, m in model.model.named_modules():
            if m is mod:
                name = n
                break
        if name is None:
            name = str(mod)
        t = exec_time.get(mod, 0.0)
        act_gb = act_b / (1024**3)
        # GB per second ratio
        ratio = act_gb / t if t > 0 else 0
        try:
            get_name = mod._get_name()
        except:
            get_name = str(mod)
        ratio_dict[name] = {
            "layer_name":get_name,
            "time_s": t,
            "activation_GB": act_gb,
            "ratio_GB_per_s": ratio
        }

    print("Per-module activation-GB to execution-time throughput (GB/s):")
    for name, info in sorted(ratio_dict.items(), key=lambda x: x[1]["ratio_GB_per_s"], reverse=True):
        print(f"{name}, {info['layer_name']}: time={info['time_s']:.6f}s, act={info['activation_GB']:.4f}GB, throughput={info['ratio_GB_per_s']:.6f}GB/s")

    # Compute IQR boundaries for throughput
    values = np.array([info["ratio_GB_per_s"] for info in ratio_dict.values()])
    names = list(ratio_dict.keys())
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1
    correction = 0.8
    upper_fence = (q3 + 1.5 * iqr)*correction

    model_cnt = 0
    for _, mod in model.model.named_modules():
        if len(list(mod.children())) == 0:
            model_cnt+=1


    # ── 기록: throughput 및 이상치 결과를 파일로 저장 ────────────
    with open("outliers_gpt2.txt", "w") as f:
        # Record the name of the last executed module
        f.write(f"Last module: {last_module}\n")
        f.write(f"Model Len: {model_cnt}\n")
        f.write("Per-module activation-GB to execution-time throughput (GB/s):\n")
        for name, info in sorted(ratio_dict.items(), key=lambda x: x[1]["ratio_GB_per_s"], reverse=True):
            f.write(f"{name}, {info['layer_name']}: time={info['time_s']:.6f}s, "
                    f"act={info['activation_GB']:.4f}GB, throughput={info['ratio_GB_per_s']:.6f}GB/s\n")
        f.write("\nOutlier modules (IQR method):\n")
        for name, value in zip(names, values):
            if value > upper_fence:
                f.write(f"{name}: throughput={value:.6f}GB/s (upper bound: {upper_fence:.6f}GB/s)\n")
    print("Saved outliers to outliers.txt")

    # ── Identify outlier modules by IQR on throughput ─────────────
    print("Outlier modules (IQR method):")
    for name, value in zip(names, values):
        if value > upper_fence:
            print(f"{name}: throughput={value:.6f}GB/s (Above upper bound {upper_fence:.6f})")

    # ── Plot throughput per module as line plot ───────────────────────
    # y축 값(throughput)만 가져오기
    ys = [info["ratio_GB_per_s"] for info in ratio_dict.values()]
    module_indices = list(range(len(ys)))

    plt.figure(figsize=(8, 4), dpi=330)
    plt.plot(module_indices, ys, marker=',', linestyle='-', linewidth="0.7")

    # 축 레이블
    plt.xlabel("Module Index")
    plt.ylabel("Throughput (GB/s)")

    # x축 눈금 레이블 숨기기
    plt.xticks([])

    # 가로 격자선 추가
    # plt.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7)

    plt.tight_layout()
    plt.savefig("throughput_vs_activation_line.png", dpi=300)
    print("Saved line chart to throughput_vs_activation_line.png")

if __name__ == "__main__":
    main()
