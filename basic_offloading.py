import argparse
import logging
import os
import time
from contextlib import nullcontext
from typing import Dict

import datasets
import torch
import torch.distributed as dist
from torch.autograd.graph import saved_tensors_hooks
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import deepspeed
from huggingface_hub import login
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    GPT2Tokenizer,
    LlamaForCausalLM,
)


logger = logging.getLogger("deepspeed")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_wikitext(tokenizer, collator, max_length=None, split_size=None):
    def mask_tokens(x):
        input_ids, labels = collator.torch_mask_tokens(
            x["input_ids"], special_tokens_mask=x["special_tokens_mask"]
        )
        return {"input_ids": input_ids, "labels": labels}

    ds = datasets.load_dataset("wikitext", "wikitext-2-v1")["train"]
    ds = ds.map(
        lambda x: tokenizer(
            x["text"],
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_special_tokens_mask=True,
        ),
        batched=True,
        load_from_cache_file=False,
    )
    ds.set_format(type="torch", columns=["input_ids", "special_tokens_mask"])
    if collator.mlm:
        ds = ds.map(
            mask_tokens, remove_columns=["special_tokens_mask"], load_from_cache_file=False
        )
    else:
        ds = ds.map(
            lambda x: {"input_ids": x["input_ids"], "labels": x["input_ids"]},
            load_from_cache_file=False,
        )
    if split_size is not None:
        ds = ds.select(range(split_size))
    return ds


def pack_activation_to_cpu(tensor: torch.Tensor) -> torch.Tensor:
    # Simple synchronous copy without extra CUDA streams
    return tensor.detach().cpu()


def unpack_activation_from_cpu(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to(device, non_blocking=False)


def init_logger(args, rank: int):
    logging.basicConfig(
        level=logging.DEBUG if args.debug == 1 else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )
    for h in logger.handlers[:]:
        logger.removeHandler(h)

    log_folder = "/home/deepspeed/outputs/PyAO/logs/experiments/basic_offloading" if rank == 0 else "/root/"
    fh = logging.FileHandler(os.path.join(log_folder, args.log_file), mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)
    logger.addHandler(logging.StreamHandler())


def create_model_and_tokenizer(model_choice: str):
    if model_choice == "llama":
        model_name = "meta-llama/Llama-3.2-1B"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    elif model_choice == "opt":
        model_name = "facebook/opt-1.3b"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    elif model_choice == "gpt":
        model_name = "openai-community/gpt2-large"
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    else:
        raise ValueError(f"Unsupported model: {model_choice}")

    tokenizer.pad_token = tokenizer.eos_token
    if not hasattr(model, "model"):
        model.model = model
    return model, tokenizer, model_name


def main():
    global device

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    master_addr = os.environ["MASTER_ADDR"]
    master_port = os.environ["MASTER_PORT"]
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    parser = argparse.ArgumentParser(description="Basic activation offloading example")
    parser.add_argument("--debug", type=int, default=0)
    parser.add_argument("--use_nsys", type=int, default=0)
    parser.add_argument("--log_file", type=str, default="training.log")
    parser.add_argument("--model", type=str, default="llama", choices=["llama", "opt", "gpt"])
    parser.add_argument("--max_steps", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--do_offload", type=int, default=0)
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()

    init_logger(args, rank)
    logger.info(
        "[Program Settings] File: %s, Max Step: %d, Offload: %s, Sample Length: %d",
        args.log_file,
        args.max_steps,
        "True" if args.do_offload == 1 else "False",
        args.max_length,
    )
    logger.info(
        "RANK: %d, WORLD_SIZE: %d, MASTER_ADDR: %s, MASTER_PORT: %s",
        rank,
        world_size,
        master_addr,
        master_port,
    )

    deepspeed.init_distributed(rank=rank, world_size=world_size)

    ds_config = "/home/deepspeed/PyAO/config/ds_config_AO_stage2.json"
    hug_token = os.environ.get("HUG_TOKEN", "")
    if hug_token:
        login(hug_token)

    model, tokenizer, model_name = create_model_and_tokenizer(args.model)
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    train_dataset = load_wikitext(tokenizer, collator, max_length=args.max_length, split_size=130)

    logger.info("Initializing DeepSpeed engine...")
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config,
    )
    logger.info("DeepSpeed engine initialized for model %s", model_name)

    batch_size = model_engine.train_micro_batch_size_per_gpu()
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        pin_memory=True,
        sampler=DistributedSampler(train_dataset, num_replicas=world_size, rank=rank),
        num_workers=world_size,
    )

    offload_ctx = (
        saved_tensors_hooks(pack_activation_to_cpu, unpack_activation_from_cpu)
        if args.do_offload == 1
        else nullcontext()
    )

    full_train_time = 0.0
    done_step = 0

    with offload_ctx:
        while done_step < args.max_steps:
            for step, batch in enumerate(train_dataloader):
                st = time.time()

                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)

                if args.use_nsys == 1 and rank == 0:
                    torch.cuda.cudart().cudaProfilerStart()

                out = model_engine(input_ids=input_ids, labels=labels)
                torch.cuda.synchronize()
                loss = out.loss
                logger.info(f"[FWD] Step {done_step} loss: {loss.item():.4f}")

                model_engine.backward(loss)
                torch.cuda.synchronize()
                logger.info("[BWD] Step %d done.", done_step)

                model_engine.step()
                model_engine.zero_grad()

                excute_time = time.time() - st
                full_train_time += excute_time
                logger.info(f"[MAIN] Step Time: {excute_time:.4f}s")

                done_step += 1
                if args.use_nsys == 1 and rank == 0:
                    torch.cuda.cudart().cudaProfilerStop()
                if done_step >= args.max_steps:
                    break

    avg_time = full_train_time / max(1, done_step)
    logger.info(
        "[MAIN] Training Time: %.4fs, Avg/step: %.4fs",
        full_train_time,
        avg_time,
    )


if __name__ == "__main__":
    main()
