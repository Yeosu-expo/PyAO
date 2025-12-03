import torch
import logging
import argparse
import time
import uuid
import json
import matplotlib.pyplot as plt
from torch.autograd.graph import saved_tensors_hooks, save_on_cpu, disable_saved_tensors_hooks
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.data.distributed import DistributedSampler
import datasets
import torch.cuda.nvtx as nvtx
from torch.utils.data import DataLoader
from torch.cuda import Stream, Event
from enum import Enum
# torch.autograd.set_detect_anomaly(True) 

# --- Packages For DeepSpeed --- #
import deepspeed
from deepspeed.accelerator import get_accelerator
import torch.distributed as dist


# --- psutil for memory measurement ---
import os
import psutil

# Process handle for memory measurements
_psutil_proc = psutil.Process(os.getpid())
# Track peak RSS observed
_cpu_peak_rss = 0
peak_gpu_mem = 0

## TODO
# 1. 텐서 리스트가 메모리 ㅈㄴ 처먹고있는가?
# 2. 재계산 시점에 이미 gpu에 존재하는가?

def tag_tensor(t: torch.Tensor):
    t.__dict__['t_uid'] = uuid.uuid4().hex

def tensors_all_equal(tensor_a: torch.Tensor, tensor_b: torch.Tensor) -> bool:
    """
    두 텐서의 shape와 모든 원소값을 비교하여 동일하면 True, 아니면 False를 반환합니다.
    """
    # 1) shape가 다르면 바로 False
    if tensor_a.shape != tensor_b.shape:
        return False
    # 2) elementwise 비교를 수행한 결과가 모두 True인지 확인
    #    tensor_a == tensor_b 는 Boolean tensor를 반환
    #    torch.all 은 모든 원소가 True인지 체크
    #    .item() 으로 Python bool 로 변환
    return torch.all(tensor_a == tensor_b).item()

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
                batched=True,
                load_from_cache_file=False)
    ds.set_format(type="torch", columns=["input_ids", "special_tokens_mask"])
    if collator.mlm:
        ds = ds.map(mask_tokens,
                    remove_columns=["special_tokens_mask"],
                    load_from_cache_file=False)
    else:
        ds = ds.map(lambda x: {"input_ids": x["input_ids"], "labels": x["input_ids"]},
                    load_from_cache_file=False)
    if split_size is not None:
        ds = ds.select(range(split_size))
    return ds

def print_gpu_mem_state(where):
    global peak_gpu_mem, logger

    mem = torch.cuda.memory_allocated()
    max_mem = torch.cuda.max_memory_allocated()
    peak_gpu_mem = max(max_mem, peak_gpu_mem)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_max_memory_allocated()
    
    logger.info(f"[GPU MEM States] ({where}) Allocated = {mem/(1024**3):.4f}GB, Max Allocated = {max_mem/(1024**3):.4f}GB")


def plot_layer_timer_graphs(names, fwd_timer, bwd_timer, save_path):
    global logger

    if not names:
        if logger is not None:
            logger.warning("[Layer Timer] No layers recorded; skip plotting.")
        return

    os.makedirs(save_path, exist_ok=True)
    indices = list(range(len(names)))

    def _plot(values, title, file_name):
        fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.3), 4))
        ax.bar(indices, values)
        ax.set_xlabel("Layer (names order)")
        ax.set_ylabel("Time (s)")
        ax.set_title(title)
        ax.set_xticks(indices)
        fig.tight_layout()
        path = os.path.join(save_path, file_name)
        fig.savefig(path)
        plt.close(fig)
        if logger is not None:
            logger.info(f"[Layer Timer] Saved {title} graph to {path}")

    fwd_values = [fwd_timer[name] for name in names]
    bwd_values = [bwd_timer[name] for name in names]

    _plot(fwd_values, "Average Forward Time", "layer_forward_time.png")
    _plot(bwd_values, "Average Backward Time", "layer_backward_time.png")


def save_layer_timer_stats(names, fwd_timer, bwd_timer, save_path):
    global logger

    if not names:
        if logger is not None:
            logger.warning("[Layer Timer] No layers recorded; skip saving stats.")
        return

    os.makedirs(save_path, exist_ok=True)
    stats = {
        "forward": {name: float(fwd_timer[name]) for name in names},
        "backward": {name: float(bwd_timer[name]) for name in names},
    }
    output_file = os.path.join(save_path, "layer_timer_stats.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    if logger is not None:
        logger.info(f"[Layer Timer] Saved stats to {output_file}")

def init_global():
    global quan_cnt, offload_event, dispatcher_uid, storage, prefetch_count, module_memory, current_module, uid, args, pre_hook_handler, post_hook_handler, model, pack_ctx, pack_ctx_for_offload, pack_ctx_for_discard, pack_ctx_for_recompute, selected_ops, cpu_refer_dict, module_sequence, prefetch_offloaded_pack, transfer_event, is_backward
    init_tracer()
    for k in list(storage.keys()):
        storage[k].clear()
    storage.clear()
    module_memory.clear()
    cpu_refer_dict.clear()
    module_sequence.clear()
    prefetch_offloaded_pack.clear()
    transfer_event.clear()
    offload_event.clear()
    current_module = None
    is_backward=False
    prefetch_count = 0
    uid = -1
    dispatcher_uid = 0
    quan_cnt = 0

    # leaf_modules = [(name, mod) for name, mod in model.model.named_modules() if len(list(mod.children())) == 0]
    # for n, m in leaf_modules:
    #     if args.do_recompute == 1:
    #         if n in selected_ops and n != last_layer:
    #             logger.debug(f"[Hook Register] {n} is registered as recomputation module.")
    #             pre_hook_handler[n] = m.register_forward_pre_hook(pre_fwd_hook_for_discard)
    #             post_hook_handler[n] = m.register_forward_hook(post_fwd_hook_for_discard)

    try:
        pack_ctx.__exit__(None, None, None)
    except:
        pass
    try:
        pack_ctx_for_offload.__exit__(None, None, None)
    except:
        pass
    try:
        pack_ctx_for_discard.__exit__(None, None, None)
    except:
        pass
    try:
        pack_ctx_for_recompute.__exit__(None, None, None)
    except:
        pass

    pack_ctx_for_offload = saved_tensors_hooks(pack_to_cpu, unpack_from_cpu)
    pack_ctx             = saved_tensors_hooks(pack_hook, unpack_hook)
    pack_ctx_for_discard = saved_tensors_hooks(pack_hook_for_discard, unpack_hook_for_discard)
    pack_ctx_for_recompute = saved_tensors_hooks(pack_hook_for_recompute, unpack_hook_for_recompute)

    if args.do_offload == 1:
        pack_ctx_for_offload.__enter__()
    else:
        pack_ctx.__enter__()

def init_tracer():
    global cpu_pack_dict, input_tracer
    cpu_pack_dict.clear()
    input_tracer.clear()

def init_timer(module_names):
    global layer_acc_fwd_timer, layer_acc_bwd_timer
    for name in module_names:
        layer_acc_fwd_timer[name] = 0.0
        layer_acc_bwd_timer[name] = 0.0

current_module = None
uid = 0

from typing import Dict, Any, Tuple, List
from collections import defaultdict
from torch.utils._python_dispatch import TorchDispatchMode

# --- Global Variable --- #
pack_ctx = None
pack_ctx_for_offload = None
pack_ctx_for_discard = None
pack_ctx_for_recompute = None
module_memory: Dict[Any, Tuple] = defaultdict() # key: module_name, value: input_id
input_tracer: Dict[Any, List[Tuple[Any, Tuple[Any, List[Tuple[Any, Any]]]]]] = defaultdict() # key: out_id, value: list(uid, (func, list(input, shape)))
cpu_pack_dict: Dict[Any, Tuple[Any, Any]] = defaultdict() # key: gpu_tensor.t_uid, value: cpu_tensor_refer
cpu_refer_dict: Dict[Any, Tuple[torch.Tensor, bool]] = defaultdict() # key: cpu_refer, value: gpu_refer
storage: Dict[Any, Dict[Any, Tuple[Any, Any]]] = defaultdict(lambda: defaultdict())
pre_hook_handler: Dict[Any, Any] = defaultdict() # key: module_name, value: handler
post_hook_handler: Dict[Any, Any] = defaultdict() # key: module_name, value: handler
prefetch_offloaded_pack: Dict[Any, List] = defaultdict() # key: module_name, value: offloaded and packed tensor
module_sequence = list()
uid = -1
dispatcher_uid = 0
args = None
selected_ops = []
device = None
fwd_dispatcher = None
model_name = None
last_layer = None
prefetch_size = 0
model_len = 0
is_backward = False
transfer_stream = None
transfer_event: Dict[Any, Event] = defaultdict()
offload_event: Dict[Any, List[Event]] = defaultdict()
device_module = getattr(torch, "cuda", torch.cuda)
pin_memory = True
prefetch_count = 0
offload_count = 0
model = None
logger:logging.Logger = None
do_quantization = 0
do_recompute = 0
quantization_bits = 8
quan_cnt = 0
layer_timer = defaultdict(float)
layer_acc_fwd_timer = defaultdict(float)
layer_acc_bwd_timer = defaultdict(float)


# -- Notes --- #

# --- Dispatcher --- #
class ForwardDispatchMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, argss=(), kwargs=None):
        global input_tracer, args, dispatcher_uid

        if kwargs is None:
            kwargs = {}
        
        extra_msg = ""
        if args.debug == 1:
            infos: List[Tuple[Any, Tuple[Any, Any]]] = list()
            for arg in argss:
                if isinstance(arg, torch.Tensor):
                    size_gb = arg.numel() * arg.element_size() / (1024**3)
                    infos.append((id(arg), (size_gb, arg.shape)))
                # elif isinstance(arg, (list, tuple)) and arg and all(isinstance(x, torch.Tensor) for x in arg):
                #     shape_list = [str(t.shape) for t in arg]
                #     shape_list = ', '.join(shape_list)
                #     id_list = [str(id(t)) for t in arg]
                #     id_list = ', '.join(id_list)
                #     size_list = [str(t.numel()*t.element_size()/(1024**3)) for t in arg]
                #     size_list = ', '.join(size_list)
                #     infos.append((id_list, (size_list, shape_list)))
                else:
                    infos.append((arg, (1, 2)))
            
            for idx, info in enumerate(infos):
                tensor_id, (size_gb, shape) = info
                if isinstance(size_gb, float):
                    extra_msg += f", ({shape}) id[{idx}]: {tensor_id}, size[{idx}]: {size_gb:.6f}GB"
                else:
                    extra_msg += f", ({shape}) id[{idx}]: {id(tensor_id)}, size[{idx}]: {size_gb}GB"

        out = func(*argss, **kwargs)

        tensor_list = list()
        for arg in argss:
            if isinstance(arg, torch.Tensor):
                tensor_list.append((id(arg), arg.shape))
            # elif isinstance(arg, (list, tuple)) and arg and all(isinstance(x, torch.Tensor) for x in arg):
            #     shape_list = [t.shape for t in arg]
            #     id_list = [id(t) for t in arg]
            #     tensor_list.append((id_list, shape_list))
            else:
                tensor_list.append((arg, None))
        if input_tracer.get(id(out), None) is None:
            input_tracer[id(out)] = [(dispatcher_uid, (func, tensor_list))]
        else:
            input_tracer[id(out)].append((dispatcher_uid, (func, tensor_list)))
        try:
            logger.debug(f"[Dispatcher][{dispatcher_uid}] ({func})out id: {id(out)}, shape: {out.shape}" + extra_msg)
        except:
            logger.debug(f"[Dispatcher][{dispatcher_uid}] ({func})out id: {id(out)}, shape: None" + extra_msg)
        dispatcher_uid+=1
        return out

# --- Pack Hook --- #
def pack_hook(saved):
    logger.debug(f"[Pack Hook] PACKED. id: {id(saved)}")
    return saved

def unpack_hook(saved):
    logger.debug("[Unpack Hook] UNPACKED.")
    return saved

# 여기서 storage 키를 넘겨줌 순
def pack_hook_for_discard(saved):
    global uid, current_module, transfer_stream, logger
    
    saved = saved.detach()
    if isinstance(saved, torch.Tensor):
        # Copy shape into a plain tuple to avoid holding a reference
        shape = saved.shape
    else:
        shape = None
    storage[current_module][uid+1] = (shape, None)
    
    # with torch.cuda.stream(transfer_stream):
    if args.debug == 1:
        size_gb = saved.numel() * saved.element_size() / (1024**3)
        logger.debug(f"[Pack Hook Discard] Discarded {size_gb:.6f}GB.")
    del saved
    uid+=1
    return uid

def unpack_hook_for_discard(saved_uid):
    global storage, current_module, logger, args

    shape, out = storage[current_module].pop(saved_uid)
    if args.debug == 1:
        size_gb = out.numel() * out.element_size() / (1024**3)
        logger.debug(f"[Unpack Hook Discard] Get Recomputed {size_gb:.6f}GB.")
    return out

def pack_hook_for_recompute(saved):
    global storage, current_module, args, logger

    if isinstance(saved, torch.Tensor):
        shape = saved.shape
    else:
        shape = None

    for saved_uid, (saved_shape, saved_tensor) in storage[current_module].items():
        if saved_tensor is not None:
            continue
        if shape == saved_shape:
            storage[current_module][saved_uid] = (saved_shape, saved)
            if args.debug == 1:
                size_gb = saved.numel() * saved.element_size() / (1024**3)
                logger.debug(f"[Pack Hook Recompute] Saved {size_gb:.6f}GB")
            return saved_uid

    raise RuntimeError("[Pack Hook Recompute] There is no matched with tensor's shape.")

def unpack_hook_for_recompute(saved):
    logger.error("[Recompute Unpack Hook] It will be never runned.")
    return torch.tensor([1, 2, 3])

def pack_to_cpu(tensor: torch.Tensor) -> tuple[torch.device, tuple[torch.cuda.Event, torch.Tensor]]:
    global quan_cnt, quantization_bits, do_quantization, pin_memory, device_module, cpu_pack_dict, prefetch_offloaded_pack, prefetch_size, transfer_stream, args, logger, dispatcher_uid

    if not pin_memory:
        size_gb = tensor.numel() * tensor.element_size() / (1024**3)
        logger.debug(f"[Pack Hook CPU] Offload {size_gb:.6f} GB")
        return (tensor.device, tensor.cpu())
    
    gpu_id = id(tensor)
    is_half = False
    with torch.cuda.stream(transfer_stream):
        tensor_detached = tensor.detach()
        if do_quantization == 1 and tensor_detached.is_floating_point():
            quan_cnt+=1
            if args.debug == 1 and torch.isnan(tensor_detached).any():
                logger.error(f"[Pack Hook CPU] Tensor has nan after clone.")
            tensor_detached, is_half = symmetric_quantize(tensor_detached, quantization_bits)
            
            packed = torch._empty_affine_quantized(
                tensor_detached.size(),
                dtype=torch.qint8,
                scale=tensor_detached.q_scale(),
                zero_point=tensor_detached.q_zero_point(),
                pin_memory=True
            )
            # Copy integer representation into the quantized buffer
            packed.copy_(tensor_detached, non_blocking=True)
        else:
            # tensor_copy = tensor_detached.contiguous()
            if args.debug == 1 and torch.isnan(tensor_detached).any():
                logger.error(f"[Pack Hook CPU] Tensor has nan after clone.")

            packed = torch.empty(
                tensor_detached.size(),
                dtype=tensor_detached.dtype,
                layout=tensor_detached.layout,
                pin_memory=True,
            )
            
            packed.copy_(tensor_detached, non_blocking=True)
        # del tensor_detached
    
    # 원본 텐서 NaN 체크 (디버깅용)
    if args.debug == 1 and torch.isnan(tensor).any():
        logger.error(f"[Pack Hook CPU] Original tensor has nan.")
    
    if args.do_recompute == 1:
        cpu_pack_dict[gpu_id] = (packed, dispatcher_uid)
        dispatcher_uid+=1
    else:
        cpu_pack_dict[gpu_id] = (packed, -1)
    cpu_refer_dict[packed] = (None, is_half)
    if prefetch_size > 0:
        try:
            prefetch_offloaded_pack[str(current_module)].append(packed)
        except:
            prefetch_offloaded_pack[str(current_module)] = list()
            prefetch_offloaded_pack[str(current_module)].append(packed)

    if args.debug == 1:
        size_gb = packed.numel() * packed.element_size() / (1024**3)
        logger.debug(f"[Pack Hook CPU] Offload {size_gb:.6f} GB, gpu_id: {gpu_id}, cpu_id: {id(packed)}, shape: {packed.shape}, pack_uid: {dispatcher_uid-1}")
    
    return (tensor.device, packed, is_half)

# The packed value is now a tuple: (device, (evt, buf))
def unpack_from_cpu(packed: tuple[torch.device,torch.Tensor, bool]) -> torch.Tensor:
    global cpu_refer_dict, logger, args
    device, buf, is_half = packed
    gpu_refer, is_half = cpu_refer_dict.pop(buf, (None, None))
    if gpu_refer is None:
        # If the offloaded buffer contains any NaNs, log or handle as needed
        if args.debug == 1 :
            if torch.isnan(buf).any():
                logger.error(f"[Unpack Hook CPU] NaN detected in buffer id={id(buf)}, shape={buf.shape}, device: {buf.device}")
            logger.debug(f"[Unpack Hook CPU] id: {id(buf)} Fetch.")
        if buf.is_quantized:
            tensor = buf.to(device, non_blocking=False)
            return symmetric_dequantize(tensor, is_half)
        else:
            return buf.to(device, non_blocking=False)
    else:
        return gpu_refer

# --- Full Module Hook --- #
def pre_fwd_hook_full(module, inp):
    global selected_ops, is_backward, do_recompute, layer_timer

    name, is_last = get_module_name(module)
    layer_timer[name] = time.time()
    if do_recompute == 1:
        if name in selected_ops and not is_last and not is_backward:
            pre_fwd_hook_for_discard(module, inp) # for excluded layers from offloading strategy
    else:
        pre_fwd_hook(module, inp)

def post_fwd_hook_full(module, inp, out):
    global selected_ops, do_recompute, layer_timer, layer_acc_fwd_timer

    name, is_last = get_module_name(module)
    layer_timer[name] = time.time() - layer_timer[name]
    layer_acc_fwd_timer[name] += layer_timer[name]
    if do_recompute == 1:
        if name in selected_ops and not is_last and not is_backward:
            post_fwd_hook_for_discard(module, inp, out)
    else:
        post_fwd_hook(module, inp, out)

def pre_bwd_hook_full(module, inp):
    global selected_ops, do_recompute, layer_timer

    name, is_last = get_module_name(module)
    layer_timer[name] = time.time()
    if do_recompute == 1:
        if name in selected_ops and not is_last:
            pre_bwd_hook_for_recompute(module, inp)
    else:
        pre_bwd_hook(module, inp)

def post_bwd_hook_full(module, inp, out):
    global selected_ops, do_recompute, layer_timer, layer_acc_bwd_timer

    name, is_last = get_module_name(module)
    layer_timer[name] = time.time() - layer_timer[name]
    layer_acc_bwd_timer[name] += layer_timer[name]
    if do_recompute == 1:
        if name in selected_ops and not is_last:
            post_bwd_hook_for_recompute(module, inp, out)
    else:
        post_bwd_hook(module, inp, out)

# --- Module Hook --- #
def pre_fwd_hook(module, inp):
    global prefetch_offloaded_pack, offload_event, current_module, pack_ctx, pack_ctx_for_offload, args, module_sequence, prefetch_size, model_len, is_backward, transfer_stream, transfer_event, logger
    
    if not is_backward:
        if args.debug == 1:
            st = time.time()
            transfer_stream.synchronize()
            ed = time.time() - st
            logger.debug(f"[Pre FWD Hook] Offloading Overhead: {ed:.6f}s.")
            print_gpu_mem_state("[Pre FWD HOOK]")
        else:
            transfer_stream.synchronize()


    name, is_last = get_module_name(module)
    current_module = name

    if not is_backward:
        offload_event[current_module] = list()

    if not name in module_sequence:
        module_sequence.append(name)

    now_idx = module_sequence.index(name)
    if  now_idx+2 == model_len and prefetch_size == 3 and not is_backward:
        prefetch_tensors_in_fwd(name)
            
    # logging
    extra_msg = ""
    if args.debug == 1:
        infos: List[Tuple[Any, Any]] = list()
        for input in inp:
            if isinstance(input, torch.Tensor):
                size_gb = input.numel() * input.element_size() / (1024**3)
                infos.append((id(input), size_gb))
        
        for idx, info in enumerate(infos):
            tensor_id, size_gb = info
            extra_msg += f", id[{idx}]: {tensor_id}, size[{idx}]: {size_gb:.6f}GB"

    # logic of last layer no offloading
    if not is_backward and args.do_offload == 1 and is_last:
        pack_ctx_for_offload.__exit__()
        pack_ctx.__enter__()

    if not is_backward and prefetch_offloaded_pack.get(str(current_module)) is None:
        prefetch_offloaded_pack[str(current_module)] = list()

    logger.debug(f"[Pre FWD Hook] {name}" + extra_msg)
    init_tracer()


def pre_fwd_hook_for_discard(module, inp):
    global prefetch_offloaded_pack, args, is_backward, dispatcher_uid, logger, current_module, pack_ctx, pack_ctx_for_offload, pack_ctx_for_discard, module_memory, cpu_pack_dict, input_tracer, module_sequence, prefetch_size, model_len, transfer_event, transfer_stream

    if not is_backward:
        if args.debug == 1:
            st = time.time()
            transfer_stream.synchronize()
            ed = time.time() - st
            logger.debug(f"[Pre FWD Hook] Offloading Overhead: {ed:.6f}s.")
        else:
            transfer_stream.synchronize()

    name, is_last = get_module_name(module)
    current_module = name
    if not name in module_sequence:
        module_sequence.append(name)

    now_idx = module_sequence.index(name)
    if  now_idx+2 == model_len and prefetch_size == 3 and not is_backward:
        prefetch_tensors_in_fwd(name)
    
    if prefetch_offloaded_pack.get(str(current_module)) is None:
        prefetch_offloaded_pack[str(current_module)] = list()

    if args.do_offload == 1:
        pack_ctx_for_offload.__exit__()
    else:
        pack_ctx.__exit__()
    pack_ctx_for_discard.__enter__()
    
    # Multi-input trace: BFS over all tensor inputs to collect CPU references
    # and record which function and argument index produced them.
    trace_queue = []
    extra_msg = ""
    for idx, x in enumerate(inp):
        if isinstance(x, torch.Tensor):
            trace_queue.append((None, None, id(x), x.shape, 0, dispatcher_uid))
            extra_msg += f"id: {id(x)}, shape: {x.shape}, idx: {idx}"
    logger.debug(f"[Pre FWD Hook Discard] Input args -> " + extra_msg)
    func_list = []
    inp_cpu_refer = []
    # -- 인풋 복원 -- #
    while trace_queue:
        producer_func, producer_arg_idx, cur_id, cur_shape, pass_pack_cnt, now_dis_uid = trace_queue.pop(0)
        if cur_shape == None:
            # in this case, cur_id -> not tensor arg
            logger.debug(f"[Pre FWD Hook Discard] arg appended. func: {producer_func}, func_idx: {producer_arg_idx}, no_tensor_id: {cur_id}")
            inp_cpu_refer.append((producer_func, producer_arg_idx, cur_id)) 
            continue
        if pass_pack_cnt != 0:
            tmp = None
        else:
            tmp = cpu_pack_dict.get(cur_id, None) # 저장안된 중간텐서임
        if tmp is None:
            if cur_id not in input_tracer:
                logger.error(f"[Pre FWD Hook Discard] Failed to trace input for id {cur_id}")
                raise RuntimeError(f"[Pre FWD Hook Discard] Failed to trace input for id {cur_id}")
            while True:
                if len(input_tracer[cur_id]) == 1:
                    dis_uid, (func, tensor_list) = input_tracer.pop(cur_id).pop(0)
                    if dis_uid <= now_dis_uid:
                        break
                else:
                    dis_uid, (func, tensor_list) = input_tracer.get(cur_id).pop(-1)
                    if dis_uid <= now_dis_uid:
                        break
            # Record the function and note its index in func_list
            func_list.append(func)
            logger.debug(f"[Pre FWD Hook Discard] func appended. func: {func}")
            func_idx = len(func_list) - 1
            # Enqueue all inputs of that op for further tracing, using func_list index
            for arg_id, arg_shape in tensor_list: # it might be arg(not tensor), none
                trace_queue.append((func, func_idx, arg_id, arg_shape, 0, dis_uid))
            continue

        cpu_tensor, pack_uid = tmp # 저장안되는 중간텐서임
        if cpu_tensor.shape != cur_shape:
            trace_queue.append((producer_func, producer_arg_idx, cur_id, cur_shape, 1, now_dis_uid))
            continue
        logger.debug(f"[Pre FWD Hook Discard] cpu_pack_dict key: {cur_id}, tensor: {id(cpu_tensor)}, shape: {cpu_tensor.shape}, pack_uid: {pack_uid}")
        if cur_id not in input_tracer:
            logger.debug(f"[Pre FWD Hook Discard] arg appended. func: {producer_func}, func_idx: {producer_arg_idx}, cpu_tensor_id: {id(cpu_tensor)}")
            inp_cpu_refer.append((producer_func, producer_arg_idx, cpu_tensor))
            continue
        dis_uid, (func, tensor_list) = input_tracer.get(cur_id)[-1]
        if dis_uid > pack_uid and now_dis_uid > dis_uid:
            while True:
                if len(input_tracer[cur_id]) == 1:
                    dis_uid, (func, tensor_list) = input_tracer.pop(cur_id).pop(0)
                    if dis_uid <= now_dis_uid:
                        break
                else:
                    dis_uid, (func, tensor_list) = input_tracer.get(cur_id).pop(-1)
                    if dis_uid <= now_dis_uid:
                        break
            func_list.append(func)
            logger.debug(f"[Pre FWD Hook Discard] func appended. func: {func}")
            func_idx = len(func_list) - 1
            for arg_id, arg_shape in tensor_list:
                trace_queue.append((func, func_idx, arg_id, arg_shape, 0, dis_uid))
            continue
        else:
            if cpu_tensor.shape != cur_shape:
                logger.error(f"[Pre FWD Hook Discard] shape doesn't match. pack_shape: {cpu_tensor.shape}, cur_shape: {cur_shape}")
                raise RuntimeError(f"[Pre FWD Hook Discard] shape doesn't match. pack_shape: {cpu_tensor.shape}, cur_shape: {cur_shape}")
            logger.debug(f"[Pre FWD Hook Discard] arg appended. func: {producer_func}, func_idx: {producer_arg_idx}, cpu_tensor_id: {id(cpu_tensor)}")
            inp_cpu_refer.append((producer_func, producer_arg_idx, cpu_tensor))
    # ----------- #
    if len(inp_cpu_refer) != 0:
        extra_msg = ""
        if args.debug == 1:
            for executed_func, func_idx, cpu_refer in inp_cpu_refer:
                extra_msg += f", func: {executed_func}, func_idx: {func_idx}, cpu_id: {id(cpu_refer)}"
        logger.debug(f"[Pre FWD Hook Discard] Success to trace input" + extra_msg)
        logger.debug(f"[Pre FWD Hook Discard] func list: {func_list}")
    else:
        logger.error(f"[Pack FWD Hook Discard] Failed to trace input.")
        raise RuntimeError(f"[Pack FWD Hook Discard] Failed to trace input.")

    module_memory[current_module] = (module, inp_cpu_refer, func_list)

    extra_msg = ""
    if args.debug == 1:
        infos: List[Tuple[Any, Any]] = list()
        for input in inp:
            if isinstance(input, torch.Tensor):
                size_gb = input.numel() * input.element_size() / (1024**3)
                infos.append((id(input), size_gb, input.shape))
            else:
                infos.append((id(input), 0, input))
        
        for idx, info in enumerate(infos):
                tensor_id, size_gb, in_shape = info
                extra_msg += f", (shape: {in_shape})id[{idx}]: {tensor_id}, size[{idx}]: {size_gb:.6f}GB"

    logger.debug(f"[Pre FWD Hook Discard] {current_module}"+ extra_msg)
    init_tracer()

def post_fwd_hook_for_discard(module, inp, out):
    global current_module, pack_ctx, pack_ctx_for_offload, pack_ctx_for_discard, module_sequence, model_len, prefetch_size, transfer_event, transfer_stream, logger

    name, is_last = get_module_name(module)
    current_module = name

    now_idx = module_sequence.index(name)
    if  now_idx+2 == model_len and prefetch_size == 3 and not is_backward:
        transfer_stream.synchronize()

    pack_ctx_for_discard.__exit__()
    if args.do_offload == 1:
        pack_ctx_for_offload.__enter__()
    else:
        pack_ctx.__enter__()

    logger.debug(f"[Post FWD Hook Discard] {current_module}")
    # init_tracer()
        

def post_fwd_hook(module, inp, out):
    global current_module, module_sequence, model_len, prefetch_size, is_backward, transfer_event, transfer_stream, logger
    
    name, is_last = get_module_name(module)
    now_idx = module_sequence.index(name)
    if  now_idx+2 == model_len and prefetch_size == 3 and not is_backward:
        transfer_stream.synchronize()

    if is_last:
        logger.debug(f"[Post FWD Hook] {name}, and it's last layer.")
    else:
        logger.debug(f"[Post FWD Hook] {name}")
    # init_tracer()

def pre_bwd_hook(module, inp):
    global current_module, prefetch_size, transfer_stream, transfer_event, logger
    
    name, is_last = get_module_name(module)
    current_module = name

    if prefetch_size > 0:
        prefetch_tensors_in_bwd(name)

    logger.debug(f"[Pre BWD Hook] {name}")

def post_bwd_hook(module, inp, out):
    global current_module, args, pack_ctx, pack_ctx_for_offload, transfer_event, transfer_stream, logger

    name, is_last = get_module_name(module)

    if args.do_offload == 1 and is_last:
        pack_ctx.__exit__()
        pack_ctx_for_offload.__enter__()

    if prefetch_size > 0:
        transfer_stream.synchronize()

    current_module = name
    logger.debug(f"[Post BWD Hook] {name}")

def pre_bwd_hook_for_recompute(module, inp):
    global logger, current_module, pack_ctx, pack_ctx_for_offload, pack_ctx_for_discard, pack_ctx_for_recompute, module_memory, pre_hook_handler, post_hook_handler, prefetch_size, transfer_stream, transfer_event

    name, is_last = get_module_name(module)
    current_module = name

    if prefetch_size > 0:
        prefetch_tensors_in_bwd(name)
    
    logger.debug(f"[Pre BWD Hook Recompute] {name}")

    if args.do_offload == 1:
        pack_ctx_for_offload.__exit__()
    else:
        pack_ctx.__exit__()
    saved_module, inp_cpu_refer, func_list = module_memory.pop(current_module)
    recomputed_input = reconstruct_input(inp_cpu_refer, func_list)
    # Ensure recomputed input participates in autograd
    recomputed_input = recomputed_input.requires_grad_()
    size_gb = recomputed_input.numel() * recomputed_input.element_size() / (1024**3)
    logger.debug(f"[Pre BWD Hook Recompute] Recomputed Input {size_gb:.4f}GB")
    pack_ctx_for_recompute.__enter__()
    # pre_hook_handler[current_module].remove()
    # post_hook_handler[current_module].remove()
    # # Temporarily register forward hooks and keep handles
    # handle_pre = saved_module.register_forward_pre_hook(pre_fwd_hook)
    # handle_post = saved_module.register_forward_hook(post_fwd_hook)
    with torch.enable_grad():
        out = saved_module(recomputed_input)
    del recomputed_input
    # # Remove the temporary hooks immediately after forward
    # handle_pre.remove()
    # handle_post.remove()
    
    pack_ctx_for_recompute.__exit__()
    pack_ctx_for_discard.__enter__()
    current_module = name

def post_bwd_hook_for_recompute(module, inp, out):
    global current_module, pack_ctx, pack_ctx_for_offload, pack_ctx_for_discard, transfer_event, logger

    name, is_last = get_module_name(module)
    current_module = name

    pack_ctx_for_discard.__exit__()
    if args.do_offload == 1:
        pack_ctx_for_offload.__enter__()
    else:
        pack_ctx.__enter__()

    if prefetch_size > 0:
        transfer_stream.synchronize()

    logger.debug(f"[Post BWD Hook Recompute] {current_module}")

# --- Assistance Function --- #
def prefetch_tensors_in_bwd(module_name):
    global prefetch_size, prefetch_offloaded_pack, cpu_refer_dict, module_sequence, device, prefetch_count, logger, transfer_stream, args

    if len(module_sequence) > model_len:
        raise RuntimeError(f"[Prefetch Tensor Func] module sequence exceed model len. {len(module_sequence)} {module_sequence}")

    if not module_name in module_sequence:
        raise RuntimeError(f"[Prefetch Tensor Func] {module_name} Invaild module name.")
    
    now_idx = module_sequence.index(module_name)
    prefetch_idx = now_idx - prefetch_size
    # Case: already all layer had been prefetched.
    if prefetch_idx < 0: 
        return

    prefetch_module_name = module_sequence[prefetch_idx]
    if not str(prefetch_module_name) in prefetch_offloaded_pack:
        raise RuntimeError(f"[Prefetch Tensor Func] {prefetch_module_name}({prefetch_idx}) is invaild module name. now_idx: {now_idx}")    
    
    offloaded_tensors = prefetch_offloaded_pack.pop(str(prefetch_module_name))
    prefetch_count += len(offloaded_tensors)
    real_prefetch_count = 0
    for packed in offloaded_tensors:
        refer, is_half = cpu_refer_dict[packed]
        if refer is not None:
            continue
        
        if isinstance(packed, torch.Tensor):
            with torch.cuda.stream(transfer_stream):
                gpu_refer = packed.to(device, non_blocking=False)
                if gpu_refer.is_quantized:
                    gpu_refer = symmetric_dequantize(gpu_refer, is_half)
                cpu_refer_dict[packed] = (gpu_refer, is_half)
                real_prefetch_count+=1
        else:
            raise RuntimeError(f"[Prefetch Tensor Func] {packed} is not Tensor.")  

    prefetch_recomp_cnt = 0
    prefetch_recomp_target_cnt = 0
    if args.do_recompute == 1:
        global module_memory
        _, inp_cpu_refers, _ = module_memory.get(prefetch_module_name, (None, None, None))
        if inp_cpu_refers is None:
            pass
        else:
            for _, _, cpu_refer in inp_cpu_refers:
                prefetch_recomp_target_cnt+=1
                if not isinstance(cpu_refer, torch.Tensor):
                    continue
                refer, is_half = cpu_refer_dict[cpu_refer]
                if refer is not None:
                    continue
                with torch.cuda.stream(transfer_stream):
                    gpu_refer = cpu_refer.to(device, non_blocking=False)
                    if gpu_refer.is_quantized:
                        gpu_refer = symmetric_dequantize(gpu_refer, is_half)
                    cpu_refer_dict[cpu_refer] = (gpu_refer, is_half)
                    real_prefetch_count+=1
                    prefetch_recomp_cnt+=1
                    
    logger.debug(f"[BWD Prefetch] now: {module_name}, target: {prefetch_module_name}, should: {len(offloaded_tensors)}, real: {real_prefetch_count}.") 
    logger.debug(f"[BWD Prefetch] now: {module_name}, target: {prefetch_module_name}, recomp: {prefetch_recomp_cnt}, recomp_target: {prefetch_recomp_target_cnt}")

def prefetch_tensors_in_fwd(module_name):
    global prefetch_size, prefetch_offloaded_pack, cpu_refer_dict, module_sequence, device, model_len, prefetch_count, logger, transfer_stream

    if len(module_sequence) > model_len:
        raise RuntimeError(f"[Prefetch Tensor Func] module sequence exceed model len. {len(module_sequence)} {module_sequence}")
    if not module_name in module_sequence:
        raise RuntimeError(f"[Prefetch Tensor Func] {module_name} Invaild module name.")
    now_idx = module_sequence.index(module_name)
    prefetch_idx = model_len - (now_idx + prefetch_size - model_len)
    # Case: already all layer had been prefetched.
    if prefetch_idx < 0: 
        return

    prefetch_module_name = module_sequence[prefetch_idx]
    if not str(prefetch_module_name) in prefetch_offloaded_pack:
        raise RuntimeError(f"[Prefetch Tensor Func] {prefetch_module_name}({prefetch_idx}) is invaild module name.")    
    
    offloaded_tensors = prefetch_offloaded_pack.pop(str(prefetch_module_name))
    prefetch_count += len(offloaded_tensors)
    real_prefetch_count = 0
    for packed in offloaded_tensors:
        refer, is_half = cpu_refer_dict[packed]
        if refer is not None:
            continue
        
        if isinstance(packed, torch.Tensor):
            with torch.cuda.stream(transfer_stream):
                gpu_refer = packed.to(device, non_blocking=False)
                if gpu_refer.is_quantized:
                    gpu_refer = symmetric_dequantize(gpu_refer, is_half)
                cpu_refer_dict[packed] = (gpu_refer, is_half)
                real_prefetch_count+=1
        else:
            raise RuntimeError(f"[Prefetch Tensor Func] {packed} is not Tensor.")    
    logger.debug(f"[FWD Prefetch] should: {len(offloaded_tensors)}, real: {real_prefetch_count}.") 

def get_module_name(module):
    global model, last_layer

    name = None
    is_last = False
    for n, m in model.model.named_modules():
        if m is module:
            name = n
            break

    if name is None:
        if model_name == module.__class__.__name__ :
            name = module.__class__.__name__
        else:
            name = "Unknown"

    if name == last_layer:
        is_last = True
    
    return name, is_last

def _move_all_to_gpu(cpu_refers):
    global device, cpu_refer_dict, transfer_stream, transfer_event

    gpu_refers = list()
    with torch.cuda.stream(transfer_stream):
        for producer_func, func_idx, cpu_refer in cpu_refers:
            
            if isinstance(cpu_refer, torch.Tensor):
                gpu_refer, is_half = cpu_refer_dict.get(cpu_refer)
                if gpu_refer is None:
                    gpu_refer = cpu_refer.to(device, non_blocking=False)
                    if gpu_refer.is_quantized:
                        gpu_refer = symmetric_dequantize(gpu_refer, is_half)
                    cpu_refer_dict[cpu_refer] = (gpu_refer, is_half)
                    
            else:
                gpu_refer = cpu_refer
            
            gpu_refers.append((producer_func, func_idx, gpu_refer))
    
    return gpu_refers

def reconstruct_input(inp_cpu_refer, func_list):
    global device, transfer_stream, transfer_event, cpu_refer_dict

    inp_gpu_refer = _move_all_to_gpu(inp_cpu_refer)
    transfer_stream.synchronize()
    # msg = ""
    # for func, func_idx, gpu_refer in inp_cpu_refer:
    #     msg += f" func: {func}, func_idx: {func_idx}, gpu_refer: {id(gpu_refer)}"
    # logger.debug("[Recontruct Func] Refers:" + msg)
    # logger.debug(f"[Reconstruct Func] func list: {func_list}")

    out = tuple()
    # logger.info(f"[Recompute Func] gpu_refers: {inp_gpu_refer}")
    for idx, func in enumerate(reversed(func_list)):
        idx = len(func_list) - 1 - idx
        if not isinstance(out, tuple):
            # Wrap single tensor into a tuple without splitting its dimensions
            out = (out,)
        for executed_func, func_idx, gpu_refer in inp_gpu_refer:
            if idx == func_idx:
                out += tuple([gpu_refer])
             
        contig_args = []
        for x in out:
            if isinstance(x, torch.Tensor) and not x.is_contiguous():
                contig_args.append(x.contiguous())
            else:
                contig_args.append(x)
        
        out = func(*tuple(contig_args))
             
        if isinstance(out, tuple):
            out = torch.cat(out, dim=0)
        else:
            out
         
    if isinstance(out, tuple):
        # Concatenate tuple of tensors along the first dimension (adjust dim as needed)
        out = torch.cat(out, dim=0)
    else:
        out = out
    
    return out

def print_cpu_mem_states(where):
    global _cpu_peak_rss, logger
    # Get process RSS
    rss = _psutil_proc.memory_info().rss
    # Update peak RSS
    if rss > _cpu_peak_rss:
        _cpu_peak_rss = rss

    # Get system-wide memory usage
    vm = psutil.virtual_memory()
    sys_used = vm.used
    sys_total = vm.total
    sys_percent = vm.percent

    # Log process and system memory in GiB
    logger.info(
        f"[CPU MEM States] ({where}) "
        f"Process: Curr: {rss/1024**3:.4f}GB, Peak: {_cpu_peak_rss/1024**3:.4f}GB; "
        f"System: Used: {sys_used/1024**3:.4f}/{sys_total/1024**3:.4f}GB ({sys_percent:.1f}%)"
    )

def symmetric_quantize(x: torch.Tensor, num_bits: int):
    """
    PyTorch per-tensor symmetric quantization using torch.quantize_per_tensor.
    """
    is_half = False
    if x.dtype == torch.float16:
        x = x.to(torch.float32)
        is_half=True
    qmax = 2**(num_bits - 1) - 1
    scale = x.abs().max() / qmax
    zero_point = 0
    return torch.quantize_per_tensor(x, scale=scale, zero_point=zero_point, dtype=torch.qint8), is_half


def symmetric_dequantize(q: torch.Tensor, is_half=False) -> torch.Tensor:
    """
    Dequantize a PyTorch quantized tensor back to float.
    """
    if is_half:
        deq = q.dequantize()
        return deq.to(torch.float16)
    return q.dequantize()

from transformers import AutoTokenizer, LlamaForCausalLM, DataCollatorForLanguageModeling, AutoModelForCausalLM, GPT2Tokenizer
from huggingface_hub import login
def main():
    local_rank = 0
    rank = os.environ["RANK"]
    world_size = os.environ["WORLD_SIZE"]
    master_addr = os.environ["MASTER_ADDR"]
    master_port  = os.environ["MASTER_PORT"]
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    # ── 파라미터 파싱 ──────────────────────────────────────
    global args, prefetch_size, logger, do_quantization, do_recompute
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", "-d", type=int, default=0,
                        help="If set to 1, log in debug mode. (default=0)")
    parser.add_argument("--use_nsys", "-n", type=int, default=0,
                        help="if set to 1, it will be profiled with nsys including nvtx (default=0)")
    parser.add_argument("--log_file", "-f", type=str, default="training.log",
                        help="로그 파일명 (default=train.log)")
    parser.add_argument("--model", "-m", type=str, default="llama",
                        help="model name. (default=Llama)(option: llama, opt)")
    parser.add_argument("--max_steps", "-s", type=int, default=1,
                        help="최대 스텝 수 (default=1)")
    parser.add_argument("--max_length", "-l", type=int, default=1024,
                        help="sample length (default=1024)")
    parser.add_argument("--do_offload", "-o", type=int, default=0,
                        help="Activation Offload Option(default=0)")
    parser.add_argument("--do_recompute", "-r", type=int, default=0,
                        help="Activation Recomputation Option(default=0)")
    parser.add_argument("--prefetch_size", "-p", type=int, default=0,
                        help="Prefetch size Option(default=0, max=3). If set to 0, engine won't prefetch activation in backward. it might incur consuming more training time.")
    parser.add_argument("--local_rank", "-rank", type=int, default=0,
                        help="local rank (default=0)"),
    parser.add_argument("--do_quantization", "-q", type=int, default=0,
                        help="Activation Quantization Option(default=0)")
    args = parser.parse_args()

    logging.basicConfig(
        level=(logging.DEBUG if args.debug == 1 else logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()]
    )
    logger = logging.getLogger("deepspeed")

    msg = f"File: {args.log_file}, Max Step: {args.max_steps}, "
    if args.do_offload == 1:
        msg += "Offload: True, "
    else:
        msg += "Offload: False, "
    if args.do_recompute == 1:
        msg += "Recompute: True, "
    else:
        msg += "Recompute: False, "
    msg += f"Prefetch Size: {args.prefetch_size}, "
    msg += f"Sample Length: {args.max_length}"
    prefetch_size = args.prefetch_size
    do_quantization = args.do_quantization
    do_recompute = args.do_recompute

    # 파일 핸들러 교체
    for h in logger.handlers[:]:
        logger.removeHandler(h)
    if int(rank) == 0:
        log_folder = "/home/deepspeed/outputs/PyAO/logs/"
    else:
        log_folder = "/root/"
    fh = logging.FileHandler(log_folder + args.log_file, mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)
    logger.addHandler(logging.StreamHandler())
    logger.info("[Program Settings] "+msg)
    logger.info(f"RANK: {rank}, WORLD_SIZE: {world_size}, MASTER_ADDR: {master_addr}, MASTER_PORT: {master_port}")

    global model, device, model_name, transfer_stream

    # single-node 환경: local_rank 로 GPU 지정
    # device = torch.device("cuda", local_rank)
    deepspeed.init_distributed(rank=int(rank), world_size=int(world_size))
    logger.info(f"Using device {device}")

    # DeepSpeed 설정
    ds_config = "/home/deepspeed/PyAO/config/ds_config_AO_stage2.json"

    # Simple tensor operation with gradient
    hug_token = os.environ.get("HUG_TOKEN", "")
    print("Hugging Face Token:", hug_token)
    login(hug_token)
    if args.model == "llama":
        modelname = "meta-llama/Llama-3.2-1B"
        tokenizer = AutoTokenizer.from_pretrained(modelname)
    elif args.model == "opt":
        modelname = "facebook/opt-1.3b"
        tokenizer = AutoTokenizer.from_pretrained(modelname)
    elif args.model == "gpt":
        modelname = "openai-community/gpt2-large"
        tokenizer = GPT2Tokenizer.from_pretrained(modelname)
    tokenizer.pad_token = tokenizer.eos_token
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transfer_stream = Stream(device=device)
    # if args.model == "llama":
    #     with deepspeed.zero.Init():
    #         model = LlamaForCausalLM.from_pretrained(modelname, ignore_mismatched_sizes=True)
    # elif args.model == "opt":
    #     with deepspeed.zero.Init():
    #         model = AutoModelForCausalLM.from_pretrained(modelname, ignore_mismatched_sizes=True)
    if args.model == "llama":
        model = LlamaForCausalLM.from_pretrained(modelname, torch_dtype=torch.float16)
    elif args.model == "opt":
        model = AutoModelForCausalLM.from_pretrained(modelname, torch_dtype=torch.float16)
    elif args.model == "gpt":
        model = AutoModelForCausalLM.from_pretrained(modelname, torch_dtype=torch.float16)
        # Ensure model.model exists as an attribute (for uniform access)
    

    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    train_dataset = load_wikitext(tokenizer, collator, max_length=args.max_length, split_size=50)

    # 엔진 초기화
    logger.info("Initializing DeepSpeed engine...")
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config,
        # training_data=train_dataset
    )
    if not hasattr(model, "model"):
        model.model = model
    model_name = model.model.__class__.__name__
    logger.info(f"[Program Settings] Model: {model_name}({modelname})")
    logger.info("DeepSpeed engine initialized.")

    batch_size = model_engine.train_micro_batch_size_per_gpu()
    logger.info(f"Batch size: {batch_size}")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        pin_memory=True,
        sampler=DistributedSampler(train_dataset, num_replicas=int(world_size), rank=int(rank)),
        num_workers=int(world_size)
    )

    # Create both contexts
    global pack_ctx, pack_ctx_for_offload, pack_ctx_for_discard, pack_ctx_for_recompute, pre_hook_handler, post_hook_handler
    if args.do_offload == 1:
        pack_ctx_for_offload = saved_tensors_hooks(pack_to_cpu, unpack_from_cpu)
    pack_ctx = saved_tensors_hooks(pack_hook, unpack_hook)
    pack_ctx_for_discard = saved_tensors_hooks(pack_hook_for_discard, unpack_hook_for_discard)
    pack_ctx_for_recompute = saved_tensors_hooks(pack_hook_for_recompute, unpack_hook_for_recompute)

    outlier_file_name = "/home/deepspeed/"
    if args.model == "llama":
        outlier_file_name += "outliers.txt"
    elif args.model == "opt":
        outlier_file_name += "outliers_opt1b3.txt"
    elif args.model == "gpt":
        outlier_file_name += "outliers_gpt2.txt"
    global selected_ops, last_layer, model_len
    try:
        with open(outlier_file_name, "r") as f:
            lines = f.readlines()
            # Extract the recorded last module name
            last_module = None
            for l in lines:
                if l.startswith("Last module:"):
                    last_module = l.split("Last module:")[1].strip()
                    break
            for l in lines:
                if l.startswith("Model Len:"):
                    tmp = l.split("Model Len:")[1].strip()
                    model_len = int(tmp)
                    break

            if last_module is not None:
                last_layer = last_module
                logger.info(f"[Selected Last Module] {last_module}")
            if model_len != 0:
                logger.info(f"[Model Length] {model_len}")
        # after the header lines, each outlier line starts with module name before colon
        for line in lines:
            line = line.strip()
            if ": throughput=" in line:
                mod_name = line.split(":")[0]
                selected_ops.append(mod_name)
    except FileNotFoundError:
        logger.warning("outliers.txt not found; selected_ops remains empty")

    if args.do_recompute == 1:
        logger.info("[Selected OPs]")
        for op in selected_ops:
            logger.debug(f"   {op}")

    # -- register hooks -- #
    leaf_modules = [(name, mod) for name, mod in model_engine.module.model.named_modules() if len(list(mod.children())) == 0]
    names = [name for name, _ in leaf_modules]
    init_timer(names)
    for n, m in leaf_modules:
        m.register_forward_pre_hook(pre_fwd_hook_full)
        m.register_forward_hook(post_fwd_hook_full)
        
        m.register_full_backward_pre_hook(pre_bwd_hook_full)
        m.register_full_backward_hook(post_bwd_hook_full)
    
    logger.info("학습 시작.")

    global fwd_dispatcher, is_backward
    fwd_dispatcher = ForwardDispatchMode()

    if args.do_offload == 1:
        pack_ctx_for_offload.__enter__()
    else:
        pack_ctx.__enter__()
    
    def count_offloaded_tensor():
        global prefetch_offloaded_pack

        full_count = 0
        for module, tensors in prefetch_offloaded_pack.items():
            logger.debug(f"[Count Func] {str(module)}: {len(tensors)}")
            full_count += len(tensors)
        logger.debug(f"[Count Func] Total count: {full_count}")


    full_train_time = 0.0
    print_cpu_mem_states("Before Train")
    if args.use_nsys == 1 and int(rank) == 0:
        torch.cuda.cudart().cudaProfilerStart()
        nvtx.range_push("train")
    done_step = 0
    from deepspeed.runtime.utils import see_memory_usage
    while done_step < args.max_steps:
        for step, batch in enumerate(train_dataloader):
            st = time.time()
            # Forward pass: simple operations
            input_ids = batch["input_ids"].to(device)
            labels    = batch["labels"].to(device)
            if args.use_nsys == 1 and int(rank) == 0:
                nvtx.range_push("forward")
            if args.do_recompute == 1:
                with fwd_dispatcher:
                    out = model_engine(input_ids=input_ids, labels=labels)
            else:
                out = model_engine(input_ids=input_ids, labels=labels)
            torch.cuda.synchronize()
            global quan_cnt
            logger.info(f"[MAIN] Quan CNT: {quan_cnt}")
            if args.use_nsys == 1 and int(rank) == 0:
                nvtx.range_pop()
            loss = out.loss
            if args.debug == 1:
                logits = out.logits
                if torch.isnan(logits).any():
                    logger.error(f"[ERROR] NAN HERE.")
                count_offloaded_tensor()
            logger.info(f"[FWD] done. loss: {loss.item():.4f}")
            # logger.debug(f"[FWD] output: {out}")
            
            # Backward pass
            is_backward = True
            if args.use_nsys == 1 and int(rank) == 0:
                nvtx.range_push("backward")
            model_engine.backward(loss, retain_graph=False)
            torch.cuda.synchronize()
            if args.use_nsys == 1 and int(rank) == 0:
                nvtx.range_pop()
            logger.info(f"[BWD] done.")

            logger.info(f"[Step] Optimization Start.")
            if args.use_nsys == 1 and int(rank) == 0:
                nvtx.range_push("optimizer")
            model_engine.step()
            if args.use_nsys == 1 and int(rank) == 0:
                nvtx.range_pop()
            
            
            logger.info(f"[Step {step}] Optimizer done.")

            model_engine.zero_grad()

            global prefetch_count
            logger.debug(f"[MAIN] total prefetched: {prefetch_count}")

            init_global()

            excute_time = time.time() - st
            logger.info(f"[MAIN] Step Time: {excute_time:.4f}s")
            full_train_time += excute_time

            # torch.cuda.synchronize()
            # deepspeed.get_accelerator().empty_cache()

            done_step+=1
            if done_step >= args.max_steps:
                break
    
    if args.do_offload == 1:
        pack_ctx_for_offload.__exit__()
    else:
        pack_ctx.__exit__()
    logger.info("학습 종료.")
    global peak_gpu_mem, _cpu_peak_rss
    # Compute average training time per step (loop variable 'step' holds the last index)
    avg_time = full_train_time / done_step
    logger.info(f"[MAIN] Training Time: {full_train_time:.4f}s, Avg/step: {avg_time:.4f}s, Peak GPU: {peak_gpu_mem/(1024**3):.4f}GB, Peak CPU: {_cpu_peak_rss/(1024**3):.4f}GB")
    nvtx.range_pop()
    torch.cuda.cudart().cudaProfilerStop()

    ## Timer Results
    global layer_acc_fwd_timer, layer_acc_bwd_timer
    for name in names:
        layer_acc_fwd_timer[name] /= args.max_steps
        layer_acc_bwd_timer[name] /= args.max_steps
    save_path_plot = os.path.join("/home", "deepspeed","outputs", "PyAO", "plots", "experiments")
    save_path_stats = os.path.join("/home", "deepspeed","outputs", "PyAO", "stats", "experiments")
    plot_layer_timer_graphs(names, layer_acc_fwd_timer, layer_acc_bwd_timer, save_path_plot)
    save_layer_timer_stats(names, layer_acc_fwd_timer, layer_acc_bwd_timer, save_path_stats)


if __name__ == "__main__":
    main()
