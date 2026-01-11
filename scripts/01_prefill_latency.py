import multiprocessing
import os
import time
import numpy as np

def benchmark_prefill(input_len, return_dict):
    # internal import per new process
    import torch
    from vllm import LLM, SamplingParams
    import gc
    
    print(f"\n Testing Input Length: {input_len} tokens..", flush=True)
    
    try:
        os.environ["CUDA_VISIBLE_DEVICE"] = "0"
        os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
        
        MODEL_ID = "microsoft/Phi-3-mini-128k-instruct" # Fast model to test physics
        TP_SIZE = 1

        # init vllm
        llm = LLM(
            model=MODEL_ID,
            tensor_parallel_size=TP_SIZE,
            gpu_memory_utilization=0.95,
            max_model_len=132000, # Qwen is 32k, we set 132k for benchmarking
            trust_remote_code=True,
            enforce_eager=True, # don't generate graph, push boundaries
            disable_log_stats=True
        )
        
        # dummy prompt
        dummy_prompt = "the " * input_len # 'the ' is usually 1 token
        
        sampling_params = SamplingParams(
            max_tokens=1, # generate only 1 token because focus on prefill
            temperature=0. # deterministic (choose highest prob)
        )
        
        # warmup (critical for accurate timing)
        llm.generate(["warmup"], sampling_params)
    
        
        # Measure TTFT - prefill reading time
        torch.cuda.synchronize() # force cpu to wait for gpu
        start = time.time()
        outputs = llm.generate([dummy_prompt], sampling_params)
        torch.cuda.synchronize()
        duration = time.time() - start
        
        # Capture Peak Memory (Theoretical)
        # Phi-3-Mini Architecture
        LAYERS = 32 # KV pairs unique in each layer, 32 layers = 32 unique KV pairs
        HIDDEN_SIZE = 3072 # hidden dim size
        DTYPE_BYTES = 2 # bf16
        
        actual_tokens = len(outputs[0].prompt_token_ids)
        
        # 2 for K and V
        kv_cache_per_token = 2 * LAYERS * HIDDEN_SIZE * DTYPE_BYTES
        total_kv_cache_bytes = kv_cache_per_token * actual_tokens
        total_kv_cache_gb = total_kv_cache_bytes / (1024**3)

        # Compute TTFT
        prefill_speed = actual_tokens / duration # tok / s
        
        print(f"[Worker] Processed {actual_tokens} tokens in {duration:.4f}s", flush=True)
        return_dict[input_len] = (duration, prefill_speed, actual_tokens, total_kv_cache_gb) # save in manager dict
        
        # good practice cleanup
        del llm
        gc.collect()
        
    except Exception as e:
        print(f"[Worker] Crashed. {e}", flush=True)
        return_dict[input_len] = (.0, .0, 0, 0.)

# Controller
if __name__ == "__main__":
    # spawn - start brand new process - re-import all modules
    # slower but safer
    multiprocessing.set_start_method('spawn', force=True) 
    
    # Input lengths to benchmark
    INPUT_LENGTHS = [
        1024, 4096,
        16384, 32768,
        65536, 128000
    ]
    
    manager = multiprocessing.Manager() # Manager process to oversee all created child processes
    results = manager.dict() # creates dict that lives in Manager process
    
    print(f"Phase 1 - Prefill Latency Benchmarking (TTFT)")
    
    for length in INPUT_LENGTHS:
        p = multiprocessing.Process(
            target=benchmark_prefill,
            args=(length, results) # pass results to be written in child proces
        )
        
        p.start()
        p.join() # wait for process to finish
    print("\n--- Latency Report (Prefill Phase) ---")
    print("| Target Tokens | Actual Tokens | TTFT (Wait Time) | Prefill Speed (tok/s) | VRAM Used (GB) |")
    print("|---------------|---------------|------------------|-----------------------|----------------|")

    for length in INPUT_LENGTHS:
            if length in results:
                t, speed, actual, mem = results[length]
                if actual > 0:
                    print(f"| {length:<13} | {actual:<13} | {t:<16.4f} | {speed:<21.2f} | {mem:<14.2f} |")
                else:
                    print(f"| {length:<13} | {'CRASHED':<13} | -                | -                     | -              |")
            else:
                print(f"| {length:<13} | {'FAILED':<13} | -                | -                     | -              |")