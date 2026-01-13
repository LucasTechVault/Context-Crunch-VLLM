import os
import time
import asyncio
import numpy as np
import multiprocessing as mp
from typing import Any, Dict

# Test 1 (Disruption) - Set to 130000 
# Test 2 (No Disruption) - Set to 512
CHUNK_SIZE = 130000

GPU_ID = '3'
MODEL_NAME = "microsoft/Phi-3-mini-128k-instruct"
HEAVY_PROMPT_LEN = 120000

async def _drain_heavy(engine, prompt, params):
    """Start heavy request and consume output to keep engine busy"""
    try:
        gen = engine.generate(prompt, params, request_id="heavy_req")
        async for _ in gen:
            pass 
    except asyncio.CancelledError:
        pass
    
async def _run_experiment() -> Dict[str, Any]:
    print(f"[Worker] Importing vLLM", flush=True)
    from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
    
    print(f"[Worker] Initializing Engine (Chunk: {CHUNK_SIZE})", flush=True)
    engine_args = AsyncEngineArgs(
        model=MODEL_NAME,
        trust_remote_code=True,
        gpu_memory_utilization=0.90,
        max_model_len=130000,
        disable_log_stats=True,
        enforce_eager=True,
        enable_chunked_prefill=True,
        max_num_batched_tokens=CHUNK_SIZE, 
    )
    
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    
    # 1. Start User A (normal chat)
    print(f"[Worker] User A starts chatting", flush=True)
    prompt_chat = "Count from 1 to 100 very slowly: one, two, three,"
    sampling_chat = SamplingParams(max_tokens=200, temperature=0.0)
    
    token_times = []
    start_event = asyncio.Event()
    
    async def chat_monitor():
        gen = engine.generate(prompt_chat, sampling_chat, request_id="chat_req")
        count = 0
        async for _ in gen:
            token_times.append(time.time())
            count += 1
            if count == 5: # Wait for 5 tokens before dropping heavy request
                start_event.set()
                
    chat_task = asyncio.create_task(chat_monitor())
    
    # Wait for User A to warm up
    await start_event.wait()
    
    # 2. Drop the Heavy Request (User B)
    print(f"[Worker] User B drops 120k Token Request", flush=True)
    prompt_heavy = "the " * HEAVY_PROMPT_LEN
    sampling_heavy = SamplingParams(max_tokens=1, temperature=0.0)
    
    heavy_start_time = time.time()
    heavy_task = asyncio.create_task(_drain_heavy(engine, prompt_heavy, sampling_heavy))
    
    # Wait for chat to finish
    await chat_task
    
    # Cleanup
    await engine.abort("heavy_req")
    await asyncio.gather(heavy_task, return_exceptions=True)
    
    # 3. Analyze Data
    # Filter tokens that happened AFTER the heavy request started
    relevant_times = [t for t in token_times if t > heavy_start_time]
    
    if len(relevant_times) < 2:
        return {"ok": False, "error": "Not enough tokens generated after collision."}
        
    # Calculate ITL (time differences between tokens)
    itls = np.diff(relevant_times)
    max_spike = np.max(itls)
    
    return {"ok": True, "spike": max_spike}

def _worker_entry(result_queue: mp.Queue):
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
    os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
    try:
        res = asyncio.run(_run_experiment())
        result_queue.put(res)
    except Exception as e:
        result_queue.put({"ok": False, "error": str(e)})

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    
    q = mp.Queue()
    p = mp.Process(target=_worker_entry, args=(q,))
    p.start()
    p.join() 

    if not q.empty():
        res = q.get()
        if res["ok"]:
            print("\n================ RESULT ================")
            print(f"Chunk Size:   {CHUNK_SIZE}")
            print(f"Worst Freeze: {res['spike']:.4f}s")
            if res['spike'] > 2.0:
                 print("Status: Massive Disruption Detected")
            else:
                 print("Status: Smooth Chat")
            print("========================================")
        else:
            print(f"Error: {res.get('error')}")