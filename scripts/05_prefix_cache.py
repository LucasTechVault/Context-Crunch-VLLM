import os
import time
import asyncio
import multiprocessing as mp
from typing import Any, Dict

# ==========================================
# CONFIGURATION
# ==========================================
# We use a moderately large prompt to make the speedup obvious
# 5000 & "9 tokens" ~= 45000 tokens
SHARED_CONTEXT_LEN = 5000
MODEL_NAME = "microsoft/Phi-3-mini-128k-instruct"
GPU_ID = "3"
# ==========================================

async def _measure_ttft(engine, prompt: str, request_id: str) -> float:
    from vllm import SamplingParams
    
    # Max tokens 1 because focus is on PREFILL not DECODE
    params = SamplingParams(max_tokens=1, temperature=0.0)
    
    start = time.time()
    gen = engine.generate(prompt, params, request_id=request_id)
    async for _ in gen:
        # Return time immediately after first token
        return time.time() - start
    return float("inf")

async def _run_caching_lab() -> Dict[str, Any]:
    print(f"[Worker] Init Engine with Prefix Caching ENABLED", flush=True)
    from vllm import AsyncLLMEngine, AsyncEngineArgs
    
    engine_args = AsyncEngineArgs(
        model=MODEL_NAME,
        trust_remote_code=True,
        gpu_memory_utilization=0.90,
        max_model_len=130000,
        disable_log_stats=True,
        enforce_eager=True,
        
        # --- PREFIX CACHE SWITCH ---
        enable_prefix_caching=True, # allows "smart" caching
        # ------------------------
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    # Construct "Shared Document"
    # In RAG, this would be the retrieved context injected into the prompt
    shared_document = "The history of computing is long and complex. " * SHARED_CONTEXT_LEN
    
    # ---------------------------------------------------------
    # REQUEST 1: The "Cold" Start - Without Prefix Cache
    # ---------------------------------------------------------
    print(f"[Worker] Sending Request A (Cold Start)...", flush=True)
    prompt_a = shared_document + "\nQuestion: Summarize the history."
    ttft_a = await _measure_ttft(engine, prompt_a, "req_A")
    print(f"[Worker] Request A Finished. TTFT: {ttft_a:.4f}s", flush=True)
    
    # Small sleep to ensure logs are clean
    await asyncio.sleep(0.5)

    # ---------------------------------------------------------
    # REQUEST 2: The "Hot" Start
    # ---------------------------------------------------------
    print(f"[Worker] Sending Request B (Shared Context)...", flush=True)
    # EXACT same prefix (shared_document), slightly different question
    prompt_b = shared_document + "\nQuestion: What is the complexity?"
    ttft_b = await _measure_ttft(engine, prompt_b, "req_B")
    print(f"[Worker] Request B Finished. TTFT: {ttft_b:.4f}s", flush=True)

    return {
        "ok": True,
        "cold_ttft": ttft_a,
        "hot_ttft": ttft_b
    }

def _worker_entry(result_queue: mp.Queue):
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
    os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
    
    try:
        res = asyncio.run(_run_caching_lab())
        result_queue.put(res)
    except Exception as e:
        result_queue.put({"ok": False, "error": str(e)})

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    print("--- Starting Prefix Caching Lab ---")
    
    q = mp.Queue()
    p = mp.Process(target=_worker_entry, args=(q,))
    p.start()
    p.join() 

    if not q.empty():
        res = q.get()
        if res["ok"]:
            cold = res['cold_ttft']
            hot = res['hot_ttft']
            speedup = cold / hot
            
            print("\n================ CACHING REPORT ================")
            print(f"Document Size:   ~{SHARED_CONTEXT_LEN} words")
            print(f"Cold Start (A):  {cold:.4f}s  (Computing Math)")
            print(f"Hot Start  (B):  {hot:.4f}s   (Memory Lookup)")
            print("-" * 48)
            print(f"Speedup:      {speedup:.1f}x FASTER")
            print("================================================")
        else:
            print(f"Error: {res.get('error')}")