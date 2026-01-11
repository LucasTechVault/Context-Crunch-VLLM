import multiprocessing
import os
import time

def benchmark_decode(output_len, return_dict):
    import torch
    from vllm import LLM, SamplingParams
    import gc
    
    print(f"\nTesting Output Generation: {output_len} tokens...", flush=True)
    
    try:
        # Select GPU 3
        os.environ["CUDA_VISIBLE_DEVICES"] = "3"
        
        MODEL_ID = "microsoft/Phi-3-mini-128k-instruct"
        TP_SIZE = 1 
        
        llm = LLM(
            model=MODEL_ID,
            tensor_parallel_size=TP_SIZE,
            gpu_memory_utilization=0.90, 
            max_model_len=120000,
            trust_remote_code=True,
            enforce_eager=True,
            disable_log_stats=False 
        )
        
        dummy_prompt = "Write a very long detailed history of the Roman Empire. Go into detail about every emperor. Start with: "
        
        sampling_params = SamplingParams(
            max_tokens=output_len, 
            min_tokens=output_len,
            ignore_eos=True, # keeps generating even if EOS
            temperature=0.0
        )
        
        print("[Worker] Starting Generation...", flush=True)
        
        # --- MEASURE GENERATION SPEED ---
        torch.cuda.synchronize()
        start = time.time()
        
        outputs = llm.generate([dummy_prompt], sampling_params)
        
        torch.cuda.synchronize()
        duration = time.time() - start
        
        # Calculate Stats
        generated_tokens = len(outputs[0].outputs[0].token_ids)
        decode_speed = generated_tokens / duration
        
        print(f"[Worker] Generated {generated_tokens} tokens in {duration:.4f}s", flush=True)
        return_dict[output_len] = (duration, decode_speed)
        
        del llm
        gc.collect()
        
    except Exception as e:
        print(f"[Worker] Crashed: {e}", flush=True)
        return_dict[output_len] = (0.0, 0.0)

# --- MAIN CONTROLLER ---
if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    
    OUTPUT_LENGTHS = [128, 256, 512, 1024, 2048]
    
    manager = multiprocessing.Manager()
    results = manager.dict()

    print(f"---Project 4: Decode Physics Lab (TPOT) ---")

    for length in OUTPUT_LENGTHS:
        p = multiprocessing.Process(target=benchmark_decode, args=(length, results))
        p.start()
        p.join()
        
    print("\n---Decode Report (Generation Phase) ---")
    print("| Target Output | Time (s) | Decode Speed (tok/s) |")
    print("|---------------|----------|----------------------|")
    for length in OUTPUT_LENGTHS:
        if length in results:
            t, speed = results[length]
            print(f"| {length:<13} | {t:<8.4f} | {speed:<20.2f} |")