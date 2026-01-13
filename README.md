# Project 4 - High Performance RAG Inference Optimization (Context Crunch)

\*Hardware Target: 2x Nvidia H200 NVL (141GB VRAM each)

---

## Objective

**Scenario:** Inference Engineer for Enterprise **RAG** Platform

**Problem:** Production Latency violating SLOs when processing massive PDF documents (100k+ tokens). This causes "Head-of-Line" blocking where single large request freezes GPU for all other users.

**Goal:** Optimize serving engine capable of Long-Context Inference (100k+ tokens) while maintaining strict latency guarantees for concurrent users.
**Solution:**Memory-Aware Scheduling & Prefix Caching

The scope includes:

**1. Latency Physics:** Deconstructing "Time-to-First-Token" (TTFT) vs "Time-per-Output-Token" (TPOT) to identify **Compute-bound vs Memory-bound** bottlenecks

**2. Memory Math:** Predicting & verifying KV Cache growth (OOM Equation) to size clusters mathematically.

**3. Scheduling Optimization:** Implement Chunked Prefill to solve "Head-of-Line" blocking problem

**4. Advanced Caching:** Enable Radix Attention (Prefix Caching) to achieve near-zero latency for shared document queries

---

## Tech Stack

**Hardware:** Nvidia H200 NVL (Utilization capped to simulate constraints)

**Serving Engine:** vLLM

**Model:** Llama-3-8B-Instruct (High throughput) & Qwen2.5-14B (Reasoning) & microsoft/Phi-3-mini-128k-instruct

**Dataset:** ZeroScrolls (Long-context benchmarks) & Synthetic "Needle-in-Haystack" generators

**Optimization Techniques:** PagedAttention (vLLM), Continuous Batching, Chunked Prefills, APC (Auto Prefix Caching)

---

## Key Achievements & Roadmap

### Phase 1: Physics of Latency

- [x] **Baseline Latency Profiling:** Measured "Prefill vs Decode" ratio. Proved that Prefill is Compute-Bound $O(n^2)$ while Decode is Memory-Bandwidth bound.

- [x] **TTFT vs Input Length:** Plotted non-linear growth of latency as context scales from 4k --> 100k tokens.

- [x] **OOM Formula**: Derived & verified the KV Cache Memory Formula (2 x L x H x S x B) against real-world VRAM logs

### Phase 2: Scheduling & Throughput

- [x] **Head-of-Line Simulation:** Simulated User A chatting as normal user (small prompts) when User B uploaded large document (120k token), causing latency for User A.

- [x] **Chunked Prefill Solution:** Configured Scheduler to break massive prompts into processing chunks (e.g. 512 tokens/step), allow User A's chat to remain undisrupted.

- [] **Dynamic Batching Decode Solution:** Tuned `max_num_seqs` to find saturation point where Throughput gains diminish (T_sat)

### Phase 3: RAG Foundations (Caching)

- [] **Prefix Caching Implementation:** Enabled Radix Attention for "Shared Document" scenario (50 users querying 1 PDF)

- [] **Cache Hit Rate Analysis:** Achieved ~50x improvement in TTFT for cached queries (skipping prefill phase)

- [] **Memory Re-Use Verification\*:** Confirmed via monitoring that VRAM usage remained stable despite 50 concurrent quests, proving block reuse.

### Phase 4: Reliability & Stress Test

- [] **Tail Latency Stress Test:** Flooded server (Request > Service Rate) to measure p99 latency degradation

- [] **Queue Management:** Implemented "Fail Fast" logic to reject requests immediately rather than allowing queue to grow infinitely.

- [] **Final Recommendation:** Delivered sizing guide (X GPUs to serve Y users with Z document size)

---

## Evidence

### Phase 1 Evidence: Physics of Latency

**1. Prefill Benchmark (TTFT)**
The latency to read documents / prompts of varying lengths before generating first word.

| Context Length      | TTFT (Wait Time) | Processing Speed        | KV Cache Size (Per User) (Theoretical) (MHA not GQA) |
| :------------------ | :--------------- | :---------------------- | :--------------------------------------------------- |
| 1,024 (Chat)        | 0.03s            | 33,156 tok/s            | 0.38 GB                                              |
| 4,096 (Doc)         | 0.08s            | **50,689 tok/s (Peak)** | 1.50 GB                                              |
| 32,768 (Paper)      | 0.87s            | 37,493 tok/s            | 12.00 GB                                             |
| 65,536 (Book)       | 2.49s            | 26,281 tok/s            | 24.00 GB                                             |
| **128,000 (Novel)** | **7.77s**        | **16,478 tok/s**        | **46.88 GB**                                         |

**Findings:**

- **Compute-Bound:** Procesing speed drops by **60%** as context scales from 4k to 128k, confirming $O(N^2)$ computation for Attention even with FlashAttention optimization
- **8 Second Wall**: Single user sending 128k request freezes GPU for **8 seconds**. This causes massive "Head-of-Line" blocking for all other requests
- **KV Cache Calc** As vLLM pre-allocates VRAM, 1 way to estimate memory consumption is via 2 _ Layers _ hidden \* Precision bytes. 46GB for 128k tokens translates to handling 3 users before OOM.

**2. Decode Benchmark (TPOT):**
The speed of generating (writing) after prompt is processed
| Target Output | Decode Speed | Analysis |
| :--- | :--- | :--- |
| 128 tokens | 95.68 tok/s | Memory Bound Baseline |
| 2048 tokens | 105.99 tok/s | Stable Generation Speed |

**Findings:**

- There is a **500x Discrepancy** between Prefill (50k tok/s) - compute bound and Decode (106 tok/s) - memory-bound
- **Implication:** GPU efficient at batch processing input prompt (prefill) but extremely inefficient at generation (decoding)
- **Danger:** Single 128k prefill takes 7.7s, creating massive blocker for other user's request

### Phase 2 Evidence: Scheduling & Throughput Improvements ("7" seconds stall)

**2.1 Problem:** In a production environment, "Prefill" is compute-bound $O(N^2)$ (every token depend on every other token) while "Decode" is memory bound.

- **Scenario:** User A uploads 120k token document while User B is chatting.
- **Baseline Behavior:** GPU locks up for 7 seconds to process the prefill request. User B thought that chat crashed

**2.1 Solution:** Configure vLLM scheduler to prioritize **Inter-Token Latency** over raw throughput.

- **Configuration:** `enable_chunked_prefill=True` & `max_num_batched_tokens=512`.
- **Mechanism:** Massive 120k request broken into 234 small batches. Scheduler injects chat tokens _between_ batches.
- **Result:** No disruption occurs

**Metric:** Inter-Token Latency Spike

| Scenario      | Scheduler Config | Worst Freeze (Spike) | User Experience    |
| :------------ | :--------------- | :------------------- | :----------------- |
| **Baseline**  | Blocking (128k)  | **6.99s**            | Connection Stalled |
| **Optimized** | Chunked (512)    | **0.13s**            | Smooth Chat        |

**2.1 Engineering Constraints Discovery:**
**Problem:** Attempted to optimize for "New User Latency" (TTFT) using concurrent partial prefills.

- **Outcome:** Failed
- **Root Cause:** Deep dive into vLLM internals revealed underlying limitations of Attention Kernels for Phi-3 do not support concurrent prefill matrices.
