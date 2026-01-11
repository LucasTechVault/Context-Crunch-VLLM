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

- [] **Head-of-Line Simulation:** Reproduced "Production Stall" where single long request spikes latency for 50 short requests

- [] **Chunked Prefill Solution:** Configured Scheduler to break massive prompts into processing chunks (e.g. 512 tokens/step), reducing User's wait time by X%.

- [] **Dynamic Batching:** Tuned `max_num_seqs` to find saturation point where Throughput gains diminish (T_sat)

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
