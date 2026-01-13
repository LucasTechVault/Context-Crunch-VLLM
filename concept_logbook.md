#### 11-Jan-2026 (00:56)

## Phase 1 - Physics of Latency

### 2 Modes of LLMs

**1. Prefill (Reader):**

- user sends 100-page PDF (100k tokens)
- model reads it all at once to build KV Cache
- Can be done all at once because model can see all words

**Physics**: Massive Matmul
**Bottleneck:** Compute-bound (dependent on GPU) - Memory bandwidth unstrained because weights loaded once & re-used
**Metric**: Time to first token (TTFT) -> How long until user sees first word

**2. Decode (Writer)**
**Action**: Model generates answer (summary of PDF is...)
**Physics**: generates 1 token at a time. For every single token, has to reload entire model from VRAM to chip.

- VRAM : Where model lives. Large but slow to access (24,40,80 GB etc.)
- SRAM : Where the actual math happens. Incredibly fast but tiny (50 MB)
- impossible to load model into SRAM
  1. Model looks at word 1 - 10
  2. Model calculates word 11.
  3. STOP. It cannot calculate word 12 until it knows what word 11 is
  4. Now it takes word 1 - 11 to calculate word 12.

##### Stream Cycle

1. Token 1: Stream 14GB weights from VRAM to SRAM -> Forward pass -> Output "The".
2. Token 2: Stream 14GB weights from VRAM to SRAM -> Forward pass

**Metric:** Time Per Output Token (TPOT) - How fast does text stream?

## Section 1.1 - "Prefill Experiment"

    - Input length from 1k to 128k token
    - measure how long Prefill takes (to read entire length)
    - Goal -> Find compute-bound limit
    - how it works:
        - send in dummy_prompt "the_ " * input length

## Section 1.2 - "Decode Experiment"

    - Output length from [128, 256, 512, 1024, 2048]
    - model needs previous words to generate next word
    - cannot process all tokens like it did for prefill
    - need to fetch model weights, forward prop -> generate output for each word
    - specify in SamplingParams(min_tokens=output_len, ignore_eos=True) to prevent early cutoff

# Phase 2 - Solutions to Prefill & Decode Problems

## 2.1 - Chunked Prefill (Handle Prefill inefficiency)

    - opening more cashier lanes for more customers
    - Chunked Prefill:
        - Customer A: 128000 items
        - Customer B: 10 items

        - Cashier scans 512 items for A
        - Cashier checks line, help B first
        - Cashier returns to A, process next chunked batch
        - B waits 0.1s, A waits 7.7s + 0.1s

#### FlashAttention Side Learning

- vLLM uses FlashAttention to compute attention quickly
- Attention Computation is costly because of the huge grid (e.g. 8000 x 8000) that cannot fit into SRAM at once.
- instead, this huge 8000 x 8000 is split into "tiles" that can fit into SRAM. Attention for each tile is computed.
- Problem is with softmax as it requires the global max. The solution is to compute a provisional softmax attention using local max of each tile.
- For every new tile, all previously computed provisional softmax will be scaled and adjusted with the latest information.
- eventually, after streaming all tiles, the adjust provisional softmax will be as though the softmax was computed in a single pass

## 2.2 - Dynamic Batching (Handle Decode Inefficiency)

- Decode phase, load model (7GB) just to process single token -> drive bus to transport 1 passenger
- Prefill phase, load model (7GB) to process many tokens (efficient)
- **Solution:** Dynamic Batching --> (wait for 50 people before moving bus) - idea: DEFINE Batch size (16, 64, 128 etc.) - Measure throughput

# Phase 3 - Retrieval-Augmented Generation (RAG) Basics

### What is RAG?

Imagine an LLM taking a test:
**Standard LLM:**

- Student answer from memory.
- If studied last year (training cut-off), data not up-to-date

**RAG:**

- Student allowed to use textbook
  - **Retrieve:** Look up relevant page in book (context)
  - **Augment:** Copy that page into notes (prompt)
  - **Generate:** Write answer based on notes (context in prompt)

**Problem:** - RAG is expensive - Every question = copy-paste huge chunk of context into prompt (prefill)

#### Prompt-Training (Few-Shot Prompting)

- Pre-trained models performing dynamic learning
- model weights do not change (forgets after restart)
- temporary memory
- LLM are good pattern recognizers

#### RAG & Few-Shot Prompting

- Feed pre-trained model Context (100-page manual)
- Model learn how to answer question based on context provided
- learning is temporarily, forgets after reset
- **Problem:** Need to paste (10k tokens) into prompt each time to calculate KV Cache

- **Solution:** Prefix Caching (Radix Attention)
  - Context attached to prompt as prefix
  - stores the prefix KV in cache

#### Prefix Cache is a form of KV Cache

**Radix Attention:** Smart Cache Layer - detects if different requests have the same prefix (prompts) using Radix tree where each node represent a sequence of tokens and pointer to their KV Cache

**Paged Attention:** Solves the need for contiguous VRAM, allows "pages" of attention to be stored in different parts of memory, using a Page table to track the location. Thus, memory can be maximize

#### How we do Prefix Caching in vLLM

- specify `enable_prefix_caching=True` in AsyncEngineArgs of vLLM
