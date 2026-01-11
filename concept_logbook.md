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

## Section 1 - "Prefill Experiment"

    - Input length from 1k to 128k token
    - measure how long Prefill takes (to read entire length)
    - Goal -> Find compute-bound limit
