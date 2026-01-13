#### 11-Jan-2026

##### 10:57

**Problem:** Rope Scaling

- RoPE Scaling `Illegal Memory Access` error for Qwen2.5-1.5B model
- prompted with >32768 tokens even though max_model_len overridden to 132k.

**Fix:**

- Identified that RoPE only trained up to 32k.
- Forcing it beyond the limit without mathematical stretching will cause it to look up invalid indices in the position matrix.
- Switch to larger model Phi-3-Mini-128k, natively trained for longer context

**Takeaway:**

- Hardware capacity != Model Capability
- cannot force a model to read more than what it was trained for unless apply scaling techniques

**Problem:** Off by 1 guardrail

- benchmark test failed when using exactly 131072 tokens
- tokenizer overhead (BOS / EOS) pushed sequence to 131 073, crossing hard limit

**Fix:**

- adjust to 128000 tokens or 130000 tokens

**Takeaways:**

- Never run at 100% theoretical limits
- always leave buffer for special tokens and system prompts

##### 17:22

**Problem:** vLLM First-Come-First-Serve Scehduling policy. If we send a heavy request (120k tokens) and include chunked_prefill, it will simply break the heavy request into chunks for processing. It does not allow new request to jump queue
**Fix:** Tried Concurrent Partial Prefills but unsupported by vllm.
**Takeaways:** Measuring Prefill TTFT in this experiment might not work.

#### 13-Jan-2026

##### 18:42

**Problem:** Address vLLM FCFS scheduling policy
**Fix:**

- instead of showing queue jumping, focus on responsiveness
- While user A chatting with bot, user B upload massive script. User A's chat should not freeze.
- measure of Inter-token latency instead of TTL
