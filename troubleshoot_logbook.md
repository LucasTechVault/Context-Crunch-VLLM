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
