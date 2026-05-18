# Workshop Venue Notes

Checked on 2026-05-18.

## Best Fits

### SCALE @ ICML 2026

URL: https://scale-icml-2026.github.io/

Why it fits: the call explicitly names compute- and memory-efficient agent
architectures, long-context reasoning, and memory benchmarks. This is the
cleanest workshop fit for memoized RLM because the contribution is a systems
substrate for efficient long-horizon agents.

Angle: "Language Memoization Graphs for Efficient Long-Context Agent Workloads."

### CATS @ ICML 2026

URL: https://cats-icml.github.io/

Why it fits: the workshop is about continual adaptation at scale and explicitly
asks whether adaptation should happen through weights, architecture, or
in-context mechanisms. Our answer is externalized reusable subproblem memory:
adaptation through persistent scoped fragments, not model updates.

Angle: "When Continual Learning Moves to Memoized Subproblem Reuse."

### MemAgents @ ICLR 2026

URL: https://iclr.cc/virtual/2026/workshop/10000792

Why it fits: this was specifically about memory for LLM-based agentic systems.
It is highly aligned intellectually, though ICLR 2026 has already happened. It
is useful as related-work positioning and as a signal that the community is
actively organizing around agent memory substrates.

Angle if there is a follow-on venue: "Execution Memory, Not Just Context
Memory."

### FAGEN @ ICML 2026

URL: https://fagen-workshop.github.io/

Why it fits: if the empirical story emphasizes failure modes of non-memoized
agents, repeated work, lost intermediate conclusions, and context bloat, this is
a strong fallback. It is less directly about the positive systems substrate,
but the "agents fail over hundreds of dependent steps" framing is close.

Angle: "Repeated Subproblem Failure as an Agentic Failure Mode."

## Less Direct, Still Relevant

### Long-Context / Conditional Memory Access Workshops

The ICML 2026 workshop listing includes long-context and conditional memory
access topics. These are relevant if the paper leans into evaluation and
attention-window limits rather than agent execution memory.

URL: https://icml.cc/Downloads/2026

## Current Recommendation

Primary target: SCALE @ ICML 2026.  
Secondary target: CATS @ ICML 2026.  
Failure-mode framing target: FAGEN @ ICML 2026.

The paper should avoid sounding like another semantic cache paper. The sharper
submission framing is:

```text
Persistent dynamic programming for language-model subproblems:
store scoped reusable fragments, compose them over long context,
and measure avoided recursive work.
```

