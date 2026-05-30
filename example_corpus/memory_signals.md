# Team Memory Signals

User interviews showed frustration with repetitive LLM responses that ignored previous exceptions.

A retrieval layer with explicit memory slices helped:
- failed rollout exceptions
- customer-facing copy constraints
- policy edge-case decisions

The strongest results came from combining semantic retrieval with lexical constraints and then running MMR to keep context varied.
