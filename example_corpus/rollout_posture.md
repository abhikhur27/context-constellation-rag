# Product Rollout Posture

The largest incident category over the last quarter was not pure latency. It was mismatch between retrieval confidence and policy confidence.

Key patterns:
- high lexical overlap does not always imply policy fit
- nearest-neighbor chunks often overrepresent one service area
- rollout decisions were stronger when evidence came from at least two operational domains

Recommendation:
Require at least two distinct evidence constellations before policy updates are promoted to default behavior.
