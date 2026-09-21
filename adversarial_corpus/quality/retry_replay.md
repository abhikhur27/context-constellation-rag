# Scanner Retry Replay - 2026-09-08

Quality replayed 600 scanner requests with delayed acknowledgements. Fourteen retries created a second pick task for the same order line, and all fourteen duplicated the inventory reservation before the original task completed.

This retry result is the cutover blocker despite the separate passed throughput test. A passing rerun must cover lost, delayed, and repeated acknowledgements and produce exactly one task and one reservation for every order line.
