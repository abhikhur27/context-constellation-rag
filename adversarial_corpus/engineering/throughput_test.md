# Fulfillment Throughput Test - 2026-09-07

The fulfillment service sustained 2.1 times forecast peak volume. P95 task creation latency stayed below 280 milliseconds and the queue drained inside the ten-minute target.

The test used one acknowledgement per scanner request and did not exercise retry idempotency. Healthy throughput is not evidence that repeated acknowledgements cannot create duplicate pick tasks.
