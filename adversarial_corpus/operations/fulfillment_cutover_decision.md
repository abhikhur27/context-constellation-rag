# Fulfillment Cutover Decision - 2026-09-08

Status: the North warehouse fulfillment cutover is paused. The rehearsal created duplicate pick tasks when a scanner retried after an acknowledgement timeout. This is an idempotency and inventory-control decision, not a throughput decision.

Resumption requires Quality to replay every retry case without duplicate tasks, Inventory Control to reconcile the 42 affected units, and Warehouse Operations to complete a witnessed scanner retry drill with no repeated picks. This decision supersedes the August cutover plan.
