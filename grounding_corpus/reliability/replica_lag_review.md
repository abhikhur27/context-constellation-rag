# Replica Lag Review - 2026-08-12

The rehearsal database sustained 1.8 times forecast write traffic with application error rate below 0.2 percent. Throughput and connection saturation remained inside their targets.

Replication did not meet the cutover contract: lag peaked at 19 minutes during the import and remained above two minutes for 71 minutes. Reliability requires a fresh four-hour observation with lag continuously below two minutes before sign-off.
