# Restore Drill Record - 2026-08-12

The recovery team restored the snapshot in 31 minutes, but the validation client could not decrypt three inventory partitions because its temporary role lacked the required key permission. The drill stopped before read validation, so the measured recovery time is incomplete and cannot satisfy the 45-minute objective.

A passing rerun must restore the snapshot, decrypt every partition, execute the 24 checked-in consistency queries, and accept reads within 45 minutes.
