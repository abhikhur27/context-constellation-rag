# Inventory Database Cutover Hold - 2026-08-12

Operations paused the warehouse inventory database cutover after the rehearsal replica fell 19 minutes behind the primary and the recovery drill failed before a restored copy could accept reads. The hold is about recoverability, not application throughput.

The cutover can resume only after Reliability shows replica lag below two minutes for a continuous four-hour window, Recovery completes a restore-and-read validation in under 45 minutes, and Security confirms the temporary decryption role is active for the drill window.
