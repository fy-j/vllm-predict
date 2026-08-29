# Superseded tickets — the host-planned design

These describe the implementation measured on 2026-08-29 and found net negative: the plan
was computed on the host, so `plan_and_launch` had to synchronise on a device-to-host copy
once per predicted layer, and `ncclSend`'s peer being a host integer meant that sync could
not be removed. Against a stock server the placed arm cost **+31.5% mean TTFT** while a
perfect placement could return at most **5.05% of a prefill step**.

They are kept because their measurements remain valid and are cited by the current spec —
in particular ticket 00's feasibility work, 06 and 10 on prediction accuracy, 12 on replica
slot memory, and 13's device-side design, whose NVSHMEM probe now passes. The current
ticket set replaces their *orchestration*, not their evidence.

`14` is the exception worth reading in full: it is the three-arm measurement that produced
the verdict, and its method is carried forward as a standing requirement in the new spec.
