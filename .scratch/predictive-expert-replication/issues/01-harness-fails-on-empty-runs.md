# 01: Harness fails loudly when a run measures nothing

**What to build:** A benchmark run either produces a number you can trust or it fails. Today
a runner can serve zero requests, write eight empty traces, and print `done`.

Three times in this project a run reported success having measured nothing: an arm whose
config was silently ignored because the console script loaded the installed vLLM; a placed
arm that activated replicas nothing routed to; and a DSV4 arm whose base checkpoint had no
chat template, so the chat endpoint raised inside the dataset sampler before a single
request went out and the traces held 54 events. Separately, a teardown that sent SIGTERM and
then waited unbounded left an idle 8-GPU server running for 26 minutes with the first arm's
result already on disk, and the remaining two arms never started.

Every later ticket makes a measurement claim, so this one comes first.

**Blocked by:** None (can start immediately).

**Status:** DONE 2026-08-29. All three runners exit 5 with `MEASURED NOTHING` when an arm
served no request, recorded no load where it should, was inert at a non-zero budget, placed
where it must not, or activated replicas without being connected. The checks live in
`check_run_measured.py` and reuse `analyse_e2e.summarize` and `is_connected` rather than
reimplementing them, with 12 unit tests over minimal fixtures. Verified in both directions
against real data: the healthy three-arm run passes, and the DSV4 run whose base checkpoint
had no chat template — the incident this ticket exists for — fails.

- [x] Every runner exits non-zero and prints `MEASURED NOTHING` when its benchmark log
      carries no TTFT, or when no rank trace contains an `execute_context_*` annotation.
      Both conditions, not either: a served run with an empty trace is equally useless.
- [x] Teardown escalates rather than trusting the graceful path: SIGTERM, a bounded poll,
      then SIGKILL, then reap the orphaned engine processes. A hung server must not consume
      the arms that follow it, and the reap must run — it currently sits after the unbounded
      wait and therefore never did.
- [x] The three-arm sweep is the default shape everywhere: feature disabled, prediction
      only, placing. `budget=0` is not a baseline; it enables prediction and only withholds
      placement, and every TTFT figure in this project before 2026-08-29 was missing the
      disabled arm.
- [x] The connectivity self-check is asserted rather than printed: the activation log line
      must appear, and the dumped physical per-rank load must diverge from canonical
      ownership by a non-zero amount. Zero divergence means no token reached a replica no
      matter what else looks healthy.
- [x] Verified by deliberately breaking one arm — a wrong endpoint, or a model with no chat
      template — and observing the non-zero exit rather than a `done`.
- [x] The guard itself is covered: a unit test over a trace fixture with and without step
      annotations, so the guard cannot rot into a no-op.
