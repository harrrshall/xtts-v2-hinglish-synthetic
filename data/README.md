# data/ — guide

This repo holds the data INPUTS and the pulled-back EVALUATION evidence. The full-scale pipeline
outputs (5,880-clip synth set, filter scores, training checkpoints) live on the GPU box under
``, because they are large.

| Dir | Role | Notes |
|-----|------|-------|
| `spontaneous_hinglish/` | INPUT: real spontaneous Hinglish | 1,497 YouTube clips + transcripts; text seed + hard eval set. Copyrighted (internal use). |
| `corpus/` | INPUT: final text corpus | `corpus.jsonl` = 1,470 rows (real seed + 2-gate-verified generated high-CS), `corpus_stats.json`. |
| `teacher_test/` | Phase 0 teacher gate | 32 teacher TTS clips + qwen round-trip that proved the teacher code-switches. |
| `student_eval/` | First listen pack | 32 student clips (8 sentences x 4 voices) + round-trip recall. |
| `eval_big/` | RESULTS: rigorous paired eval | 89 held-out sentences, student vs teacher. `aggregate_report.json` has the headline deltas + CIs; `*_student/teacher.json` are per-metric; `sample_student_clips/` for listening. |
| `_archive_local_validation/` | superseded scratch | early LOCAL validation outputs (calib, stub manifests). The authoritative full outputs are on the box. Kept only for trace. |

## Authoritative full outputs (on the box, not in this repo)

- `data/synth/` — 5,880 synthesized clips (8.56 h)
- `data/filtered/` — filter scores, bin-aware re-accept, train manifest, cosyvoice2 export
- `runs/xtts_hinglish/` — training run + `RELEASE/` (the shippable model)
