#!/usr/bin/env python3
"""Round-trip the fine-tuned student's output through Qwen3-ASR to measure intelligibility.

For each student clip: ASR -> convention-robust content_word_recall + cer vs the intended text,
grouped by cs_mode. Compares against the teacher (teacher TTS) Phase-0 reference recall per bin.
Needs the qwen venv (GPU). Run:
  CUDA_VISIBLE_DEVICES=6 <qwen-venv>/bin/python \
      scripts/hinglish/08_student_eval.py --manifest data/student_eval/student_manifest.jsonl
"""
from __future__ import annotations
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
import soundfile as sf, torch
from qwen_asr import Qwen3ASRModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

# teacher (teacher TTS) recall per cmi_bin from the Phase-0 verified-good round-trip
TEACHER_REF = {"cs_none": 0.969, "cs_low": 1.0, "cs_med": 0.889, "cs_high": 0.769,
               "tech": 0.827, "question": 1.0, "emotion": 0.969, "numbers": 0.694}


def cer(ref, hyp):
    a = common.to_compare_space(ref); b = common.to_compare_space(hyp)
    if not a:
        return 1.0
    # char-level Levenshtein
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, dp[0] = dp[0], i
        for j, cb in enumerate(b, 1):
            cur = dp[j]; dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (ca != cb)); prev = cur
    return dp[-1] / max(1, len(a))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default="data/student_eval/student_roundtrip.json")
    args = ap.parse_args()

    man = [json.loads(l) for l in open(args.manifest, encoding="utf-8") if l.strip()]
    model = Qwen3ASRModel.from_pretrained("Qwen/Qwen3-ASR-1.7B", dtype=torch.bfloat16,
                                          device_map="cuda:0", max_new_tokens=256)
    rows = []
    for m in man:
        audio, sr = sf.read(m["wav"], dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        hyp = model.transcribe(audio=(audio, sr), language="Hindi")[0].text.strip()
        rec = common.content_word_recall(m["ref_text"], hyp)
        rows.append({**m, "hyp": hyp, "recall": round(rec, 3), "cer": round(cer(m["ref_text"], hyp), 3)})
        print(f"  {m['utt_id']:22s} recall={rec:.2f} cer={rows[-1]['cer']:.2f}  {hyp[:55]}")

    bybin = defaultdict(list)
    for r in rows:
        bybin[r["cs_mode"]].append(r["recall"])
    overall = sum(r["recall"] for r in rows) / len(rows)
    print(f"\nSTUDENT intelligibility (recall), vs teacher Phase-0 ref:")
    for b in sorted(bybin):
        s = sum(bybin[b]) / len(bybin[b]); t = TEACHER_REF.get(b)
        tx = f"  teacher~{t:.2f}  delta {s-t:+.2f}" if t else ""
        print(f"  {b:9s} student={s:.3f}{tx}")
    print(f"  OVERALL student recall={overall:.3f}")
    Path(args.out).write_text(json.dumps({"overall_recall": round(overall, 3),
        "by_cs_mode": {b: round(sum(v)/len(v), 3) for b, v in bybin.items()},
        "teacher_ref": TEACHER_REF, "rows": rows}, indent=2, ensure_ascii=False))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
