#!/usr/bin/env python3
"""S4: assemble the final Hinglish training manifest.

This stage is the join-and-gate point of the pipeline. It reads the three
upstream manifests, joins them on corpus_id and utt_id, applies the non-ASR
gates and the ASR-filter verdict through the shared accept_clip policy, enforces
the synthetic-reliance policy, splits train and dev by corpus_id so no
transcript leaks across the split, and writes the training artifacts.

Inputs (all built by earlier stages, all schema corpus_manifest_v1):
  data/corpus/corpus.jsonl        the text corpus, one row per transcript variant.
  data/synth/synth_index.jsonl    one row per synthesized clip (audio_path, sha256,
                                  duration_s, sample_rate via duration, speaker_id).
  data/filtered/filter_scores.jsonl the qwen filter verdict per clip (asr_hyp,
                                  filter_recall, filter_cer_roman, accept, reject_reason).

Optional input:
  --anchor-manifest path          real Hinglish rows. When supplied, the true 0.5
                                  synthetic cap is enforced. When absent, the cap
                                  degrades to the diversity-spread guardrail plus the
                                  entropy and repetition alarms (Section 4 of the plan).

Outputs:
  data/filtered/train_manifest.json    JSON array, corpus_manifest_v1, the kept rows
                                       tagged train or dev_synth.
  data/filtered/drop_report.json       per-gate drop counts, regen distribution,
                                       per-voice and per-cmi balance, token entropy and
                                       4-gram repetition vs the real spontaneous baseline.
  data/filtered/cosyvoice2/{train,dev}/{wav.scp,text,utt2spk,spk2utt,domain.scp,
                                       manifest.jsonl}   the Kaldi-style trainer export.

The accept decision is made by hinglish_common.accept_clip so a clip judged here
is judged by the identical rule at eval (S5). This stage never calls the network
or a GPU; it only joins and filters files written earlier. It runs offline on the
dry-run synth output, which is what the smoke test below exercises.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import wave
from collections import Counter, defaultdict
from pathlib import Path

# Import the shared module. The file is named common.py and lives beside this
# script; import it whether the script is run from repo root or from its own dir.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import common as hc  # noqa: E402

_REPO = _HERE.parent.parent
_DEF_CORPUS = _REPO / "data" / "corpus" / "corpus.jsonl"
_DEF_SYNTH = _REPO / "data" / "synth" / "synth_index.jsonl"
_DEF_FILTER = _REPO / "data" / "filtered" / "filter_scores.jsonl"
_DEF_EVAL = (_REPO / "data" / "spontaneous_hinglish"
             / "eval_spontaneous_combined_manifest.json")
_DEF_OUT_DIR = _REPO / "data" / "filtered"


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _index_by_utt(rows):
    """Map utt_id -> row, last write wins (resumed appends keep the newest)."""
    out = {}
    for r in rows:
        uid = r.get("utt_id")
        if uid:
            out[uid] = r
    return out


def _index_by_corpus(rows):
    """Map corpus_id -> row for the text corpus (one text row per corpus_id).

    The corpus may carry several rows that share a corpus_id only if a transcript
    was duplicated upstream; we keep the first since they carry identical text.
    """
    out = {}
    for r in rows:
        cid = r.get("corpus_id")
        if cid and cid not in out:
            out[cid] = r
    return out


def _wav_sample_rate(audio_path):
    """Read the sample rate of a WAV without loading audio. None if unreadable.

    Used for the sample-rate gate. The synth_index already records duration; the
    sample rate is read straight from the file header so a mislabelled clip cannot
    slip through. A missing or unreadable file returns None and is reported as a
    bad sample rate so it gets dropped, not silently kept.
    """
    if not audio_path:
        return None
    p = Path(audio_path)
    if not p.exists():
        return None
    try:
        with wave.open(str(p), "rb") as w:
            return w.getframerate()
    except Exception:
        return None


def _corpus_id_of_row(row):
    """corpus_id of a row, deriving it from ref_orig if it was never stamped."""
    cid = row.get("corpus_id")
    if cid:
        return cid
    ref = row.get("ref_orig")
    return hc.corpus_id_of(ref) if ref else None


# ----------------------------------------------------------------------------
# Join + gate
# ----------------------------------------------------------------------------

def join_and_gate(corpus_rows, synth_rows, filter_rows, cfg, sr_cache):
    """Join the three manifests and produce a gated row per synthesized clip.

    For each synth clip we pull the transcript fields from the corpus by
    corpus_id and the ASR verdict from the filter by utt_id, then run the single
    accept_clip policy (recall + cer + duration + sample-rate + optional dnsmos).
    A clip the filter already rejected keeps that filter reason; the non-ASR gates
    here can override only to a non-ASR reason. Returns (gated_rows, gate_counts).
    """
    by_cid = _index_by_corpus(corpus_rows)
    by_uid_filter = _index_by_utt(filter_rows)

    gated = []
    gate_counts = Counter()

    for srow in synth_rows:
        uid = srow.get("utt_id")
        cid = _corpus_id_of_row(srow)
        crow = by_cid.get(cid, {})
        frow = by_uid_filter.get(uid, {})

        # Start from the synth row (it owns audio_path, sha256, duration, speaker)
        # and layer corpus text fields and filter scores on top.
        ref_orig = srow.get("ref_orig") or crow.get("ref_orig")
        ref_surface = srow.get("ref_surface") or crow.get("ref_surface")
        lang_tags = srow.get("lang_tags") or crow.get("lang_tags") or []
        cmi_bin = srow.get("cmi_bin") or crow.get("cmi_bin") or "none"
        cs_density = srow.get("cs_density")
        if cs_density is None:
            cs_density = crow.get("cs_density", 0.0)

        dur = srow.get("duration_s")
        audio_path = srow.get("audio_path")
        sr = sr_cache.get(uid)
        if sr is None:
            sr = _wav_sample_rate(audio_path)
            sr_cache[uid] = sr

        scores = {
            "asr_hyp": frow.get("asr_hyp"),
            "filter_recall": frow.get("filter_recall"),
            "filter_cer_roman": frow.get("filter_cer_roman"),
            "dnsmos": frow.get("dnsmos"),
        }

        # If the filter already rejected for an ASR reason, that reason stands.
        filter_accept = frow.get("accept")
        filter_reason = frow.get("reject_reason")

        # Fail closed when the ASR filter never scored this clip. The S3 ASR gate
        # is the entire quality mechanism for synthetic audio; a missing filter
        # row (empty/partial/missing filter_scores.jsonl, or an S2/S3 utt_id
        # mismatch) leaves filter_recall null, and accept_clip would then skip the
        # recall and cer checks and accept the clip. Treat absent or null-recall
        # coverage as an explicit reject so coverage gaps surface instead of
        # leaking unfiltered audio into the train manifest.
        filtered_here = (uid in by_uid_filter
                         and scores["filter_recall"] is not None)
        if not filtered_here:
            accept, reason = False, "asr_missing"
        else:
            accept, reason = hc.accept_clip(scores, dur, sr, cfg)
            # The shared policy already covers recall/cer/duration/sr/dnsmos. If
            # the filter said no for a reason accept_clip cannot see (it only sees
            # the scores we pass), honour the filter's stored reason.
            if filter_accept is False and accept:
                accept = False
                reason = filter_reason or "recall_low"
            # Align with the _wav_sample_rate contract: an unreadable or missing
            # WAV header yields sr None, which accept_clip cannot gate on. When a
            # row claims an audio_path but the header would not read, reject it as
            # sr_bad rather than letting the sample-rate gate fail open.
            if accept and audio_path and sr is None:
                accept = False
                reason = "sr_bad"

        if accept:
            gate_counts["accept"] += 1
        else:
            gate_counts[reason or "unknown"] += 1

        rep4 = srow.get("rep_4gram")
        if rep4 is None and ref_orig:
            rep4 = hc.ngram_repetition(ref_orig, n=4)

        flags = list(srow.get("flags") or [])
        regen = srow.get("regen_attempt", 0) or 0

        row = hc.new_row(
            utt_id=uid,
            audio_path=audio_path,
            ref_orig=ref_orig,
            ref_surface=ref_surface,
            ref_iso15919=srow.get("ref_iso15919"),
            cmi_bin=cmi_bin,
            cs_density=cs_density,
            lang_tags=lang_tags,
            speaker_id=srow.get("speaker_id"),
            duration_s=dur,
            sha256=srow.get("sha256"),
            dataset=srow.get("dataset") or "synthetic_hinglish",
            partition=srow.get("partition") or "train",
            is_synthetic=True,
            license=srow.get("license") or "synthetic_teacher_tts",
            flags=flags,
            corpus_id=cid,
            speed=srow.get("speed"),
            temp_tier=srow.get("temp_tier"),
            teacher=srow.get("teacher") or hc.DEFAULT_TEACHER,
            chunks=srow.get("chunks"),
            asr_hyp=scores["asr_hyp"],
            filter_recall=scores["filter_recall"],
            filter_cer_roman=scores["filter_cer_roman"],
            filter_wer_raw=frow.get("filter_wer_raw"),
            rep_4gram=rep4,
            accept=accept,
            reject_reason=reason,
            regen_attempt=regen,
        )
        gated.append(row)

    return gated, gate_counts


# ----------------------------------------------------------------------------
# Synthetic-reliance policy
# ----------------------------------------------------------------------------

def apply_reliance_policy(kept_synth, anchor_rows, cfg):
    """Enforce the synthetic-reliance policy. Returns (final_rows, policy_info).

    With real anchor rows supplied, the true cap holds: synthetic rows are capped
    at max_synth_frac of the combined real+synthetic pool, trimming the most
    over-represented (voice, cmi_bin) cells first so the diversity spread survives
    the trim. With no anchor, the cap cannot be honoured (there is no real audio),
    so it degrades to the diversity-spread target and the entropy and repetition
    alarms become the guardrail. Anchor rows are passed through unchanged and
    tagged partition real so the trainer and the report can tell them apart.
    """
    info = {
        "max_synth_frac": cfg.get("max_synth_frac", 0.5),
        "anchor_supplied": bool(anchor_rows),
        "synth_kept_pre_cap": len(kept_synth),
        "synth_dropped_by_cap": 0,
        "real_rows": len(anchor_rows),
        "mode": None,
    }

    if not anchor_rows:
        info["mode"] = "diversity_spread_no_anchor"
        return list(kept_synth), info

    info["mode"] = "true_cap_with_anchor"
    frac = float(cfg.get("max_synth_frac", 0.5))
    n_real = len(anchor_rows)
    # synth <= frac * (synth + real)  ->  synth <= frac/(1-frac) * real
    if frac >= 1.0:
        max_synth = len(kept_synth)
    else:
        max_synth = int((frac / (1.0 - frac)) * n_real)

    if len(kept_synth) <= max_synth:
        return list(kept_synth) + list(anchor_rows), info

    # Trim the most over-represented (voice, cmi_bin) cells first so the kept
    # synthetic set stays balanced rather than dominated by one voice or density.
    keep = _balanced_trim(kept_synth, max_synth)
    info["synth_dropped_by_cap"] = len(kept_synth) - len(keep)
    return list(keep) + list(anchor_rows), info


def _balanced_trim(rows, target):
    """Keep `target` rows, dropping from the largest (voice, cmi_bin) cells first.

    Deterministic: rows inside a cell are ordered by utt_id, and we remove from
    whichever cell is currently largest, round-robin, until we hit the target.
    This preserves the voice and density spread instead of truncating a sorted
    list (which would wipe out whole cells).
    """
    if target >= len(rows):
        return list(rows)
    cells = defaultdict(list)
    for r in rows:
        key = (r.get("speaker_id"), r.get("cmi_bin"))
        cells[key].append(r)
    for key in cells:
        cells[key].sort(key=lambda r: r.get("utt_id") or "")

    total = len(rows)
    # Remove the last element of the current-largest cell until at target.
    while total > target:
        largest = max(cells, key=lambda k: len(cells[k]))
        if not cells[largest]:
            break
        cells[largest].pop()
        total -= 1

    out = []
    for key in cells:
        out.extend(cells[key])
    out.sort(key=lambda r: r.get("utt_id") or "")
    return out


# ----------------------------------------------------------------------------
# Train / dev split by corpus_id (no text leakage)
# ----------------------------------------------------------------------------

def split_by_corpus(rows, dev_frac, seed):
    """Split rows into train and dev by corpus_id so no transcript leaks across.

    All clips of one transcript (every voice, speed, regen variant share a
    corpus_id) land on the same side. Deterministic given the seed. Real anchor
    rows always go to train; only synthetic rows form dev_synth, because dev is
    used to watch the synthetic distribution during training.
    """
    synth_cids = sorted({r["corpus_id"] for r in rows
                         if r.get("is_synthetic") and r.get("corpus_id")})
    rng = random.Random(seed)
    rng.shuffle(synth_cids)
    n_dev = int(round(len(synth_cids) * dev_frac))
    dev_cids = set(synth_cids[:n_dev])

    for r in rows:
        if not r.get("is_synthetic"):
            r["partition"] = "train"
            continue
        cid = r.get("corpus_id")
        r["partition"] = "dev_synth" if cid in dev_cids else "train"
    return rows


# ----------------------------------------------------------------------------
# Diagnostics vs the real baseline
# ----------------------------------------------------------------------------

def diagnostics(kept_rows, eval_path):
    """Token entropy and 4-gram repetition of the kept corpus vs the real set.

    The real spontaneous eval set is the baseline; a meaningful entropy drop or a
    repetition rise on the synthetic side is the Synthetic-Erosion alarm. Returns
    a dict suitable for the drop report. If the eval set is unavailable the real
    baseline fields are null but the synthetic numbers still compute.

    Synthetic rows repeat the same transcript across voices, speeds and regen
    attempts, so a per-clip sample would be weighted by synthesis multiplicity
    rather than by distinct transcripts. We dedup by corpus_id first (one
    transcript per corpus_id) so the reported entropy and repetition are the
    per-transcript figures a reader expects.
    """
    seen_cids = set()
    synth_texts = []
    for r in kept_rows:
        if not (r.get("is_synthetic") and r.get("ref_orig")):
            continue
        cid = r.get("corpus_id") or _corpus_id_of_row(r)
        if cid in seen_cids:
            continue
        seen_cids.add(cid)
        synth_texts.append(r["ref_orig"])
    real_texts = []
    if eval_path and Path(eval_path).exists():
        real_texts = [r.get("ref_orig") for r in hc.read_manifest(eval_path)
                      if r.get("ref_orig")]

    def rep_mean(texts):
        if not texts:
            return None
        vals = [hc.ngram_repetition(t, n=4) for t in texts]
        return sum(vals) / len(vals)

    synth_entropy = hc.token_entropy(synth_texts) if synth_texts else 0.0
    real_entropy = hc.token_entropy(real_texts) if real_texts else None
    out = {
        "synth_token_entropy": round(synth_entropy, 4),
        "real_token_entropy": (round(real_entropy, 4)
                               if real_entropy is not None else None),
        "token_entropy_ratio": (round(synth_entropy / real_entropy, 4)
                                if real_entropy else None),
        "synth_rep_4gram_mean": (round(rep_mean(synth_texts), 5)
                                 if synth_texts else None),
        "real_rep_4gram_mean": (round(rep_mean(real_texts), 5)
                                if real_texts else None),
        "synth_distinct_transcripts": len(synth_texts),
        "metric_basis": "per_transcript_deduped_by_corpus_id",
    }
    # A simple alarm flag the orchestrator can read without re-deriving.
    out["erosion_alarm"] = bool(
        real_entropy and synth_entropy < 0.85 * real_entropy)
    return out


def balance_tables(kept_rows):
    """Per-voice and per-cmi_bin counts (synthetic rows only) for the report."""
    by_voice = Counter()
    by_cmi = Counter()
    by_voice_cmi = Counter()
    by_speed = Counter()
    by_regen = Counter()
    for r in kept_rows:
        if not r.get("is_synthetic"):
            continue
        by_voice[r.get("speaker_id")] += 1
        by_cmi[r.get("cmi_bin")] += 1
        by_voice_cmi["%s|%s" % (r.get("speaker_id"), r.get("cmi_bin"))] += 1
        by_speed[str(r.get("speed"))] += 1
        by_regen[str(r.get("regen_attempt", 0))] += 1
    return {
        "per_voice": dict(by_voice),
        "per_cmi_bin": dict(by_cmi),
        "per_voice_cmi_bin": dict(by_voice_cmi),
        "per_speed": dict(by_speed),
        "regen_distribution": dict(by_regen),
    }


# ----------------------------------------------------------------------------
# CosyVoice2 Kaldi-style export
# ----------------------------------------------------------------------------

def write_cosyvoice2(rows, out_dir):
    """Write the CosyVoice2 Kaldi-style export under out_dir/{train,dev}.

    For each split we write wav.scp (utt_id -> audio_path), text (utt_id ->
    ref_orig, the Devanagari/Latin training transcript), utt2spk and its inverse
    spk2utt (speaker_id is the teacher voice and becomes the speaker label),
    domain.scp (utt_id -> synthetic or real so a trainer can weight or hold out
    by domain), and manifest.jsonl (the full rows for that split). Lines are
    sorted by utt_id, which Kaldi tools expect.
    """
    splits = {
        "train": [r for r in rows if r.get("partition") in ("train",)],
        "dev": [r for r in rows if r.get("partition") == "dev_synth"],
    }
    written = {}
    for split, srows in splits.items():
        d = Path(out_dir) / split
        d.mkdir(parents=True, exist_ok=True)
        srows = sorted(srows, key=lambda r: r.get("utt_id") or "")

        spk2utt = defaultdict(list)
        with open(d / "wav.scp", "w", encoding="utf-8") as fw, \
             open(d / "text", "w", encoding="utf-8") as ft, \
             open(d / "utt2spk", "w", encoding="utf-8") as fu, \
             open(d / "domain.scp", "w", encoding="utf-8") as fd:
            for r in srows:
                uid = r["utt_id"]
                spk = r.get("speaker_id") or "unknown"
                fw.write("%s %s\n" % (uid, r.get("audio_path") or ""))
                ft.write("%s %s\n" % (uid, (r.get("ref_orig") or "").strip()))
                fu.write("%s %s\n" % (uid, spk))
                domain = "synthetic" if r.get("is_synthetic") else "real"
                fd.write("%s %s\n" % (uid, domain))
                spk2utt[spk].append(uid)

        with open(d / "spk2utt", "w", encoding="utf-8") as fs:
            for spk in sorted(spk2utt):
                fs.write("%s %s\n" % (spk, " ".join(sorted(spk2utt[spk]))))

        hc.write_manifest(str(d / "manifest.jsonl"), srows)
        written[split] = len(srows)
    return written


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="S4: join, gate, split and export the Hinglish train manifest.")
    ap.add_argument("--corpus", default=str(_DEF_CORPUS),
                    help="corpus.jsonl from S1.")
    ap.add_argument("--synth-index", default=str(_DEF_SYNTH),
                    help="synth_index.jsonl from S2.")
    ap.add_argument("--filter-scores", default=str(_DEF_FILTER),
                    help="filter_scores.jsonl from S3.")
    ap.add_argument("--eval-manifest", default=str(_DEF_EVAL),
                    help="real spontaneous eval set, the entropy/rep baseline.")
    ap.add_argument("--anchor-manifest", default=None,
                    help="optional real Hinglish rows; enables the true 0.5 cap.")
    ap.add_argument("--config", default=None,
                    help="experiment config json (thresholds, seed, caps).")
    ap.add_argument("--out-dir", default=str(_DEF_OUT_DIR),
                    help="output directory for the manifest, report and export.")
    ap.add_argument("--dev-frac", type=float, default=0.05,
                    help="fraction of synthetic corpus_ids held out as dev_synth.")
    ap.add_argument("--require-audio", action="store_true",
                    help="drop rows whose audio file is missing on disk.")
    args = ap.parse_args(argv)

    # Config: defaults from common.load_config (calibrated thresholds when present)
    if args.config:
        cfg = hc.load_config(args.config)
    else:
        cfg = {
            "tau_recall": 0.80, "tau_cer": 0.30, "min_s": 3.0, "max_s": 30.0,
            "dnsmos_min": None, "max_synth_frac": 0.5, "split_seed": 20260617,
        }
        calib = Path(args.out_dir) / "calib_report.json"
        if calib.exists():
            try:
                c = json.loads(calib.read_text(encoding="utf-8"))
                cfg["tau_recall"] = c.get("tau_recall", cfg["tau_recall"])
                cfg["tau_cer"] = c.get("tau_cer", cfg["tau_cer"])
            except Exception:
                pass

    corpus_rows = hc.read_manifest(args.corpus)
    synth_rows = hc.read_manifest(args.synth_index)
    filter_rows = hc.read_manifest(args.filter_scores)
    anchor_rows = (hc.read_manifest(args.anchor_manifest)
                   if args.anchor_manifest else [])

    if not synth_rows:
        print("WARNING: no synth rows found at %s" % args.synth_index,
              file=sys.stderr)

    # Normalise anchor rows so they validate and split cleanly: stamp is_synthetic
    # false, give each a corpus_id, and re-key partition to train.
    norm_anchor = []
    for r in anchor_rows:
        r = dict(r)
        r["is_synthetic"] = False
        if not r.get("corpus_id"):
            r["corpus_id"] = _corpus_id_of_row(r)
        r["partition"] = "train"
        # backfill additive fields so it is a complete v1+additive row
        for f in hc.ADDITIVE_FIELDS:
            r.setdefault(f, None if f != "regen_attempt" else 0)
        norm_anchor.append(r)

    sr_cache = {}
    gated, gate_counts = join_and_gate(
        corpus_rows, synth_rows, filter_rows, cfg, sr_cache)

    # Keep accepted rows; optionally require the audio file to exist on disk.
    kept_synth = []
    for r in gated:
        if not r.get("accept"):
            continue
        if args.require_audio:
            ap_ = r.get("audio_path")
            if not ap_ or not Path(ap_).exists():
                gate_counts["audio_missing"] += 1
                continue
        kept_synth.append(r)

    final_rows, policy_info = apply_reliance_policy(kept_synth, norm_anchor, cfg)

    final_rows = split_by_corpus(
        final_rows, args.dev_frac, cfg.get("split_seed", 20260617))

    # Validate every row before writing so a bad row never reaches the trainer.
    # Synthetic rows that survived the gate must also pass the 'filtered' profile,
    # which requires a non-null filter_recall and accept. This is the schema-level
    # backstop for the fail-open path: a row admitted with filter_recall null would
    # be caught here even if the gate logic regressed.
    schema_problems = []
    filter_coverage_gaps = 0
    for r in final_rows:
        if r.get("is_synthetic"):
            probs = hc.validate_row(r, profile="train")
            filt_probs = hc.validate_row(r, profile="filtered")
            if filt_probs:
                filter_coverage_gaps += 1
                probs = probs + filt_probs
            if probs:
                schema_problems.append((r.get("utt_id"), probs))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Train manifest is a JSON array (corpus_manifest_v1), drop-in for the loader.
    train_path = out_dir / "train_manifest.json"
    hc.write_manifest(str(train_path), final_rows, json_array=True)

    cosy = write_cosyvoice2(final_rows, out_dir / "cosyvoice2")

    diag = diagnostics(final_rows, args.eval_manifest)
    balance = balance_tables(final_rows)

    n_train = sum(1 for r in final_rows if r.get("partition") == "train")
    n_dev = sum(1 for r in final_rows if r.get("partition") == "dev_synth")
    n_real = sum(1 for r in final_rows if not r.get("is_synthetic"))
    n_synth = sum(1 for r in final_rows if r.get("is_synthetic"))
    synth_frac = (n_synth / (n_synth + n_real)) if (n_synth + n_real) else 0.0

    report = {
        "inputs": {
            "corpus": args.corpus,
            "synth_index": args.synth_index,
            "filter_scores": args.filter_scores,
            "anchor_manifest": args.anchor_manifest,
            "eval_manifest": args.eval_manifest,
        },
        "config": {
            "tau_recall": cfg.get("tau_recall"),
            "tau_cer": cfg.get("tau_cer"),
            "min_s": cfg.get("min_s"),
            "max_s": cfg.get("max_s"),
            "dnsmos_min": cfg.get("dnsmos_min"),
            "max_synth_frac": cfg.get("max_synth_frac"),
            "split_seed": cfg.get("split_seed", 20260617),
            "dev_frac": args.dev_frac,
        },
        "counts": {
            "synth_clips_in": len(synth_rows),
            "gated": len(gated),
            "accepted_synth": len(kept_synth),
            "final_total": len(final_rows),
            "final_train": n_train,
            "final_dev_synth": n_dev,
            "final_real": n_real,
            "final_synthetic": n_synth,
            "synthetic_fraction": round(synth_frac, 4),
        },
        "gate_drops": dict(gate_counts),
        "reliance_policy": policy_info,
        "balance": balance,
        "diagnostics_vs_real_baseline": diag,
        "cosyvoice2_export": cosy,
        "schema_problems": [
            {"utt_id": u, "problems": p} for u, p in schema_problems[:50]
        ],
        "schema_problem_count": len(schema_problems),
        "filter_coverage_gaps": filter_coverage_gaps,
    }
    report_path = out_dir / "drop_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # Console summary for the operator / orchestrator log.
    print("S4 assemble_manifest")
    print("  synth clips in        : %d" % len(synth_rows))
    print("  accepted (post-gate)  : %d" % len(kept_synth))
    print("  reliance mode         : %s" % policy_info["mode"])
    if policy_info["synth_dropped_by_cap"]:
        print("  dropped by synth cap  : %d" % policy_info["synth_dropped_by_cap"])
    print("  final train / dev     : %d / %d" % (n_train, n_dev))
    print("  real / synthetic      : %d / %d (synth_frac=%.3f)"
          % (n_real, n_synth, synth_frac))
    print("  gate drops            : %s" % dict(gate_counts))
    print("  entropy synth/real    : %s / %s (alarm=%s)"
          % (diag["synth_token_entropy"], diag["real_token_entropy"],
             diag["erosion_alarm"]))
    print("  cosyvoice2 export     : %s" % cosy)
    print("  train_manifest.json   : %s" % train_path)
    print("  drop_report.json      : %s" % report_path)
    if schema_problems:
        print("  WARNING: %d schema problems (see drop_report.json)"
              % len(schema_problems))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
