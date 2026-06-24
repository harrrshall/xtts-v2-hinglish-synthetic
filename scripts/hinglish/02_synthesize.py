#!/usr/bin/env python3
"""S2: synthesize the corpus into teacher audio (the teacher TTS).

This stage turns the text corpus from S1 into a synthesis plan and then, unless
running in --dry-run, into one WAV per clip. The flow is deliberately two
phased so the spend is reviewable before any money is spent:

  1. PLAN. Read data/corpus/corpus.jsonl. For every corpus row, expand across
     the configured voices, speeds, and temperature tiers. For each combination
     mint a deterministic utt_id, chunk the transcript to <=250 chars on clause
     boundaries, and write one plan row to data/synth/synth_plan.jsonl. The plan
     carries everything needed to execute (voice, speed, tier, chunks) and is
     human-reviewable: you can read it, count the rows, and estimate the API
     spend before committing.

  2. EXECUTE. Walk the plan. For each clip, POST every chunk to teacher TTS,
     write each chunk WAV to a temp area, concatenate them into
     data/synth/wav/<utt_id>.wav at 24 kHz mono, compute duration_s and the
     audio sha256, and append a schema-valid row to data/synth/synth_index.jsonl.

The stage is idempotent. Plan rows whose utt_id already sits in the plan file
are not re-planned; clips whose utt_id already sits in the index are not
re-synthesized. So an interrupted run resumes safely without duplicating work or
re-spending on the API.

--dry-run swaps the live POST for synth_request(dry_run=True), which returns a
valid silent 24 kHz WAV. That writes real WAV files and schema-valid index rows,
so every downstream stage (S3 filter, S4 assemble, S5 eval) can run end to end
with no GPU and no API key. The plan written in dry-run is identical to the plan
a live run would execute, so a dry-run is also a faithful spend preview.

Secrets: the API key is read only inside common.synth_request from the env var
TEACHER_TTS_API_KEY. It is never read, logged, or printed here. A live run without
the key set fails inside synth_request with a clear message and the plan is left
intact so you can set the key and resume.

A second teacher backend (the optional CosyVoice2 blend named in the schema's
teacher field) slots in at exactly one place: _synthesize_clip dispatches on the
row's teacher value. Everything else (planning, chunking, concat, indexing) is
backend agnostic.

Run from the repo root:

  # offline preview: build the plan and silent WAVs, no API, no key needed
  python3 scripts/hinglish/02_synthesize.py --dry-run

  # plan only, do not synthesize at all (review the spend first)
  python3 scripts/hinglish/02_synthesize.py --plan-only

  # live run (needs TEACHER_TTS_API_KEY in the env)
  python3 scripts/hinglish/02_synthesize.py --execute

  # self-contained smoke test (writes to a temp dir, no real corpus needed)
  python3 scripts/hinglish/02_synthesize.py --smoke-test
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import time
import wave
from pathlib import Path

# Import the shared module. The script lives in scripts/hinglish/ next to it.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import common  # noqa: E402  (path set above so the import resolves)

_REPO_ROOT = _HERE.parent.parent
_DEFAULT_CORPUS = _REPO_ROOT / "data" / "corpus" / "corpus.jsonl"
_DEFAULT_SYNTH_DIR = _REPO_ROOT / "data" / "synth"
_DEFAULT_CONFIG = _HERE / "configs" / "experiment.example.json"


# ----------------------------------------------------------------------------
# Planning
# ----------------------------------------------------------------------------

def _normalize_temp_tiers(cfg: dict) -> list:
    """Return the list of temperature tiers to expand across.

    The config may carry temp_tiers as a list that includes JSON null for the
    plain SFT slot. We keep null as Python None so mint_utt_id stamps the 'tx'
    marker. An empty or missing list degrades to a single None tier so a plain
    run still produces one clip per voice/speed.
    """
    tiers = cfg.get("temp_tiers")
    if not tiers:
        return [None]
    out = []
    for t in tiers:
        out.append(None if t in (None, "", "null", "x") else str(t))
    # de-duplicate while keeping order so the plan is stable
    seen = set()
    uniq = []
    for t in out:
        key = "x" if t is None else t
        if key not in seen:
            seen.add(key)
            uniq.append(t)
    return uniq


def _plan_row_for(corpus_row: dict, voice: str, speed: float,
                  temp_tier, cfg: dict) -> dict:
    """Build one synthesis-plan row from a corpus row and one expansion arm.

    The plan row is a thin synthesis record, not a full manifest row: it holds
    the join key (corpus_id), the minted utt_id, the text and its <=250-char
    chunks, and the synthesis parameters. It is intentionally small and readable
    so the plan file can be eyeballed for spend before execution. The full
    schema-valid manifest row is constructed later, after synthesis, in the
    index so it can carry duration_s, sha256, and audio_path.
    """
    ref_orig = corpus_row["ref_orig"]
    corpus_id = corpus_row.get("corpus_id") or common.corpus_id_of(ref_orig)
    regen_attempt = int(corpus_row.get("regen_attempt", 0) or 0)
    utt_id = common.mint_utt_id(corpus_id, voice, speed, temp_tier,
                                regen_attempt)
    chunks = common.chunk_text(ref_orig, cfg.get("max_chars", 250))
    return {
        "utt_id": utt_id,
        "corpus_id": corpus_id,
        "ref_orig": ref_orig,
        "speaker_id": voice,
        "speed": float(speed),
        "temp_tier": temp_tier,
        "teacher": cfg.get("teacher", common.DEFAULT_TEACHER),
        "sample_rate": int(cfg.get("sample_rate", 24000)),
        "regen_attempt": regen_attempt,
        "chunks": chunks,
        "n_chunks": len(chunks),
        # carry the corpus metadata forward so the index row never needs to
        # re-derive it (and a v1-valid row can be stamped after synthesis)
        "ref_surface": corpus_row.get("ref_surface"),
        "lang_tags": corpus_row.get("lang_tags"),
        "cmi_bin": corpus_row.get("cmi_bin"),
        "cs_density": corpus_row.get("cs_density"),
        "rep_4gram": corpus_row.get("rep_4gram"),
        "dataset": corpus_row.get("dataset", "synthetic_hinglish"),
    }


def build_plan(corpus_rows: list, cfg: dict, voices=None, speeds=None,
               temp_tiers=None) -> list:
    """Expand the corpus into the full synthesis plan (list of plan rows).

    The expansion is the cross product corpus x voices x speeds x temp_tiers.
    voices/speeds/temp_tiers override the config when given (the CLI passes the
    --voices and --speeds flags through here) so an ablation arm can be planned
    without editing the config. Rows with an empty transcript or no chunks are
    skipped (they cannot be synthesized) and counted by the caller.
    """
    voices = voices or cfg.get("voices", list(common.KNOWN_VOICES))
    speeds = speeds or cfg.get("speeds", [1.0])
    tiers = temp_tiers if temp_tiers is not None else _normalize_temp_tiers(cfg)

    plan = []
    for crow in corpus_rows:
        ref = (crow.get("ref_orig") or "").strip()
        if not ref:
            continue
        for voice in voices:
            for speed in speeds:
                for tier in tiers:
                    prow = _plan_row_for(crow, voice, float(speed), tier, cfg)
                    if not prow["chunks"]:
                        continue
                    plan.append(prow)
    return plan


def write_plan(plan_path: str, plan_rows: list, resume: bool = True) -> int:
    """Write (or extend) the plan file, returning how many NEW rows were added.

    When resume is true, plan rows whose utt_id already exists in the file are
    not re-written, so re-planning after a corpus top-up only appends the new
    arms. The plan is JSONL via the shared atomic writer.
    """
    existing = common.resume_done_ids(plan_path) if resume else set()
    new_rows = [r for r in plan_rows if r["utt_id"] not in existing]
    if not new_rows:
        return 0
    # append when extending an existing plan, fresh write otherwise
    append = bool(existing) and Path(plan_path).exists()
    common.write_manifest(plan_path, new_rows, append=append)
    return len(new_rows)


# ----------------------------------------------------------------------------
# Execution
# ----------------------------------------------------------------------------

def _sha256_file(path: str) -> str:
    """Stream the sha256 of a file so a large WAV does not load into memory."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _synthesize_clip(plan_row: dict, wav_dir: Path, dry_run: bool,
                     max_retries: int) -> dict:
    """Synthesize one clip from a plan row and return a schema-valid index row.

    Each chunk is POSTed (or stubbed silently in dry-run), written to a temp
    WAV, and the temp WAVs are concatenated into the final clip WAV. The audio
    metadata (duration_s, sha256) is computed from the concatenated file. The
    returned row is built only through common.new_row so the schema cannot
    drift; it is stamped is_synthetic=True with the voice as speaker_id (which
    becomes utt2spk downstream) and the teacher engine recorded separately.
    """
    utt_id = plan_row["utt_id"]
    voice = plan_row["speaker_id"]
    speed = float(plan_row["speed"])
    sample_rate = int(plan_row.get("sample_rate", 24000))
    teacher = plan_row.get("teacher", common.DEFAULT_TEACHER)
    chunks = plan_row["chunks"]

    out_wav = wav_dir / (utt_id + ".wav")

    # Dispatch point for a second teacher backend. teacher TTS is the only
    # backend wired today; another engine would branch here on `teacher`.
    if teacher not in (common.DEFAULT_TEACHER, "teacher_tts"):
        raise NotImplementedError(
            "teacher backend %r is not wired; only teacher TTS is implemented"
            % teacher)

    with tempfile.TemporaryDirectory() as td:
        chunk_paths = []
        for idx, chunk in enumerate(chunks):
            # synth_request enforces the 250-char limit and never logs the key
            audio_bytes = common.synth_request(
                chunk, voice, speed=speed, sample_rate=sample_rate,
                dry_run=dry_run, max_retries=max_retries)
            cp = Path(td) / ("chunk_%03d.wav" % idx)
            cp.write_bytes(audio_bytes)
            chunk_paths.append(str(cp))
        duration_s = common.concat_wavs(chunk_paths, str(out_wav), sample_rate)

    sha = _sha256_file(str(out_wav))
    rel_path = str(out_wav.relative_to(_REPO_ROOT)) \
        if str(out_wav).startswith(str(_REPO_ROOT)) else str(out_wav)

    flags = ["run_synth"]
    regen = int(plan_row.get("regen_attempt", 0) or 0)
    flags.append("regen_attempt:%d" % regen)
    if dry_run:
        flags.append("dry_run")

    row = common.new_row(
        utt_id=utt_id,
        audio_path=rel_path,
        ref_orig=plan_row["ref_orig"],
        ref_surface=plan_row.get("ref_surface"),
        ref_iso15919=None,
        cmi_bin=plan_row.get("cmi_bin"),
        cs_density=plan_row.get("cs_density"),
        lang_tags=plan_row.get("lang_tags"),
        speaker_id=voice,
        duration_s=round(duration_s, 3),
        sha256=sha,
        dataset=plan_row.get("dataset", "synthetic_hinglish"),
        partition="train",
        is_synthetic=True,
        license="synthetic_" + teacher,
        flags=flags,
        # additive fields
        corpus_id=plan_row["corpus_id"],
        speed=speed,
        temp_tier=plan_row.get("temp_tier"),
        teacher=teacher,
        chunks=chunks,
        rep_4gram=plan_row.get("rep_4gram"),
        regen_attempt=regen,
    )
    problems = common.validate_row(row, profile="synth")
    if problems:
        raise ValueError("synth row %s failed validation: %s"
                         % (utt_id, problems))
    return row


def execute_plan(plan_rows: list, wav_dir: Path, index_path: str,
                 dry_run: bool, max_retries: int = 4, limit: int = 0,
                 progress_every: int = 25) -> dict:
    """Synthesize every not-yet-done plan row, appending to the index.

    Resumable: utt_ids already in the index are skipped. The index is appended
    one row at a time right after each clip is written, so a crash mid-run loses
    at most the in-flight clip and never the index of what is already on disk.
    Returns a small stats dict. On a per-clip error the clip is skipped, the
    error counted, and the run continues (a single bad transcript must not abort
    a long synthesis job).
    """
    wav_dir.mkdir(parents=True, exist_ok=True)
    done = common.resume_done_ids(index_path)

    todo = [r for r in plan_rows if r["utt_id"] not in done]
    if limit and limit > 0:
        todo = todo[:limit]

    stats = {"planned": len(plan_rows), "already_done": len(done),
             "attempted": len(todo), "synthesized": 0, "skipped_done": 0,
             "errors": 0, "error_ids": []}

    for i, prow in enumerate(todo, 1):
        try:
            row = _synthesize_clip(prow, wav_dir, dry_run, max_retries)
            common.write_manifest(index_path, [row], append=True)
            stats["synthesized"] += 1
        except Exception as e:  # keep going on a single-clip failure
            stats["errors"] += 1
            if len(stats["error_ids"]) < 50:
                stats["error_ids"].append(prow["utt_id"])
            print("  [error] %s: %s" % (prow["utt_id"], e), flush=True)
        if progress_every and (i % progress_every == 0 or i == len(todo)):
            print("  progress %d/%d synthesized=%d errors=%d"
                  % (i, len(todo), stats["synthesized"], stats["errors"]),
                  flush=True)
    return stats


# ----------------------------------------------------------------------------
# Spend estimate (printed with the plan so spend is reviewable)
# ----------------------------------------------------------------------------

def summarize_plan(plan_rows: list) -> dict:
    """Compute a reviewable summary of the plan: clip and chunk counts.

    Chunks are the unit billed by the API (one POST per chunk). The summary
    breaks counts down per voice and per speed so an unbalanced expansion is
    visible before any spend.
    """
    total_clips = len(plan_rows)
    total_chunks = sum(r["n_chunks"] for r in plan_rows)
    total_chars = sum(sum(len(c) for c in r["chunks"]) for r in plan_rows)
    per_voice = {}
    per_speed = {}
    for r in plan_rows:
        per_voice[r["speaker_id"]] = per_voice.get(r["speaker_id"], 0) + 1
        sp = "%.1f" % r["speed"]
        per_speed[sp] = per_speed.get(sp, 0) + 1
    return {
        "clips": total_clips,
        "api_calls_chunks": total_chunks,
        "total_chars": total_chars,
        "per_voice": per_voice,
        "per_speed": per_speed,
    }


def _print_summary(summary: dict, dry_run: bool, executed: bool) -> None:
    """Print the plan summary in a compact, readable block."""
    print("synthesis plan summary")
    print("  clips                : %d" % summary["clips"])
    print("  chunk API calls      : %d" % summary["api_calls_chunks"])
    print("  total characters     : %d" % summary["total_chars"])
    print("  per voice            : %s" % json.dumps(summary["per_voice"],
                                                     sort_keys=True))
    print("  per speed            : %s" % json.dumps(summary["per_speed"],
                                                     sort_keys=True))
    if not executed:
        mode = "dry-run (silent WAVs)" if dry_run else "plan only (no synthesis)"
        print("  mode                 : %s" % mode)


# ----------------------------------------------------------------------------
# Smoke test (self-contained, no real corpus, no API, no GPU)
# ----------------------------------------------------------------------------

def _smoke_test() -> int:
    """End-to-end offline smoke test in a temp dir.

    Fabricates a tiny corpus (including one long transcript that forces multi
    chunk concatenation), builds the plan, executes it in dry-run, and asserts
    the plan/index/WAVs are correct and resumable. Proves S2 runs clean with no
    real corpus, no API key, and no GPU.
    """
    failures = []

    def check(name, cond):
        print("  [%s] %s" % ("PASS" if cond else "FAIL", name))
        if not cond:
            failures.append(name)

    cfg = {
        "voices": ["maya", "arjun"],
        "speeds": [1.0, 1.1],
        "temp_tiers": [None],
        "teacher": common.DEFAULT_TEACHER,
        "sample_rate": 24000,
        "max_chars": 250,
    }

    # one short, one long (forces >1 chunk so concat is exercised)
    long_ref = ("यह एक बहुत लंबा वाक्य है जो coaching center के बारे में है, "
                * 8).strip()
    corpus_rows = [
        common.new_row(
            utt_id="seed1", audio_path=None,
            ref_orig="ये coaching center बहुत अच्छा है।",
            ref_surface=None, ref_iso15919=None, cmi_bin="med", cs_density=0.2,
            lang_tags=common.tag_languages("ये coaching center बहुत अच्छा है।"),
            speaker_id="src", duration_s=None, sha256=None,
            dataset="synthetic_hinglish", partition="train",
            is_synthetic=False, license="seed", flags=[],
            corpus_id=common.corpus_id_of("ये coaching center बहुत अच्छा है।")),
        common.new_row(
            utt_id="seed2", audio_path=None, ref_orig=long_ref,
            ref_surface=None, ref_iso15919=None, cmi_bin="low", cs_density=0.1,
            lang_tags=common.tag_languages(long_ref),
            speaker_id="src", duration_s=None, sha256=None,
            dataset="synthetic_hinglish", partition="train",
            is_synthetic=False, license="seed", flags=[],
            corpus_id=common.corpus_id_of(long_ref)),
    ]

    with tempfile.TemporaryDirectory() as td:
        plan_path = str(Path(td) / "synth_plan.jsonl")
        index_path = str(Path(td) / "synth_index.jsonl")
        wav_dir = Path(td) / "wav"

        plan = build_plan(corpus_rows, cfg)
        # 2 corpus x 2 voices x 2 speeds x 1 tier = 8 clips
        check("plan cross-product size == 8", len(plan) == 8)
        check("long transcript chunked to >1 chunk",
              any(r["n_chunks"] > 1 for r in plan
                  if r["corpus_id"] == corpus_rows[1]["corpus_id"]))
        check("all chunks <= 250 chars",
              all(all(len(c) <= 250 for c in r["chunks"]) for r in plan))
        check("utt_ids unique", len({r["utt_id"] for r in plan}) == len(plan))

        added = write_plan(plan_path, plan)
        check("write_plan added all rows", added == 8)
        # re-write is a no-op (idempotent planning)
        again = write_plan(plan_path, plan)
        check("re-plan adds nothing (idempotent)", again == 0)

        summary = summarize_plan(plan)
        check("summary clips matches", summary["clips"] == 8)
        check("summary counts chunks", summary["api_calls_chunks"] >= 8)

        stats = execute_plan(plan, wav_dir, index_path, dry_run=True,
                             progress_every=0)
        check("execute synthesized all", stats["synthesized"] == 8)
        check("execute had no errors", stats["errors"] == 0)

        index = common.read_manifest(index_path)
        check("index has 8 rows", len(index) == 8)
        check("every index row is synth-valid",
              all(common.validate_row(r, "synth") == [] for r in index))
        check("every index row is_synthetic", all(r["is_synthetic"]
                                                   for r in index))
        check("speaker_id holds the voice",
              {r["speaker_id"] for r in index} == {"maya", "arjun"})
        check("teacher recorded separately",
              all(r["teacher"] == common.DEFAULT_TEACHER for r in index))

        # WAV files exist and are 24 kHz mono with positive duration
        all_wavs_ok = True
        for r in index:
            # resolve the row's audio_path field (what downstream stages consume)
            # so the smoke test verifies audio_path is correct, not a re-derived
            # path. It is absolute when the WAV is outside the repo (temp dir),
            # otherwise relative to the repo root.
            ap = Path(r["audio_path"])
            wpath = ap if ap.is_absolute() else (_REPO_ROOT / ap)
            if not wpath.exists():
                all_wavs_ok = False
                break
            with wave.open(str(wpath), "rb") as w:
                if w.getframerate() != 24000 or w.getnchannels() != 1:
                    all_wavs_ok = False
                    break
            if not (r["duration_s"] and r["duration_s"] > 0):
                all_wavs_ok = False
                break
            if not r["sha256"]:
                all_wavs_ok = False
                break
        check("all WAVs written 24 kHz mono with duration+sha", all_wavs_ok)

        # resume: a second execute does nothing new
        stats2 = execute_plan(plan, wav_dir, index_path, dry_run=True,
                              progress_every=0)
        check("resume skips all done", stats2["synthesized"] == 0
              and stats2["already_done"] == 8)

        # limit flag synthesizes only N
        index_path3 = str(Path(td) / "synth_index_limit.jsonl")
        wav_dir3 = Path(td) / "wav_limit"
        stats3 = execute_plan(plan, wav_dir3, index_path3, dry_run=True,
                              limit=3, progress_every=0)
        check("limit caps synthesis at 3", stats3["synthesized"] == 3)

    print()
    if failures:
        print("SMOKE TEST FAILED: %s" % failures)
        return 1
    print("SMOKE TEST PASSED: all checks green.")
    return 0


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def _parse_args(argv):
    p = argparse.ArgumentParser(
        description="S2 synthesize: expand corpus to teacher audio via "
                    "the teacher TTS.")
    p.add_argument("--corpus", default=str(_DEFAULT_CORPUS),
                   help="input corpus.jsonl from S1 (default data/corpus/corpus.jsonl)")
    p.add_argument("--synth-dir", default=str(_DEFAULT_SYNTH_DIR),
                   help="output dir for plan, index, and wav/ (default data/synth)")
    p.add_argument("--config", default=str(_DEFAULT_CONFIG),
                   help="experiment config json (voices, speeds, temp_tiers, ...)")
    p.add_argument("--voices", nargs="+", default=None,
                   help="override config voices (subset of known voices)")
    p.add_argument("--speeds", nargs="+", type=float, default=None,
                   help="override config speeds, e.g. --speeds 0.9 1.0 1.1")
    p.add_argument("--limit", type=int, default=0,
                   help="synthesize at most N clips this run (0 = all)")
    p.add_argument("--max-retries", type=int, default=4,
                   help="synth_request retry attempts on transient errors")

    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true",
                      help="write the plan and SILENT valid WAVs offline; no API")
    mode.add_argument("--execute", action="store_true",
                      help="LIVE synthesis; needs TEACHER_TTS_API_KEY in the env")
    mode.add_argument("--plan-only", action="store_true",
                      help="write the plan and print the spend summary; no synthesis")
    mode.add_argument("--smoke-test", action="store_true",
                      help="self-contained offline smoke test in a temp dir")

    p.add_argument("--no-resume", action="store_true",
                   help="do not skip rows already present in plan/index")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    if args.smoke_test:
        return _smoke_test()

    # default mode is dry-run so an accidental run never spends money
    dry_run = args.dry_run or not (args.execute or args.plan_only)
    plan_only = args.plan_only
    resume = not args.no_resume

    cfg = common.load_config(args.config)

    # Validate any CLI overrides through the SAME checker load_config uses, so
    # --voices notarealvoice or --speeds 9.0 are rejected before any planning or
    # any live API call (the override path must not bypass validation).
    if args.voices is not None or args.speeds is not None:
        eff_voices = args.voices if args.voices is not None else cfg["voices"]
        eff_speeds = args.speeds if args.speeds is not None else cfg["speeds"]
        try:
            common.validate_voices_speeds(eff_voices, eff_speeds)
        except ValueError as e:
            print("error: invalid --voices/--speeds override: %s" % e,
                  file=sys.stderr)
            return 2

    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        print("error: corpus not found at %s (run S1 01_build_corpus first, "
              "or pass --corpus)" % corpus_path, file=sys.stderr)
        return 2
    corpus_rows = common.read_manifest(str(corpus_path))
    if not corpus_rows:
        print("error: corpus at %s is empty" % corpus_path, file=sys.stderr)
        return 2

    synth_dir = Path(args.synth_dir)
    plan_path = str(synth_dir / "synth_plan.jsonl")
    index_path = str(synth_dir / "synth_index.jsonl")
    wav_dir = synth_dir / "wav"

    t0 = time.time()
    plan = build_plan(corpus_rows, cfg, voices=args.voices, speeds=args.speeds)
    if not plan:
        print("error: plan is empty (no synthesizable corpus rows)",
              file=sys.stderr)
        return 2

    added = write_plan(plan_path, plan, resume=resume)
    print("planned %d clips (%d new written to %s)"
          % (len(plan), added, plan_path))

    summary = summarize_plan(plan)
    _print_summary(summary, dry_run=dry_run, executed=not plan_only)
    print("  plan file            : %s" % plan_path)

    if plan_only:
        print("plan-only: stopping before synthesis. Review %s then re-run "
              "with --dry-run or --execute." % plan_path)
        return 0

    if not dry_run:
        print("LIVE synthesis against teacher TTS. The API key is read from "
              "TEACHER_TTS_API_KEY inside the client and is never logged.")

    stats = execute_plan(plan, wav_dir, index_path, dry_run=dry_run,
                         max_retries=args.max_retries, limit=args.limit)
    print("synthesis done in %.1fs: %s" % (time.time() - t0,
          json.dumps({k: v for k, v in stats.items() if k != "error_ids"})))
    print("  index file           : %s" % index_path)
    print("  wav dir              : %s" % wav_dir)
    if stats["errors"]:
        print("  %d clip(s) errored; re-run to retry only the failures "
              "(the index resumes the rest)." % stats["errors"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
