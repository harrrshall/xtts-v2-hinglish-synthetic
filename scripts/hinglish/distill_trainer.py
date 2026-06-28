#!/usr/bin/env python3
"""Knowledge-distillation trainer: shrink the fine-tuned XTTS-v2 Hinglish GPT 30 -> 12 layers (~443M
-> ~200M) at parity, by distilling from the frozen 443M GPT into a 12-layer student.

The student is the SAME architecture with fewer layers (d=1024, heads=16, audio vocab 1026, DVAE,
HiFi-GAN, perceiver-32 ALL unchanged), so every frozen interface stays byte-identical and no output
adapter is needed. The ONLY architecture delta is gpt_layers: 30 -> 12.

Per step we run BOTH the 12-layer student (with grad) and the frozen 30-layer teacher (no grad) on the
same batch, and add three signals on top of coqui's existing text-CE + mel-CE:
  (a) loss_kd_logit  : temperature-softened KL between student/teacher next-audio-token distributions
                       over the 1026 mel classes (the mel_head logits, already returned by GPT.forward).
  (b) loss_kd_hidden : 1 - cosine between student layer hidden states and the teacher layer each student
                       layer was initialized from (identity projection, d=1024 both).
  (c) loss_kd_attn   : optional attention-map KL (off by default; costs extra memory).
The CE-on-teacher-audio-tokens term is the EXISTING coqui mel-CE; the data audio is the Smallest.ai
teacher corpus already DVAE-encoded by the trainer, so that term is free. Smallest.ai itself is a closed
API and CANNOT be a logit/hidden teacher; the 443M XTTS GPT is the white-box teacher for (a)/(b), while
the Smallest.ai audio remains the ground-truth CE anchor (see docs/COMPRESSION_PLAN.md).

Verified against coqui-tts 0.27.5 source (GPTTrainer.forward returns (loss_text, loss_mel, mel_logits);
the inner self.xtts.gpt.gpt is a HF GPT2Model that accepts output_hidden_states). NOT run here; validate
with `train_xtts.py --distill --smoke` on the GPU box (it asserts shapes, finiteness, and the hook fired).
"""
import copy
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer
from TTS.tts.models.xtts import Xtts

# Teacher block indices (out of 0..29) copied into the 12 student blocks. Endpoints 0 and 29 kept so the
# student inherits the teacher's input-conditioning and output-shaping layers, even strided between.
KEEP_DEFAULT = (0, 2, 5, 8, 11, 14, 17, 20, 23, 26, 28, 29)
# hidden_states indices to match: each student block j -> teacher block KEEP[j], whose output is
# hidden_states[KEEP[j] + 1]. (hidden_states has length n_layer + 1; index 0 is the post-embedding input.)
LAYER_MAP_DEFAULT = "1,3,6,9,12,15,18,21,24,27,29,30"


def strided_keep(student_layers, teacher_layers=30):
    """Pick `student_layers` teacher block indices from 0..teacher_layers-1, endpoints included,
    evenly strided. Used both to seed the student blocks and to map hidden states for KD, so any
    student depth (12, 16, ...) works without a hand-written layer map."""
    if student_layers >= teacher_layers:
        return tuple(range(teacher_layers))
    raw = [round(i * (teacher_layers - 1) / (student_layers - 1)) for i in range(student_layers)]
    seen, out = set(), []
    for x in raw:
        if x not in seen:
            seen.add(x); out.append(x)
    i = 0
    while len(out) < student_layers and i < teacher_layers:
        if i not in seen:
            seen.add(i); out.append(i)
        i += 1
    return tuple(sorted(out))


# --------------------------------------------------------------------------------------------------
# checkpoint helpers (work for both Xtts-format "gpt.*" keys and GPTTrainer-format "xtts.gpt.*" keys)
# --------------------------------------------------------------------------------------------------
def _load_gpt_state(ckpt_path: str) -> dict:
    """Return the bare GPT-module state dict (inner transformer blocks at 'gpt.h.{i}.*')."""
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = obj.get("model", obj) if isinstance(obj, dict) else obj
    out = {}
    for k, v in sd.items():
        kk = k[len("xtts."):] if k.startswith("xtts.") else k
        if kk.startswith("gpt."):
            out[kk[len("gpt."):]] = v
    return out


def _strided_student_state(ckpt_path: str, keep=KEEP_DEFAULT) -> dict:
    """Remap a 30-layer teacher GPT state dict to a 12-layer student: copy the KEEP blocks (renumbered
    0..11) and everything that is not layer-indexed (embeddings, heads, conditioning, perceiver, ln_f)
    verbatim. Load with strict=False."""
    g = _load_gpt_state(ckpt_path)
    idx_of = {ti: si for si, ti in enumerate(keep)}
    out = {}
    for k, v in g.items():
        if k.startswith("gpt.h."):                 # inner GPT2Model transformer blocks
            parts = k.split(".")
            ti = int(parts[2])
            if ti in idx_of:                        # else: this teacher layer is dropped
                parts[2] = str(idx_of[ti])
                out[".".join(parts)] = v
        else:
            out[k] = v
    return out


def build_student_init(teacher_ckpt: str, out_path: str, keep=KEEP_DEFAULT) -> str:
    """Standalone: write the strided 12-layer warm-start state dict to disk (optional; the trainer does
    this in memory at __init__)."""
    sd = _strided_student_state(teacher_ckpt, keep)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(sd, out_path)
    print(f"[distill] wrote strided student init ({len(sd)} tensors) -> {out_path}")
    return out_path


# --------------------------------------------------------------------------------------------------
@dataclass
class DistillGPTArgs(GPTArgs):
    teacher_ckpt: str = ""              # the fine-tuned 443M model.pth (white-box KD teacher)
    kd_temperature: float = 2.0
    kd_logit_weight: float = 1.0       # (a)
    kd_hidden_weight: float = 1.0      # (b)
    kd_attn_weight: float = 0.0        # (c), 0 disables (saves attention memory)
    kd_layer_map: str = ""             # empty = auto-derive from gpt_layers via strided_keep
    kd_cond_offset: int = 32           # perceiver latent count prepended to the sequence (v2 = 32)
    smoke: bool = False


class DistillGPTTrainer(GPTTrainer):
    """GPTTrainer that distills a frozen 30-layer teacher into the (12-layer) student it already builds."""

    def __init__(self, config):
        super().__init__(config)        # builds the 12-layer student self.xtts, DVAE, mel extractors
        a = self.args
        assert a.teacher_ckpt, "DistillGPTTrainer requires teacher_ckpt (the fine-tuned 443M model.pth)"
        # derive the teacher->student layer selection from the student depth, unless an explicit map
        # is given. keep[j] = teacher block seeding student block j; layer_map[j] = keep[j]+1 (its
        # hidden_states index). This makes any student depth (12, 16, ...) work with no hand-edited map.
        if a.kd_layer_map.strip():
            self._layer_map = [int(x) for x in a.kd_layer_map.split(",")]
            keep = tuple(i - 1 for i in self._layer_map)
        else:
            keep = strided_keep(a.gpt_layers)
            self._layer_map = [k + 1 for k in keep]
        assert len(self._layer_map) == a.gpt_layers, (
            f"layer_map has {len(self._layer_map)} entries but gpt_layers={a.gpt_layers}")
        assert min(self._layer_map) >= 1, "layer_map indexes hidden_states (1..n_layer)"
        print(f"[distill] student_layers={a.gpt_layers} keep={list(keep)} layer_map={self._layer_map}")
        self._hs_store: dict = {}

        # warm-start the student GPT from the strided fine-tuned teacher layers (overwrites whatever
        # super().__init__ loaded from xtts_checkpoint for the GPT)
        student_sd = _strided_student_state(a.teacher_ckpt, keep)
        miss, unexp = self.xtts.gpt.load_state_dict(student_sd, strict=False)
        print(f"[distill] student warm-start: loaded={len(student_sd)} missing={len(miss)} "
              f"unexpected={len(unexp)}")

        # frozen 30-layer teacher, hidden in a plain dict so nn.Module does NOT register it: it is then
        # excluded from get_optimizer (only self.xtts.gpt is optimized), from .to() (moved lazily), and
        # from saved checkpoints (no 443M bloat per save).
        teacher_gpt = self._build_teacher(config)
        self._teacher = {"m": teacher_gpt, "ready": False}
        self._register_kd_hooks(teacher_gpt)

    # ---- teacher construction -------------------------------------------------------------------
    def _build_teacher(self, config):
        ta = copy.deepcopy(config.model_args)
        ta.gpt_layers = 30
        tcfg = copy.deepcopy(config)
        tcfg.model_args = ta
        txtts = Xtts(tcfg)
        txtts.tokenizer = self.xtts.tokenizer       # identical vocab
        txtts.init_models()                          # builds the 30-layer GPT (+ hifigan we will drop)
        sd = _load_gpt_state(self.args.teacher_ckpt)
        miss, unexp = txtts.gpt.load_state_dict(sd, strict=False)
        critical = [m for m in miss if not (m.startswith("gpt_inference") or "wte" in m or "wpe" in m)]
        assert not critical, f"teacher GPT missing critical keys: {critical[:8]}"
        tg = txtts.gpt
        tg.eval()
        tg.requires_grad_(False)
        del txtts                                    # free hifigan etc.; tg keeps the GPT alive
        n = sum(p.numel() for p in tg.parameters())
        print(f"[distill] teacher GPT loaded: {n/1e6:.1f}M params (frozen)  "
              f"missing={len(miss)} unexpected={len(unexp)}")
        return tg

    def _register_kd_hooks(self, teacher_gpt):
        """Inject output_hidden_states into the inner GPT2Model call (get_logits does not pass it) and
        capture the hidden-states tuple, with no edit to coqui source."""
        def pre(module, args, kwargs):
            kwargs["output_hidden_states"] = True
            kwargs["return_dict"] = True
            if self.args.kd_attn_weight > 0:
                kwargs["output_attentions"] = True
            return args, kwargs

        def capture(tag):
            def f(module, inp, out):
                self._hs_store[tag] = out.hidden_states
                if self.args.kd_attn_weight > 0:
                    self._hs_store[tag + "_attn"] = getattr(out, "attentions", None)
            return f

        self.xtts.gpt.gpt.register_forward_pre_hook(pre, with_kwargs=True)
        self.xtts.gpt.gpt.register_forward_hook(capture("student"))
        teacher_gpt.gpt.register_forward_pre_hook(pre, with_kwargs=True)
        teacher_gpt.gpt.register_forward_hook(capture("teacher"))

    def _teacher_on(self, device):
        """Lazily move the (non-registered) teacher onto the student's device + dtype, kept eval/frozen."""
        tg = self._teacher["m"]
        dtype = next(self.xtts.gpt.parameters()).dtype
        if (not self._teacher["ready"]) or next(tg.parameters()).device != device:
            tg.to(device=device, dtype=dtype)
            tg.eval()
            tg.requires_grad_(False)
            self._teacher["ready"] = True
        return tg

    # ---- masking --------------------------------------------------------------------------------
    def _mel_valid_mask(self, wav_lengths, S, device):
        """Boolean (B, S) over valid mel-code positions, mirroring how GPT.forward derives mel lengths
        (wav_lengths // gpt_code_stride_len). The +1 approximates the stop token; an off-by-one here is
        sub-perceptual. Smoke prints mask_frac so you can sanity-check it on the box."""
        stride = getattr(self.args, "gpt_code_stride_len", 1024)
        mel_lens = (wav_lengths.to(device).long() // stride) + 1
        mel_lens = mel_lens.clamp(1, S)
        ar = torch.arange(S, device=device).unsqueeze(0)        # (1, S)
        return ar < mel_lens.unsqueeze(1)                        # (B, S)

    # ---- the KD step ----------------------------------------------------------------------------
    def train_step(self, batch, criterion):
        a = self.args
        loss_dict = {}
        self._hs_store.clear()

        cond_mels = batch["cond_mels"]
        text_inputs = batch["text_inputs"]
        text_lengths = batch["text_lengths"]
        audio_codes = batch["audio_codes"]
        wav_lengths = batch["wav_lengths"]
        cond_idxs = batch["cond_idxs"]
        cond_lens = batch["cond_lens"]

        # ---- STUDENT forward (grad); hook fills _hs_store["student"] ----
        loss_text, loss_mel, s_logits = self.forward(
            text_inputs, text_lengths, audio_codes, wav_lengths, cond_mels, cond_idxs, cond_lens)
        student_hs = self._hs_store.get("student")

        # ---- TEACHER forward (frozen, no grad); hook fills _hs_store["teacher"] ----
        tg = self._teacher_on(text_inputs.device)
        with torch.no_grad():
            _, _, t_logits = tg(
                text_inputs, text_lengths, audio_codes, wav_lengths,
                cond_mels=cond_mels, cond_idxs=cond_idxs, cond_lens=cond_lens)
        teacher_hs = self._hs_store.get("teacher")

        # ---- existing CE terms (unchanged weights: text 0.01, mel 1.0) ----
        loss_dict["loss_text_ce"] = loss_text * a.gpt_loss_text_ce_weight
        loss_dict["loss_mel_ce"] = loss_mel * a.gpt_loss_mel_ce_weight

        assert s_logits.shape == t_logits.shape, ("logit shape mismatch", s_logits.shape, t_logits.shape)
        assert student_hs is not None and teacher_hs is not None, "hidden-state hook did not fire"

        # ---- (a) temperature-softened logit-KL over the 1026 audio classes, masked to valid mel ----
        T = a.kd_temperature
        S = s_logits.shape[-1]
        mask = self._mel_valid_mask(wav_lengths, S, s_logits.device)        # (B, S)
        s = s_logits.permute(0, 2, 1).float()                              # (B, S, C)
        t = t_logits.permute(0, 2, 1).float().detach()
        logp_s = F.log_softmax(s / T, dim=-1)
        p_t = F.softmax(t / T, dim=-1)
        kl = F.kl_div(logp_s, p_t, reduction="none").sum(-1)               # (B, S) KL(teacher||student)
        kd_logit = (kl * mask).sum() / mask.sum().clamp_min(1) * (T * T)
        loss_dict["loss_kd_logit"] = a.kd_logit_weight * kd_logit

        # ---- (b) hidden-state cosine matching (identity proj, d=1024), conditioning prefix sliced off ----
        off = a.kd_cond_offset
        kd_hidden = s_logits.new_zeros(())
        for j, ti in enumerate(self._layer_map):
            hs = student_hs[j + 1][:, off:, :].float()                     # student block j+1 output
            ht = teacher_hs[ti][:, off:, :].float().detach()               # the teacher block it copied
            kd_hidden = kd_hidden + (1.0 - F.cosine_similarity(hs, ht, dim=-1)).mean()
        kd_hidden = kd_hidden / len(self._layer_map)
        loss_dict["loss_kd_hidden"] = a.kd_hidden_weight * kd_hidden

        # ---- (c) optional attention KL (attentions tuple has length n_layer; no embedding entry) ----
        if a.kd_attn_weight > 0:
            sa = self._hs_store.get("student_attn")
            ta_ = self._hs_store.get("teacher_attn")
            # the HF GPT2 SDPA backend returns no attention weights (None); skip rather than crash
            if sa is None or ta_ is None or sa[0] is None or ta_[0] is None:
                if a.smoke:
                    print("[distill] attention KD unavailable (SDPA backend returns no weights); skipping")
            else:
                kd_attn = s_logits.new_zeros(())
                for j, ti in enumerate(self._layer_map):
                    a_s = sa[j].clamp_min(1e-9)
                    a_t = ta_[ti - 1].detach()
                    kd_attn = kd_attn + F.kl_div(a_s.log(), a_t, reduction="batchmean")
                loss_dict["loss_kd_attn"] = a.kd_attn_weight * (kd_attn / len(self._layer_map))

        total = (loss_dict["loss_text_ce"] + loss_dict["loss_mel_ce"]
                 + loss_dict["loss_kd_logit"] + loss_dict["loss_kd_hidden"])
        if "loss_kd_attn" in loss_dict:
            total = total + loss_dict["loss_kd_attn"]
        loss_dict["loss"] = total

        assert torch.isfinite(total), ("non-finite loss", {k: float(v) for k, v in loss_dict.items()})
        if a.smoke:
            print(f"[distill/smoke] mel_ce={float(loss_dict['loss_mel_ce']):.4f} "
                  f"kd_logit={float(loss_dict['loss_kd_logit']):.4f} "
                  f"kd_hidden={float(loss_dict['loss_kd_hidden']):.4f} "
                  f"S={S} mask_frac={float(mask.float().mean()):.2f} "
                  f"hs(student={len(student_hs)} teacher={len(teacher_hs)})")
        return {"model_outputs": None}, loss_dict

    def on_train_epoch_start(self, trainer):
        super().on_train_epoch_start(trainer)       # flips student GPT to train(), rest to eval()
        tg = self._teacher["m"]                      # keep the teacher frozen regardless
        tg.eval()
        tg.requires_grad_(False)

    @staticmethod
    def init_from_config(config, samples=None):
        return DistillGPTTrainer(config)
