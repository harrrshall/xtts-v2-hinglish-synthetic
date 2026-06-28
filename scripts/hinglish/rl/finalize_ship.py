#!/usr/bin/env python3
"""Generate per-voice demo samples, then create the HF repo and upload the ship dir."""
import sys
from pathlib import Path
import numpy as np, soundfile as sf
sys.path.insert(0, "scripts/hinglish/fp16")
import xtts_patch  # noqa
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from huggingface_hub import HfApi, create_repo

SHIP = Path("runs/rl/ship_265m")
REPO = "harrrshall/xtts-hinglish-265m"

cfg = XttsConfig(); cfg.load_json(str(SHIP / "config.json")); cfg.model_args.gpt_layers = 16
cfg.model_args.dvae_checkpoint = str(SHIP / "dvae.pth"); cfg.model_args.mel_norm_file = str(SHIP / "mel_stats.pth")
m = Xtts.init_from_config(cfg)
m.load_checkpoint(cfg, checkpoint_path=str(SHIP / "model.pth"), vocab_path=str(SHIP / "vocab.json"), use_deepspeed=False)
m.cuda().eval()
(SHIP / "samples").mkdir(exist_ok=True)
texts = {
    "arjun": "यार weekend पर एक road trip plan करते हैं मौसम भी काफी pleasant है",
    "maya": "मेरा manager चाहता है कि हम इस sprint में सारे pending tickets close कर दें",
    "aadya": "कल रात मैंने एक ranked match खेला but teammate ने पूरी game throw कर दी",
}
for v, t in texts.items():
    gc, sp = m.get_conditioning_latents(audio_path=[str(SHIP / "refs" / f"{v}.wav")])
    o = m.inference(t, "hi", gc, sp, temperature=0.7, enable_text_splitting=False)
    sf.write(str(SHIP / "samples" / f"demo_{v}.wav"), np.asarray(o["wav"], np.float32), 24000)
    print("[ship] sample", v)

print("[ship] creating repo + uploading", REPO)
create_repo(REPO, repo_type="model", private=False, exist_ok=True)
HfApi().upload_folder(folder_path=str(SHIP), repo_id=REPO, repo_type="model",
                      commit_message="XTTS-v2 Hinglish 265M: distilled 443M->265M + RFT/DPO accent recovery, certified no quality loss (UTMOS/SECS)")
print("UPLOADED https://huggingface.co/" + REPO)
