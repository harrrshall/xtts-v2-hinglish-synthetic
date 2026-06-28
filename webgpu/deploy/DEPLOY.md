# Deploying syntts (the in-browser Hinglish TTS)

The one fact that decides everything: the model is **~740 MB across 3 files** (gpt_prefill 342 MB,
gpt_decode 324 MB, vocoder 74 MB). Plan around that.

## What NOT to do (and why)

- **GitHub Pages for the models.** GitHub blocks any file over **100 MB**. The two big ONNX files
  exceed that. Git LFS works around the size but GitHub Pages does not serve LFS content well and LFS
  bandwidth is metered. So GitHub Pages cannot host the models.
- **Vercel/Netlify/Cloudflare Pages for the models.** Static hosts cap per-file size (Cloudflare Pages
  is 25 MB) and meter bandwidth. Each visitor pulls 740 MB, so ~135 visitors burns a 100 GB/month free
  tier. Wrong tool for big model weights.

## Recommended: HF Hub for the models + GitHub Pages for the app

The Hugging Face Hub is built for GB-scale model files, serves them over a CDN with the right CORS and
range-request headers (this is exactly how Transformers.js loads models), and the bandwidth is free and
generous. Put the **740 MB of weights on the Hub**, and the **tiny static app (~3.4 MB)** anywhere.

### Step 1 — put the models on the HF Hub

```bash
# one-time: log in (run it yourself so the token stays private)
! pip install -U "huggingface_hub[cli]"
! huggingface-cli login

# create a repo and upload the 3 onnx files under models/
huggingface-cli repo create syntts-webgpu --type model -y
huggingface-cli upload harrrshall/syntts-webgpu webgpu/app/models models --repo-type=model
```

Files are then served (CORS-enabled, CDN-cached) at:
`https://huggingface.co/harrrshall/syntts-webgpu/resolve/main/models/gpt_prefill.onnx` (etc).

### Step 2 — point the app at that base

In `webgpu/app/config.json` set:

```json
"modelBase": "https://huggingface.co/harrrshall/syntts-webgpu/resolve/main/"
```

The app loads the models from there, streams them with the progress UI, and caches them in the browser
Cache API so revisits are instant. Locally, leave `modelBase` as `""` (relative paths).

### Step 3 — deploy the static app to GitHub Pages

The app folder minus `models/` is only the HTML, JS, tokenizer, normalizer, and voices (~3.4 MB), all
well under GitHub limits. The included workflow `.github/workflows/deploy-pages.yml` publishes
`webgpu/app/` to Pages on every push to main. Then in the repo: **Settings → Pages → Source: GitHub
Actions**. Your site goes live at `https://<user>.github.io/<repo>/`.

(Prefer one place? See the HF Static Space alternative below.)

## Alternative: one place, Hugging Face Static Space

A HF **Static Space** can host the app AND the models together (large files go through Git LFS, served
from the same origin, so you do not even need `modelBase` or CORS). Simplest single deploy:

```bash
huggingface-cli repo create syntts --type space --space_sdk static -y
cd webgpu/app && git init && git lfs install && git lfs track "*.onnx"
huggingface-cli upload harrrshall/syntts . . --repo-type=space
```

Trade-off: the big files live in the Space repo via LFS. The HF-Hub-plus-Pages split keeps the app repo
tiny and the weights on the Hub where they belong. Either works.

## Analytics (see how many use it vs abandon the download)

The app already fires a funnel of custom events (no cookies, no PII):

| event | when | useful props |
|---|---|---|
| `app_open` | page load | device (webgpu/wasm) |
| `download_start` | first byte of a non-cached model | |
| `download_complete` | all models ready | seconds, cached |
| `download_abandon` | visitor leaves mid-download | device |
| `app_ready` | usable | device |
| `generate_start` / `generate_done` / `generate_error` | per synthesis | voice, chars, codes, seconds |

**"Wasted" loads = `download_start` fired but no `download_complete`** (they left during the 740 MB pull).
That is the number you care about, and it is why a smaller model (fp16, ~370 MB) would lift completion.

Wire a provider in `index.html` (a commented snippet is there) and you are done:
- **Umami** (recommended, free cloud at umami.is, custom events, GDPR-friendly, no cookie banner):
  `<script defer src="https://cloud.umami.is/script.js" data-website-id="YOUR-ID"></script>`
- **Plausible** also works (use the `script.tagged-events.js` build).
- No third party? Set `analytics.beacon` in `config.json` to your own collector URL; events POST there
  via `navigator.sendBeacon`.

With `analytics.debug: true` (current default) every event logs to the browser console so you can verify
the funnel before going live.

## The user experience to expect

- **First visit:** a one-time ~740 MB download with a live per-file progress panel (percent, MB of MB,
  per-model bars), then it is cached forever in the browser.
- **Every visit after:** loads from cache in a second or two.
- **Speed:** on a WebGPU machine (Chrome/Edge 121+, Safari 26+) it uses the GPU and is near or above
  real-time. On a device without WebGPU it falls back to WASM (correct, just slower).
- **Biggest UX lever left:** an **fp16 build halves the download to ~370 MB** and speeds WebGPU. It is
  the documented follow-up in `docs/WEBGPU_PLAN.md`; do it before a public launch if first-load size
  matters.
