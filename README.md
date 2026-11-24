# SpeechSAE

Understand speech-model representations with sparse autoencoders. This project includes a practical HuBERT SAE notebook and **Speech Scope**, an interactive audio-analysis website.

- **[Live demo on Hugging Face](https://huggingface.co/spaces/Kelelu/SpeechSAE)**
- **[Notebook](undestandingSAE.ipynb)**
- **[Demo source](Demo_Code)**

The public Space runs inference on free CPU resources, so startup and uploads can take time. A localhost link works only on the computer running the application; use the Hugging Face link when sharing the demo.

## Explore the demo

Upload or record a short audio clip to inspect real HuBERT activations, sparse features, reconstruction measurements, spectrograms, per-feature timelines, and a feature heatmap. Select F1465 or another feature and listen around its peak. Download results as JSON.

The **Compare words** section includes saved, approximately aligned examples of **came** and **gave**, with playable excerpts, shared-scale curves, within-word activation measurements, and CSV export. Choose metadata-only samples or strongest recording peaks. Public examples are a small snapshot of the local analysis, not a live scan of the entire research dataset. New uploaded recordings are analyzed live.

F1465 is a **came-associated candidate**, not a verified exclusive word detector. In the exploratory dataset analysis it was active somewhere in 448/467 came recordings and 56/13,593 recordings without came. Approximate word timing, repeated sentences, speakers, and synthesis systems limit the conclusions. Other features are unmapped.

## Run locally

Python 3.11 or 3.12 and FFmpeg are recommended. Install PyTorch appropriate for your hardware, then:

```bash
git clone https://github.com/kehinde-elelu/SpeechSAE.git
cd SpeechSAE
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python Demo_Code/server.py --port 7865
```

Open **http://localhost:7865**. Model weights download from Hugging Face at first startup. The backend shows its loading status. It listens on loopback by default. Use `python app.py` for the Gradio-compatible deployment entry point on port 7860.

Uploads are limited to 30 seconds and 20 MB. Uploaded audio is not added to a dataset; temporary decoding files are removed. The demo does not predict MOS or assign verified semantic names. The public Gradio interface is also available at `/gradio`.

## Run the notebook with your dataset

The notebook is the original educational workflow, with execution outputs cleared to avoid embedding recordings and machine-specific output. It requires local dataset files and optional local recordings; the full VoiceMOS corpus is not included.

Set `SPEECHSAE_BASE` to the repository root and `SPEECHSAE_DATA` to a folder containing:

```text
metadata.csv   # wavID | intended emotion | transcript; pipe-separated, no header
wav/           # recordings
sets/train.csv # pipe-separated listener ratings with column headers
```

Start Jupyter from the repository root and select your Python environment. The notebook selects 40 training examples, extracts HuBERT hidden states, plots sparse activations, checks reconstruction, and investigates F1465. The additional own-recording cells need your local `audio_recordbyme/me_1_analysis/` files.

## Models and extraction convention

- Backbone: `facebook/hubert-base-ls960`.
- SAE: `Egorgij21/Audio-SAE-hubert-base`, `layer_11/ae.pt`.
- HuBERT hidden width: 768; dictionary size: 6,144; training target k: 50.
- Thresholded inference uses the checkpoint's learned threshold, approximately 0.30227. The active count varies by frame.
- The upstream wrapper setting `sae_after_layer=11` selects Hugging Face hidden-state tuple **index 10** in this implementation. Keep the checkpoint and extraction point matched.

The decoder reconstructs hidden activations, not audio. This HuBERT SAE is not the SAE trained on TransportMOS's actual encoder.

## Hosting

`app.py` provides a Gradio-compatible Space entry point and proxies the custom website to its local analysis backend. `requirements.txt` and `packages.txt` supply Python and audio-decoding dependencies. The existing public Space is [Kelelu/SpeechSAE](https://huggingface.co/spaces/Kelelu/SpeechSAE).

GitHub Pages alone cannot execute the Python speech models. Free Space resources may sleep and must warm up after restart. Model caches, checkpoints, full corpus data, private recordings, logs, and server PID files are excluded from this repository.

## Attribution

The SAE implementation and interpretation helpers in `vendor/audio_sae` are adapted from [AudioSAE](https://github.com/audiosae/audio-sae) under its MIT license; see `vendor/LICENSE`. The educational notebook adapts the upstream interpretation workflow. Website layout inspiration comes from Ayush Gharat's Latent Scope; the speech implementation and evidence are specific to this project. Backbone and SAE weights retain their respective model licenses.
