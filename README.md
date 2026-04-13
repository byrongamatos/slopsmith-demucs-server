# Slopsmith Demucs Server

A lightweight GPU-accelerated service that provides AI source separation and lyrics alignment for [Slopsmith](https://github.com/byrongamatos/slopsmith). Designed to run on a desktop with a CUDA GPU while Slopsmith runs on a NAS or Docker host.

## Features

### Source Separation (`POST /separate`)

Splits audio into individual stems using [Demucs](https://github.com/facebookresearch/demucs):

- **6-stem model** (htdemucs_6s): drums, bass, vocals, other, guitar, piano
- **4-stem model** (htdemucs): drums, bass, vocals, other
- File upload or URL input
- Per-stem caching (avoids re-processing)
- WebSocket progress updates

### Lyrics Alignment (`POST /align`)

Forced alignment of plain text lyrics against an audio file using [Whisper](https://github.com/openai/whisper) via [stable-ts](https://github.com/jianfch/stable-ts):

- **Line, word, or syllable granularity**
- Syllable splitting via pyphen hyphenation with CJK character support
- Automatic language detection (or manual language hint)
- Lazy-loaded Whisper medium model with CUDA acceleration
- Used by the [Lyrics Sync plugin](https://github.com/byrongamatos/slopsmith-plugin-lyrics-sync)

## Setup

### Requirements

- Python 3.10+
- CUDA-capable GPU (recommended) or CPU fallback
- FFmpeg

### Install

```bash
git clone https://github.com/byrongamatos/slopsmith-demucs-server.git
cd slopsmith-demucs-server
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run

```bash
python server.py --port 7865
```

Options:
- `--port` — port to listen on (default: 7865)
- `--host` — host to bind to (default: 0.0.0.0)
- `--model` — Demucs model (default: htdemucs_6s)
- `--device` — force cpu or cuda (auto-detected by default)
- `--api-key` — optional API key for authentication

### Run as a systemd service

```bash
cp slopsmith-demucs.service ~/.config/systemd/user/
systemctl --user enable slopsmith-demucs
systemctl --user start slopsmith-demucs
```

Edit the service file to adjust the path to your clone and desired model.

### Configure in Slopsmith

In Slopsmith settings, set the Demucs Server URL to `http://<your-desktop-ip>:7865`.

## API

### `GET /health`

Returns server status, model, GPU availability, and cache directory.

### `POST /separate`

Separate audio into stems.

| Parameter | Type | Description |
|-----------|------|-------------|
| `file` | Upload | Audio file |
| `stems` | Query | Comma-separated stem names (default: `drums,bass,vocals,other`) |
| `model` | Query | Override model (optional) |

### `POST /align`

Forced-align lyrics against audio using Whisper.

| Parameter | Type | Description |
|-----------|------|-------------|
| `file` | Form (file) | Audio file (vocals stem) |
| `text` | Form | Plain text lyrics |
| `language` | Form | ISO language code hint (optional, auto-detected) |
| `granularity` | Form | `line` (default), `word`, or `syllable` |

Returns: `{"segments": [{"start": 12.34, "end": 15.67, "text": "lyrics line"}, ...]}`

### `GET /download/{job_id}/{stem}`

Download a separated stem by job ID.

### `GET /jobs` / `GET /jobs/{job_id}`

List or inspect separation jobs.

### `WS /ws/jobs/{job_id}`

WebSocket for real-time separation progress updates.
