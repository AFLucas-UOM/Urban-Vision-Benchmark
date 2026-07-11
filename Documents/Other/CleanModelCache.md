# Cleaning the Model Cache — `clean_model_cache.ps1`

A PowerShell utility for listing and deleting the large model weights that
PromptDetect downloads, so you can reclaim disk space safely.

- **Script:** [`Scripts/Other-Scripts/PromptDetect/clean_model_cache.ps1`](../../Scripts/Other-Scripts/PromptDetect/clean_model_cache.ps1)
- **Platform:** Windows PowerShell (PowerShell 5.1 or PowerShell 7+)
- **Scope:** touches **only** PromptDetect's models — other cached models are never affected.

---

## What it does

PromptDetect's models (SAM 3, SAM 3.1, Cosmos Reason2 2B/8B/32B, LocateAnything
3B) are **not** stored in the project folder. They download into the shared
**Hugging Face hub cache**, one folder per model:

```
%USERPROFILE%\.cache\huggingface\hub\models--<org>--<name>\
```

(e.g. `…\hub\models--nvidia--Cosmos-Reason2-32B\`). These are large — together
they approach **~100 GB**, and the 32B Cosmos model alone is ~60 GB.

The script:

1. **Locates** the active Hugging Face cache directory (respecting
   `HF_HUB_CACHE` / `HF_HOME` overrides, else the default under your user profile).
2. **Lists** each PromptDetect model with its size on disk and a total.
3. **Optionally deletes** the ones you name (or all of them), after a confirmation
   prompt, and reports how much space was reclaimed.

It only ever considers this fixed set of repositories:

| Short name | Hugging Face repo |
|------------|-------------------|
| `sam3` | `facebook/sam3` |
| `sam3.1` | `facebook/sam3.1` |
| `Cosmos-Reason2-2B` | `nvidia/Cosmos-Reason2-2B` |
| `Cosmos-Reason2-8B` | `nvidia/Cosmos-Reason2-8B` |
| `Cosmos-Reason2-32B` | `nvidia/Cosmos-Reason2-32B` |
| `LocateAnything-3B` | `nvidia/LocateAnything-3B` |

Anything else in your cache (other projects' models, datasets, etc.) is left
untouched.

---

## When to use it

- You're **low on disk space** and want to see which PromptDetect models are
  taking the most room.
- You've **finished evaluating** a particular model (e.g. the 32B Cosmos) and
  want its weights gone.
- You want to **reset** all PromptDetect downloads.

Deleting only removes the cached files; the weights **re-download automatically**
the next time you click **Load Model** for that model. So it is always safe to
delete — it costs only the re-download time/bandwidth later.

---

## Prerequisites

- **Windows** with PowerShell (built-in `powershell.exe`, or `pwsh` for
  PowerShell 7+).
- The script reads file sizes and deletes folders only inside the Hugging Face
  cache — no admin rights, no Python, and no internet needed to run it.
- Nothing else: it does not import any project code or require a conda env.

> If your PowerShell blocks running local scripts (execution policy), see
> [Troubleshooting](#troubleshooting).

---

## How to run it

Open a PowerShell terminal in the script's folder:

```powershell
cd "Scripts\Other-Scripts\PromptDetect"
```

### List models and sizes (default — no deletion)

```powershell
.\clean_model_cache.ps1
```

### Delete specific models

Pass one or more **short names** (comma-separated, no spaces):

```powershell
.\clean_model_cache.ps1 -Delete Cosmos-Reason2-32B,Cosmos-Reason2-8B
```

You'll see what will be removed and how much it frees, then a `y/N` prompt.

### Delete every PromptDetect model

```powershell
.\clean_model_cache.ps1 -All
```

### Delete without the confirmation prompt (e.g. in a script)

```powershell
.\clean_model_cache.ps1 -All -Yes
.\clean_model_cache.ps1 -Delete Cosmos-Reason2-32B -Yes
```

### Run from anywhere (full path)

```powershell
powershell -File "E:\path\to\Urban-Vision-Benchmark\Scripts\Other-Scripts\PromptDetect\clean_model_cache.ps1"
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `-Delete <names>` | string[] | Short names to delete (comma-separated), or `all`. Omit to only list. |
| `-All` | switch | Shorthand for `-Delete all`. |
| `-Yes` | switch | Skip the `y/N` confirmation (non-interactive). |

---

## Expected output

### Listing (no arguments)

```
Hugging Face hub cache: %USERPROFILE%\.cache\huggingface\hub

PromptDetect models in cache:
  sam3                      6.42 GB
  sam3.1                    3.27 GB
  Cosmos-Reason2-2B         4.55 GB
  Cosmos-Reason2-8B        16.34 GB
  Cosmos-Reason2-32B       62.14 GB
  LocateAnything-3B         7.14 GB
  TOTAL                    99.87 GB

Nothing deleted. Use -Delete <name,...> or -All to remove (add -Yes to skip the prompt).
```

Models you haven't downloaded show `(not cached)` and contribute 0 to the total.

### Deleting

```
...
Will delete:
  Cosmos-Reason2-32B  (62.14 GB)
Reclaims ~62.14 GB
Proceed with deletion? (y/N): y
Deleting Cosmos-Reason2-32B ...
Done. Reclaimed ~62.14 GB
```

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `…clean_model_cache.ps1 cannot be loaded because running scripts is disabled` | Your execution policy blocks local scripts. Run once: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` (affects only the current window), or invoke via `powershell -ExecutionPolicy Bypass -File .\clean_model_cache.ps1`. |
| `Cache directory not found — nothing to do.` | No Hugging Face cache exists yet (nothing downloaded), or `HF_HOME` / `HF_HUB_CACHE` points elsewhere. Check `echo $env:HF_HOME` and `echo $env:HF_HUB_CACHE`. |
| All models show `(not cached)` but you expected some | The app is using a different cache location (an `HF_HOME` set only inside the conda env). Set the same variable in your PowerShell session, or pass the real path. |
| A model shows a **larger** size than expected | Partial/`.incomplete` downloads from an interrupted load are counted (they occupy real disk). Deleting clears them too. |
| Deletion seems to leave space used | Recycle Bin is bypassed (`Remove-Item -Force`), but check no other process (a running app / worker) is holding the files; close the app first. |
| Wrong model name | Use the **short names** from the table above (e.g. `Cosmos-Reason2-8B`, not the full `nvidia/...`). Unknown names are reported and skipped. |

---

## Related

- [PromptDetect.md](../PromptDetect.md) — the application this cache belongs to.
- [ModelArchitectures.md](../ModelArchitectures.md) — per-model details and footprints.
