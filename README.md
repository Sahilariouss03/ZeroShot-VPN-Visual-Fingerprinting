# FlowPic VPN Fingerprinting

This project turns multiplexed VPN traffic into FlowPic images, learns 128-dimensional fingerprints with triplet-loss metric learning, and identifies applications through template matching without retraining the CNN every time a new application is added.

## What is implemented

- `src/flowpic/generate.py`: creates model-ready FlowPic images from PCAP windows
- `src/flowpic/capture.py`: captures live traffic and generates FlowPics
- `src/flowpic/data_builder.py`: scans `data/raw/<app_name>/` and builds a processed dataset plus manifest
- `src/flowpic/train_pipeline.py`: trains the 128-d embedding CNN with triplet margin loss
- `src/flowpic/library.py`: creates embeddings and registers application templates in `templates/library.json`
- `src/flowpic/predict.py`: predicts a label for an unknown FlowPic or PCAP
- `src/flowpic/gui.py`: desktop GUI for the full workflow
- `src/flowpic/demo.py`: synthetic demo pipeline for showing known, unknown, and zero-shot template registration behavior

## Install

Use Python 3.11+.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .[dev]
```

## Launch the GUI

Start the desktop app with:

```powershell
.\launch_gui.bat
```

The GUI has five tabs:

- `FlowPic`: capture live traffic or convert a PCAP into a FlowPic image
- `Dataset`: build `data/processed/manifest.csv` from labeled raw PCAP folders
- `Training`: train the CNN and save `checkpoints/backbone_v1.pth`
- `Library + Predict`: register known apps into `templates/library.json` and predict unknown samples
- `Demo`: create a ready-made synthetic demo that shows known match, unknown rejection, and zero-shot registration

## GUI walkthrough

### 1. `FlowPic` tab

Use this when you want to show how traffic becomes an image.

- `Output PNG`: where the generated FlowPic image will be saved
- `PCAP Output`: where a live capture should be saved
- `Duration`: how long live capture should run
- `Interface`: the network adapter name for live capture
- `Filter`: packet filter such as `tcp or udp`
- `PCAP Input`: existing PCAP file for offline generation

Buttons:

- `List Interfaces`: prints Scapy-detected interfaces into the log area
- `Capture Live`: captures traffic and generates a FlowPic
- `Generate From PCAP`: converts an existing PCAP into a FlowPic

The latest generated image appears in the preview panel under the tabs.

### 2. `Dataset` tab

Use this after collecting PCAPs for known applications.

Expected raw layout:

```text
data/raw/
  YouTube/
    capture1.pcap
    capture2.pcap
  Zoom/
    session1.pcap
```

Fields:

- `Raw Root`: folder containing one subfolder per application label
- `Processed Root`: folder where FlowPics should be written
- `Manifest`: CSV file that indexes the generated FlowPics

Click `Build Dataset` to slice each PCAP into 5-second windows, generate FlowPics, and create the manifest.

### 3. `Training` tab

Use this to train the embedding backbone.

Fields:

- `Manifest`: processed dataset CSV, usually `data/processed/manifest.csv`
- `Checkpoint`: destination for the best saved model
- `Epochs`: number of training epochs
- `Batch Size`: number of samples per training batch

Click `Train Model` to train the CNN with triplet margin loss. The best checkpoint is saved to the path you provide.

### 4. `Library + Predict` tab

Use this to show recognition behavior.

Registration fields:

- `Checkpoint`: trained model checkpoint
- `Library JSON`: template storage file
- `Register Label`: app name to register, such as `YouTube`
- `Register Manifest`: manifest containing FlowPics for that label

Prediction fields:

- `Predict Image`: unknown FlowPic image to classify
- `Predict PCAP`: unknown PCAP to classify
- `Threshold`: distance threshold for deciding `Unknown`

Buttons:

- `Register App`: creates the average embedding centroid for that app and stores it in the library
- `Predict Unknown`: compares the input against the stored template library

Only one of `Predict Image` or `Predict PCAP` should be filled in when running prediction.

### 5. `Demo` tab

Use this when you want a smooth presentation without collecting real traffic first.

This tab creates synthetic FlowPics that behave like three app families:

- two known apps used for training and template registration
- one unseen app that should first be labeled `Unknown`
- the same unseen app after registration, which should then be recognized without retraining

Fields:

- `Demo Root`: folder where demo images, manifests, checkpoint, library, and report will be stored
- `Demo Epochs`: training epochs for the demo checkpoint
- `Demo Threshold`: matching threshold for the demo predictions

Buttons:

- `Create Demo Assets`: generates synthetic FlowPic images and manifests only
- `Run Full Demo`: creates the assets, trains the model, registers known apps, predicts a known app, predicts a zero-day app as `Unknown`, registers that zero-day app, and predicts it again

After `Run Full Demo`, the results are written to `demo_assets/demo_report.json` or the folder you selected.

Important note:

- this built-in demo is synthetic and is meant for explaining the workflow and the zero-shot template update behavior
- your real evaluation story should still use actual VPN traffic captured into `data/raw/<app_name>/`

## Suggested real demo flow

If you want to present the real pipeline in front of people:

1. Use the `FlowPic` tab to capture or import traffic and show the generated image.
2. Use the `Dataset` tab to build labeled FlowPics for known apps.
3. Use the `Training` tab to train the backbone once.
4. Use the `Library + Predict` tab to register known apps into the template library.
5. Predict a known sample and show that it matches.
6. Predict an unseen sample and show that it returns `Unknown`.
7. Register the unseen app into the library without retraining.
8. Predict the same unseen app again and show that it is now recognized.

## Command-line equivalents

Generate a FlowPic from a PCAP:

```powershell
python -m flowpic.generate --pcap path\to\capture.pcap --output outputs\preview.png --show
```

Capture live traffic:

```powershell
python -m flowpic.capture --list-interfaces
python -m flowpic.capture --output outputs\live_preview.png --duration 15 --interface "Wi-Fi" --filter "tcp or udp" --show
```

Build the dataset:

```powershell
python -m flowpic.data_builder --raw-root data\raw --processed-root data\processed
```

Train the model:

```powershell
python -m flowpic.train_pipeline --manifest data\processed\manifest.csv --checkpoint checkpoints\backbone_v1.pth --epochs 10 --batch-size 8
```

Register an app template:

```powershell
python -m flowpic.library register --label YouTube --checkpoint checkpoints\backbone_v1.pth --library templates\library.json --manifest data\processed\manifest.csv
```

Predict an unknown sample:

```powershell
python -m flowpic.predict --checkpoint checkpoints\backbone_v1.pth --library templates\library.json --image data\processed\YouTube\sample_w0000.png
```

Run the synthetic demo:

```powershell
python -m flowpic.demo --root demo_assets --epochs 6 --threshold 0.08
```

## Project folders

- `data/raw/<app_name>/`: raw PCAPs grouped by label
- `data/processed/<app_name>/`: generated FlowPic images
- `data/processed/manifest.csv`: dataset index for training
- `checkpoints/backbone_v1.pth`: trained CNN checkpoint
- `templates/library.json`: registered template embeddings
- `demo_assets/`: synthetic demo artifacts if you run the demo module

## Notes

- The system uses tunnel-level time windows, not isolated 5-tuple flows.
- Embeddings are L2-normalized to keep distance comparisons stable.
- New apps can be added by registering new template embeddings into the library without retraining the CNN.
- Live capture on Windows usually requires Npcap and an elevated PowerShell session.
