
# EEG‑Viz

A Python toolkit for reproducible EEG data visualization in driving hazard detection research, supporting both event‑related potentials (ERP) and frequency‑domain topographic maps (topomaps) for expert and novice driver comparisons. 

## Repository Structure  

```text
├── scripts/
│   ├── erp_viz.ipynb                # Unified ERP visualization (time‑domain)
│   ├── freq_topomap_virtual.ipynb   # Frequency‑domain topomaps for virtual environment
│   └── freq_topomap_real.ipynb      # Frequency‑domain topomaps for real‑world data
├── environment.yml                  # Conda environment specification
├── requirements.txt                 # pip dependencies。
└── README.md                        # Project overview and usage
```

## Features

* **ERP Visualization**: Generates publication‑quality line plots with baseline correction and shaded temporal windows for channels of interest.
* **Frequency‑Domain Topomaps**: Computes mean power in dB across specified bands and time windows for expert vs. novice drivers, with:

  * **Sequential maps** (white→blue) for each group
  * **Diverging difference maps** (orange→white→blue) highlighting Expert − Novice contrasts

## Installation

### Conda (recommended)

```bash
conda env create -f environment.yml
conda activate eeg-viz
```

### Pip

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
.\.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

## Usage

### 1. ERP Visualization

```bash
python scripts/erp_viz.ipynb \
  --set-ref-dir /path/to/eeglab/sets \
  --set-ref-file subject01.set \
  --csv-dir /path/to/csv_time/output \
  --output-dir /path/to/erp_figures \
  --envs v r \
  --n-expert 36 \
  --n-novice 64 \
  --tmin -200 \
  --tmax 600 \
  --sfreq 1000
```

* **--envs**: `v` for virtual, `r` for real
* Generates `.png` and `.svg` Figures per channel under `<output-dir>`

### 2. Frequency‑Domain Topomaps (Virtual)

```bash
python scripts/freq_topomap_virtual.ipynb \
  --set-ref-dir /path/to/eeglab/sets \
  --set-ref-file subject01.set \
  --csv-dir /path/to/csv_freq/output \
  --output-dir /path/to/topomap_figures \
  --conditions h o c \
  --bands theta gamma1 \
  --n-expert 36 \
  --n-novice 64 \
  --contrast 1.5 \
  --percentile 98 \
  --t-range 50-350 250-350
```

### 3. Frequency‑Domain Topomaps (Real)

```bash
python scripts/freq_topomap_real.ipynb \
  --set-ref-dir /path/to/eeglab/sets \
  --set-ref-file subject01.set \
  --csv-dir /path/to/csv_freq/output \
  --output-dir /path/to/topomap_figures \
  --conditions h o c \
  --bands delta theta beta gamma1 \
  --n-expert 36 \
  --n-novice 64 \
  --contrast 1.5 \
  --percentile 98 \
  --t-range 50-550 150-250
```

## Data

If you need our data, please follow the download link in paper, or contact us :)
