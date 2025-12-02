# Fish Re-Identification Pipeline

This repository contains a pipeline for automated stream fish re-identification using natural spot patterns and computer vision.

## Installing uv

Before setting up the project, you need to install `uv`, a fast Python package installer and resolver.

**PowerShell (Recommended):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Alternative: Using pip (if you already have Python installed):**
```powershell
pip install uv
```

After installation, restart your terminal or run `refreshenv` to ensure `uv` is available in your PATH.

Verify the installation by running:
```powershell
uv --version
```

## Prerequisites

- Python 3.11 or higher
- [uv](https://github.com/astral-sh/uv) package manager installed
- CUDA-capable GPU (recommended for faster processing)

## Setup

### 1. Create and Sync Virtual Environment

Open PowerShell or Command Prompt in the project directory and run:

```powershell
uv venv
uv sync
```

This will create a virtual environment and install all required dependencies.

### 2. Activate the Virtual Environment

**PowerShell:**
```powershell
.\.venv\Scripts\Activate.ps1
```

**Command Prompt:**
```cmd
.venv\Scripts\activate.bat
```

### 3. Configure the Pipeline

Edit `configs/config.yaml` to set the paths for your data directories:

- **`data.raw_dir`**: Path to directory containing raw fish images (JPG files)
- **`data.standardized_dir`**: Path where standardized images will be saved (or already exist)
- **`data.masks_dir`**: Path where segmentation masks will be saved
- **`data.metadata_dir`**: Path where metadata JSON files will be saved
- **`data.features_dir`**: Path where extracted features will be saved

You can use either absolute paths (e.g., `C:\Users\YourName\Documents\fish_images`) or relative paths (relative to the project root).

## Running the Pipeline

### Option A: Full Pipeline (Process Raw Images)

If you have raw fish images that need to be processed:

1. **Process Images** (segmentation, metric extraction, and standardization):
   ```powershell
   python scripts/process_images.py
   ```

   This script will:
   - Segment fish and rulers from images using SAM3
   - Extract metrics (length measurements, scale calibration)
   - Create standardized "passport photos" of fish

   Optional flags:
   - `--no-metrics`: Skip metric extraction
   - `--no-standardize`: Skip standardization
   - `--no-skip-existing`: Reprocess all images even if already processed
   - `--verbose` or `-v`: Enable detailed logging

2. **Run Matching** (feature extraction and matching):
   ```powershell
   python scripts/match_fish.py
   ```

   This script will:
   - Extract SIFT/RootSIFT features from standardized images
   - Build a feature database
   - Run matching queries (all-vs-all by default)
   - Perform geometric verification
   - Save results to `data/results/matching_results.json`

   Optional flags:
   - `--query <image_id>`: Query a specific image instead of all-vs-all
   - `--force-recompute`: Force re-extraction of features
   - `--matching-method hotspotter|pairwise`: Override matching method

### Option B: Matching Only (If Standardized Images Already Exist)

If you already have standardized images and have configured the `standardized_dir` path in `configs/config.yaml`:

Simply run the matching script:

```powershell
python scripts/match_fish.py
```

## Exploring Results

After running the matching pipeline, use the Quarto notebook `compare_fish_py.qmd` to explore and visualize results:

1. **Open the notebook** in your Quarto-compatible editor (e.g., RStudio, VS Code with Quarto extension)

2. **Run the Setup cell** to:
   - Load configuration
   - Set up paths
   - Load matching results

3. **Run the "Create Matches DataFrame" cell** to:
   - Convert matching results to a structured DataFrame
   - Extract side information (left/right) from image IDs
   - Calculate match statistics (scores, inlier ratios)
   - Save results to `data/results/matches.csv`

4. **Run the "Fish Comparison Function" cell** to:
   - Compare specific fish pairs visually
   - Update the `FISH_1` and `FISH_2` variables with your desired fish IDs
   - View side-by-side comparisons with match scores

The notebook filters out self-matches (same fish, different sides) and provides visualizations of the matching results.

## Output Files

- **`data/processed/masks/`**: Segmentation masks for fish, rulers, eyes, and fins
- **`data/processed/qa/`**: Quality assurance visualizations
- **`data/metadata/metrics.json`**: Extracted metrics (lengths, scale calibrations)
- **`data/metadata/scale_distribution.json`**: Analysis of scale distribution
- **`data/processed/features/`**: Extracted SIFT features (pickle files)
- **`data/results/matching_results.json`**: Full matching results
- **`data/results/matches.csv`**: Structured match data for analysis

