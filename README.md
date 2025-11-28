# Fish Re-Identification Project

**Automated stream fish re-identification using natural spot patterns and computer vision**

Re-identify individual fish across multiple years using natural spot patterns and estimate biometrics from images. This project implements a complete pipeline from raw images to standardized "passport photos" and scalable matching using the HotSpotter algorithm.

## Overview

This project provides a computer vision pipeline for wildlife conservation research, specifically designed for re-identifying individual fish using their unique spot patterns. The system processes raw field photographs through segmentation, metric extraction, standardization, and matching stages to enable automated individual identification.

### Key Features

- **SAM 3 Segmentation**: Text-prompted semantic segmentation for fish, eyes, fins, and rulers
- **Automated Metric Extraction**: Ruler-based scale calculation and fish length measurement
- **Standardized Passport Photos**: Consistent orientation and alignment for comparison
- **HotSpotter Matching**: Scalable one-vs-many matching with LNBNN scoring (O(N) complexity)
- **Affine Geometric Verification**: Handles flexible/curved fish bodies better than homography
- **Cross-Platform Support**: Works on Linux, macOS, and Windows

## Quick Start (Standardization Pipeline)

**Basic steps for just running the standardization pipeline**:

### 1. Install uv (if not already installed)

**Linux/macOS:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env  # or restart terminal
```

**Windows (PowerShell):**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Set up the project

```bash
# Clone and enter directory
git clone <repository-url>
cd fish_recap

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv sync
```

### 3. Install SAM 3

```bash
git clone https://github.com/facebookresearch/sam3.git
cd sam3
uv pip install -e .
cd ..
```

### 4. Set up Hugging Face access

1. **Create account**: Go to [https://huggingface.co/join](https://huggingface.co/join)
2. **Get token**: Settings → Access Tokens → New token (type: Read) → Copy token
3. **Request model access**: Visit [https://huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3) → Click "Request access"
4. **Authenticate**:
   ```bash
   uv pip install huggingface_hub
   huggingface-cli login  # Paste your token when prompted
   ```

### 5. Configure paths (optional)

**Option A: Use default paths** (images in `data/raw/`)

If your images are in the default location (`data/raw/`), you can skip this step.

**Option B: Customize paths in config file**

Edit `configs/config.yaml` to set your input/output directories. Paths can be:
- **Relative** (to project root): `"data/raw"` or `"../my_images"`
- **Absolute** (anywhere on your system): `"/home/user/images"` or `"C:\Users\user\images"`

```yaml
data:
  raw_dir: "/path/to/your/images"           # Input images
  standardized_dir: "/path/to/output"        # Standardized images
  masks_dir: "/path/to/masks"                # Segmentation masks
  metadata_dir: "/path/to/metadata"          # Metrics JSON
```

**Option C: Override paths via command line**

You can also override paths without editing config:

```bash
python3 scripts/process_images.py \
  --raw-dir "/path/to/your/images" \
  --standardized-dir "/path/to/output" \
  --masks-dir "/path/to/masks"
```

### 6. Add your images

Place images in your configured input directory with format: `{ID}{L|R}.jpg` (e.g., `CSDD001L.jpg`)

### 7. Run the pipeline

```bash
python3 scripts/process_images.py
```

That's it! Results will be in your configured output directories (default: `data/processed/standardized/` and `data/processed/qa/`).

**Need help?** See [Full Installation Guide](#installation) or [Troubleshooting](#troubleshooting).

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Workflow](#workflow)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Development](#development)
- [References](#references)

## Installation

### Prerequisites

- **Python 3.11+** (3.11 or 3.12 recommended)
- **uv** package manager ([Installation instructions](#installing-uv))

### Installing uv

**Linux/macOS:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

After installation, restart your terminal or run:
- Linux/macOS: `source $HOME/.cargo/env`
- Windows: Add `%USERPROFILE%\.cargo\bin` to your PATH

### Project Setup

1. **Clone the repository:**
```bash
git clone <repository-url>
cd fish_recap
```

2. **Create virtual environment and install dependencies:**
```bash
# Create virtual environment
uv venv

# Activate virtual environment
# Linux/macOS:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

# Install project dependencies
uv sync
```

3. **Install SAM 3 (Required for segmentation):**

SAM 3 must be installed separately as it's not available on PyPI:

```bash
# Clone SAM 3 repository
git clone https://github.com/facebookresearch/sam3.git
cd sam3

# Install in editable mode
uv pip install -e .

# Return to project directory
cd ..
```

**Windows Note:** You may need Visual Studio Build Tools for compiling SAM 3. Install from [Microsoft's website](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022).

4. **Set up Hugging Face authentication:**

SAM 3 requires Hugging Face authentication for model checkpoint access. If you're new to Hugging Face, follow these steps:

**Step 4a: Create a Hugging Face Account**

1. Go to [https://huggingface.co/join](https://huggingface.co/join)
2. Sign up with your email, GitHub account, or Google account
3. Verify your email address (check your inbox)

**Step 4b: Create an Access Token**

1. After logging in, go to your profile settings:
   - Click your profile picture (top right)
   - Select **Settings**
2. Navigate to **Access Tokens** in the left sidebar
3. Click **New token** button
4. Configure your token:
   - **Name**: `fish-recap-sam3` (or any descriptive name)
   - **Type**: Select **Read** (sufficient for downloading models)
   - Click **Generate token**
5. **Important**: Copy the token immediately - you won't be able to see it again!
   - It will look like: `hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`
   - Save it securely (password manager, secure note, etc.)

**Step 4c: Request Access to SAM 3 Model**

1. Visit the SAM 3 model page: [https://huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3)
2. Click the **Request access** button (you may need to be logged in)
3. Wait for approval (usually instant, but can take a few minutes)
4. You'll receive an email notification when access is granted

**Step 4d: Authenticate with Hugging Face**

Now authenticate using your access token:

```bash
# Install Hugging Face CLI (if not already installed)
uv pip install huggingface_hub

# Login to Hugging Face (will prompt for your token)
huggingface-cli login

# When prompted, paste your access token and press Enter
# Token: hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

**Alternative: Environment Variable Method**

If you prefer not to use the CLI login, you can set an environment variable:

**Linux/macOS:**
```bash
export HUGGING_FACE_HUB_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

**Windows (PowerShell):**
```powershell
$env:HUGGING_FACE_HUB_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

**Windows (Command Prompt):**
```cmd
set HUGGING_FACE_HUB_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

**Troubleshooting Authentication:**

- **Token not working**: Make sure you copied the entire token (starts with `hf_`)
- **Access denied**: Ensure you've requested and been granted access to the SAM 3 model
- **Token expired**: Generate a new token from Settings → Access Tokens
- **CLI not found**: Make sure `huggingface_hub` is installed: `uv pip install huggingface_hub`

5. **Verify installation:**
```bash
# Test SAM 3 access
uv run python scripts/check_hf_access.py

# Run tests
uv run pytest tests/
```

## Quick Start

### Basic Workflow

1. **Place your images in `data/raw/`:**
   - Format: `{ID}{L|R}.jpg` (e.g., `CSDD001L.jpg`, `CSDD001R.jpg`)
   - Left (`L`) and Right (`R`) sides of fish

2. **Run preprocessing pipeline:**
```bash
uv run python scripts/process_images.py
```

This will:
- Segment fish, eyes, fins, and rulers using SAM 3
- Extract metrics (scale, fish length)
- Create standardized passport photos
- Generate QA visualizations

3. **Run matching:**
```bash
# HotSpotter matching (default, recommended)
uv run python scripts/match_fish.py

# Or pairwise matching (slower, for small datasets)
uv run python scripts/match_fish.py --matching-method pairwise
```

4. **View results:**
   - Matching results: `data/results/matching_results.json`
   - QA visualizations: `data/processed/qa/`
   - Metrics: `data/metadata/metrics.json`

## Workflow

### Preprocessing Pipeline

The preprocessing pipeline converts raw images into standardized "passport photos":

#### Step 1: Segmentation (SAM 3)

**Script:** `scripts/process_images.py`

Segments fish, eyes, dorsal fins, and rulers using text prompts:

```bash
uv run python scripts/process_images.py
```

**Output:**
- `data/processed/masks/{image}_whole fish_mask.png`
- `data/processed/masks/{image}_fish eye_mask.png`
- `data/processed/masks/{image}_dorsal fin_mask.png`
- `data/processed/masks/{image}_ruler_mask.png`

**Options:**
- `--verbose`: Detailed logging
- `--no-metrics`: Skip metric extraction (run standardization only)
- `--no-standardize`: Skip standardization
- `--no-skip-existing`: Force reprocessing of all steps (by default, completed steps are skipped)

**Skip Logic:**
The script intelligently skips completed steps by checking for existing outputs:
- **Segmentation**: Skipped if masks exist in `masks_dir`
- **Metrics**: Skipped if entry exists in `metadata_dir/metrics.json` with valid metrics
- **Standardization**: Skipped if both `{image}_standardized.png` and `{image}_standardized_mask.png` exist in `standardized_dir`

When running standardization-only mode (`--no-metrics`), images with existing standardized outputs are automatically skipped, even if segmentation masks are missing. This allows you to re-run standardization on new images without reprocessing already-standardized ones.

**Examples:**
```bash
# Run standardization only (skip images that already have standardized outputs)
uv run python scripts/process_images.py --no-metrics

# Force reprocessing of all steps (ignore existing outputs)
uv run python scripts/process_images.py --no-skip-existing

# Run only segmentation (skip metrics and standardization)
uv run python scripts/process_images.py --no-metrics --no-standardize
```

#### Step 2: Metric Extraction

Automatically extracts:
- **Scale factor** (`pixels_per_mm`) from ruler tick marks
- **Fish length** in pixels and millimeters
- **Distribution analysis** for outlier detection

**Output:**
- `data/metadata/metrics.json`
- `data/metadata/scale_distribution.json`
- QA plots: `data/processed/qa/{image}_ruler_qa.png`

#### Step 3: Standardization

Creates aligned passport photos:
- **Rotation**: PCA-based alignment (fish horizontal)
- **Orientation**: Head-right, dorsal-up (using eye/fin masks)
- **Cropping**: Fish bounding box with margin
- **Background removal**: Solid color background

**Output:**
- `data/processed/standardized/{image}_standardized.png`
- `data/processed/standardized/{image}_standardized_mask.png`
- QA plots: `data/processed/qa/{image}_standardization_qa.png`

### Matching Pipeline

#### HotSpotter Matching (Recommended)

**Script:** `scripts/match_fish.py`

Scalable one-vs-many matching using LNBNN scoring:

```bash
# Default: HotSpotter matching
uv run python scripts/match_fish.py

# Query specific image
uv run python scripts/match_fish.py --query CSDD001L

# Force re-extraction of features
uv run python scripts/match_fish.py --force-recompute
```

**How it works:**
1. **Feature Extraction**: SIFT/RootSIFT descriptors from standardized images
2. **Global Index**: Single FLANN index for all database descriptors
3. **LNBNN Scoring**: Local Naive Bayes Nearest Neighbor scoring
4. **Spatial Reranking**: Affine RANSAC verification on top candidates

**Performance:**
- O(N) complexity (vs O(N²) for pairwise)
- Matches 1000+ images in seconds
- Handles ambiguous features naturally

#### Pairwise Matching (Alternative)

For small datasets or comparison:

```bash
uv run python scripts/match_fish.py --matching-method pairwise
```

**Output:**
- `data/results/matching_results.json`: Ranked matches per query
- Format: `{query_id: [{match_id, score, inliers, total_matches}, ...]}`

## Project Structure

```
fish_recap/
├── src/
│   ├── preprocessing/          # Image preprocessing modules
│   │   ├── segmentation.py     # SAM 3 segmentation
│   │   ├── metric_extraction.py # Ruler-based metrics
│   │   └── standardization.py  # Passport photo creation
│   ├── matching/               # Matching engine
│   │   ├── feature_extraction.py # SIFT/RootSIFT
│   │   ├── matcher.py          # HotSpotter + Pairwise
│   │   ├── geometric_verification.py # RANSAC (Affine/Homography)
│   │   └── deep_metric.py      # DINOv2 embeddings
│   └── utils/                  # Utilities
│       ├── image_utils.py
│       ├── visualization.py
│       └── data_utils.py
├── scripts/
│   ├── process_images.py       # Preprocessing pipeline
│   ├── match_fish.py           # Matching pipeline
│   ├── test_sam3.py            # SAM 3 testing
│   └── check_hf_access.py     # HF authentication check
├── configs/
│   └── config.yaml             # Configuration file
├── data/
│   ├── raw/                    # Raw input images
│   ├── processed/
│   │   ├── masks/              # Segmentation masks
│   │   ├── standardized/       # Passport photos
│   │   ├── features/            # Extracted features (pickle)
│   │   └── qa/                 # QA visualizations
│   ├── metadata/               # Metrics and metadata
│   └── results/                # Matching results
├── tests/                      # Unit tests
├── pyproject.toml             # UV project configuration
├── uv.lock                    # Dependency lock file
└── README.md                  # This file
```

## Configuration

Configuration is managed via `configs/config.yaml`. Key parameters:

### Segmentation
```yaml
segmentation:
  prompts: ["whole fish", "fish eye", "dorsal fin", "ruler"]
  device: "cuda"  # or "cpu"
  confidence_threshold: 0.5
```

### Matching
```yaml
matching:
  method: "hotspotter"  # or "pairwise"
  lnbnn_k: 5            # Nearest neighbors for LNBNN
  top_candidates: 20    # Candidates for spatial reranking
```

### Geometric Verification
```yaml
geometric_verification:
  model: "affine"       # or "homography"
  ransac_threshold: 5.0
  min_inliers: 10
```

See `configs/config.yaml` for all options.

## Troubleshooting

### Common Issues

#### SAM 3 Access Denied
**Problem:** `403 Client Error: Forbidden` when loading SAM 3 model

**Solution:**
1. **Verify you have a Hugging Face account**: If you're new, see [Hugging Face Setup](#step-4-set-up-hugging-face-authentication) above
2. **Request model access**: Visit [https://huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3) and click "Request access"
3. **Check access status**: Wait for approval email (usually instant, but can take a few minutes)
4. **Verify authentication**: 
   ```bash
   # Test if you're logged in
   huggingface-cli whoami
   
   # If not logged in, authenticate
   huggingface-cli login
   ```
5. **Check token permissions**: Ensure your access token has "Read" permissions
6. **Try refreshing**: Sometimes you need to wait a few minutes after access is granted

**Still having issues?**
- Verify your token is valid: Check Settings → Access Tokens in your Hugging Face account
- Try generating a new token if the old one might be expired
- Check that you're using the correct token (starts with `hf_`)

#### No Features Extracted
**Problem:** Matching fails with "No features extracted"

**Solution:**
- Check standardized images exist: `data/processed/standardized/`
- Verify images aren't empty or corrupted
- Try lowering SIFT contrast threshold in config

#### Windows: SAM 3 Installation Fails
**Problem:** Compilation errors on Windows

**Solution:**
1. Install Visual Studio Build Tools
2. Install CUDA toolkit (if using GPU)
3. Use Windows Subsystem for Linux (WSL) as alternative

#### Out of Memory
**Problem:** GPU out of memory during segmentation

**Solution:**
- Process images in smaller batches
- Use CPU: Set `device: "cpu"` in config
- Reduce image resolution (pre-process images)

### Windows-Specific Notes

- **Path Separators**: Code uses `pathlib.Path` (cross-platform)
- **Line Endings**: Git handles automatically
- **Scripts**: Use `uv run python scripts/...` instead of direct execution
- **SAM 3**: May require Visual Studio Build Tools for compilation

### Performance Tips

- **GPU**: Use CUDA for SAM 3 (10x faster)
- **Batch Processing**: Process multiple images (automatic)
- **Feature Caching**: Features are cached in `data/processed/features/`
- **Progress Tracking**: Resume interrupted runs (automatically skips completed steps based on output file existence)

## Development

### Setup Development Environment

```bash
# Install with dev dependencies
uv sync --dev

# Install pre-commit hooks (optional)
uv run pre-commit install
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_matching.py

# With coverage
uv run pytest --cov=src tests/
```

## References

### Papers

- **HotSpotter**: Crall et al. "HotSpotter - Patterned Species Instance Recognition" (2013)
- **SAM 3**: Meta AI Research - Segment Anything Model 3
- **SIFT**: Lowe, D. G. "Distinctive image features from scale-invariant keypoints" (2004)

### Documentation

- [SAM 3 Repository](https://github.com/facebookresearch/sam3)
- [OpenCV Documentation](https://docs.opencv.org/)
- [UV Documentation](https://docs.astral.sh/uv/)

### Related Projects

- [Wild-ID](https://github.com/wildmeorg/wildbook) - Similar approach for wildlife ID
- [StripeSpotter](https://github.com/mayanklahiri/stripespotter) - Zebra identification

