# Synthetic Medical Imaging System

A production-ready AI system for generating fully synthetic medical images (X-ray, MRI, CT) using Stable Diffusion and Large Language Models.

## ⚠️ Important Safety Notice

**All generated images are 100% synthetic and artificial. This system:**
- Does NOT use any real patient data
- Should NOT be used for diagnostic purposes
- Is intended for research, education, and AI training only
- Generates completely artificial medical imagery

## Features

- **Multi-modal Support**: X-ray, MRI, and CT scan generation
- **LLM-Powered Prompts**: Intelligent medical prompt generation
- **Flexible Configuration**: Adjustable detail levels, resolutions, and parameters
- **Safety-First Design**: Built-in filters and validation
- **Production Ready**: Clean architecture, logging, and error handling

## Installation

### 1. Prerequisites
- Python 3.8 or higher
- 8GB+ RAM (16GB recommended)
- CUDA-capable GPU (optional but recommended)
- 15GB disk space for models

### 2. Setup

```bash
# Clone or create project directory
mkdir synthetic_medical_imaging
cd synthetic_medical_imaging

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure API Keys

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your API keys
# Minimum: ANTHROPIC_API_KEY or OPENAI_API_KEY
```

### 4. First Run

The system will automatically download required models (~5-10GB) on first use.

## Usage

### Basic Usage

```bash
python main.py --modality xray --region chest --count 3
```

### Advanced Options

```bash
python main.py \
  --modality mri \
  --region brain \
  --detail high \
  --count 5 \
  --resolution 768 \
  --angle axial \
  --llm claude \
  --safety-mode strict
```

### Available Parameters

- `--modality`: xray, mri, ct (default: xray)
- `--region`: chest, brain, abdomen, spine, etc. (default: chest)
- `--detail`: low, medium, high (default: medium)
- `--count`: Number of images (default: 1)
- `--resolution`: 512, 768, 1024 (default: 512)
- `--angle`: Imaging view (auto-selected if not provided)
- `--llm`: claude, gpt, huggingface (default: claude)
- `--safety-mode`: strict, moderate, permissive (default: strict)
- `--seed`: Random seed for reproducibility
- `--steps`: Number of inference steps (overrides detail default)

## Project Structure

```
synthetic_medical_imaging/
├── config.py
├── prompt_generator.py
├── image_generator.py
├── llm_wrapper.py
├── safety_filter.py
├── main.py
├── requirements.txt
├── .env.example
├── outputs/
```

