# DiffTool

**A performant directory diff tool for large-scale Unreal Engine projects.**\
Built in Rust, with optional AI-enhanced chunk analysis and multithreaded hashing.

---

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Enabling AI Chunk Prioritization](#enabling-ai-chunk-prioritization)

---

## Overview

**DiffTool** is a directory diffing tool designed to compute the minimal set of operations required to synchronize a *target* directory with a *reference* directory. It focuses on speed, accuracy, and long-term maintainability—optimized for large projects such as Unreal Engine repositories.

Compares file contents using robust hashing strategies. Reference states are stored externally between runs to accommodate environments with one-sided directory access.

---

## Installation

### Prerequisites

- Rust (1.70 or later)
- Git (optional, for cloning)

### Build Instructions

```bash
git clone https://github.com/qdoyonnas/diff_tool.git
cd diff_tool
cargo build --release
```

### Run

```bash
./target/release/diff_tool --help
```

---

## Usage

```bash
diff_tool <path_to_target> [OPTIONS]
```

### Required Parameters

- Path to the directory to be updated/synced

### Optional Parameters

- `--state <path>`: Path to file where diff state is saved/loaded
- `--json <path>`: Output results in JSON format (for scripting/CI)
- `--verbose`, `-v`: Print processing info
- `--very_verbose`, `-v`: Print detailed processing info

---

## Enabling AI Chunk Prioritization 
# (Experimental)

DiffTool includes an optional AI-based chunk prioritization feature that can improve performance on large datasets by ordering file processing based on predicted relevance.

### Prerequisites

To enable the AI mode, ensure the following dependencies are installed:

- [LibTorch](https://pytorch.org/cppdocs/installing.html) C++ library (required by `tch` crate)
- Python-trained TorchScript model saved as `models/chunk_priority_model.pt`
- Enable the `ai_priority` feature flag during build

On Ubuntu (example):

```bash
sudo apt-get install libtorch-dev
```

Or follow the [LibTorch installation guide](https://pytorch.org/get-started/locally/) for your platform.

You must also set the following environment variables:

- LIBTORCH: Path to your local LibTorch installation directory
- Ensure all required LibTorch .dll files (on Windows) are in your system PATH

### Build with AI Support

Use the `--features ai_priority` flag to include AI support at compile time:

```bash
cargo build --release --features ai_priority
```

### Run with AI Model

Ensure the model is available at `models/chunk_priority_model.pt` relative to your working directory. Then execute DiffTool as normal:

```bash
./target/release/difftool ./your_directory
```

When compiled with AI support, the tool will use the model to prioritize file hashing based on size, depth, extension, and Unreal asset type.

---





