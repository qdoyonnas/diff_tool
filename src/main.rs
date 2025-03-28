#![cfg_attr(not(feature = "ai_priority"), allow(unused_imports))]

use std::fs::{self, File as FileSystem};
use std::io;
use std::path::{Path, PathBuf};
use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use walkdir::WalkDir;
use rayon::prelude::*;
use blake3;
use fastcdc::v2020::FastCDC;
use memmap2::Mmap;
use fs2::FileExt;
use std::time::Instant;
use clap::Parser;

#[cfg(feature = "ai_priority")]
use tch::{CModule, Tensor};

macro_rules! vprintln {
    ($enabled:expr, $($arg:tt)*) => {
        if $enabled {
            println!($($arg)*);
        }
    };
}

/// Command-line arguments accepted by the DiffTool
/// Defines the structure for command-line arguments using `clap`.
#[derive(Parser, Debug)]
#[command(name = "DiffTool")]
struct Args {
    /// JSON file to reference from and save to
    #[arg(long, default_value = "state")]
    state: String,

    /// Target directory to scan
    target: String,

    ///Output operations to JSON file
    #[arg(long)]
    json: Option<String>,

    /// Enable vebose logging
    /// Set verbosity level (-v for progress, -vv for detailed per-file output)
    #[arg(short = 'v', long = "verbose", action = clap::ArgAction::Count)]
    verbosity: u8,
}

/// Represents a file or directory's hashed state for comparison between runs.
/// Used to track changes across executions.
#[derive(Debug, Serialize, Deserialize)]
struct Fingerprint {
    relative_path: PathBuf,
    is_dir: bool,
    chunks: Option<Vec<String>>
}

/// Represents a file operation (create, delete, copy) needed to synchronize directories.
#[derive(Debug, Serialize, Deserialize)]
struct DiffOperation {
    op: String,
    path: String,
}

#[cfg(feature = "ai_priority")]
fn file_features(path: &Path, root: &Path) -> Vec<f32> {
    let size = fs::metadata(path).map(|m| m.len()).unwrap_or(0) as f32;
    let log_size = size.ln().max(0.0);

    let ext_hash = path.extension()
        .and_then(|e| e.to_str())
        .map(|ext| ext.bytes().fold(0u32, |acc, b| acc + b as u32) % 1000)
        .unwrap_or(0) as f32;

    let depth = path.strip_prefix(root).map(|p| p.components().count()).unwrap_or(0) as f32;

    let is_uasset = path.extension()
        .and_then(|e| e.to_str())
        .map(|e| (e == "uasset" || e == "umap") as i32 as f32)
        .unwrap_or(0.0);

    vec![log_size, ext_hash, depth, is_uasset]
}

#[cfg(feature = "ai_priority")]
fn predict_priority(model: &CModule, path: &Path, root: &Path) -> f32 {
    let features = file_features(path, root);
    let input = Tensor::from_slice(&features).unsqueeze(0);
    let output = model.forward_ts(&[input]).unwrap();
    output.double_value(&[0]) as f32
}

// Hashes a file into a vector of chunk hashes using memory mapping
/// Reads a file, splits it into content-defined chunks, and returns BLAKE3 hashes of each chunk.
/// Uses memory mapping and FastCDC for efficiency.
fn chunk_and_hash_file(file_path: &Path, args: &Args) -> io::Result<Vec<String>> {
    let file_name = file_path.file_name().unwrap();
    vprintln!(args.verbosity >= 2, "Started hashing '{}'...", file_name.to_str().unwrap());
    let start = Instant::now();

    let file = FileSystem::open(file_path)?;
    FileExt::lock_shared(&file)?;
    let mut chunk_hashes = Vec::new();
    let mmap = unsafe { Mmap::map(&file)? };

    for chunk in FastCDC::new(&mmap, 4096, 16384, 65535) {
        let data = &mmap[chunk.offset .. chunk.offset + chunk.length];
        chunk_hashes.push(blake3::hash(data).to_hex().to_string());
    }

    FileExt::unlock(&file)?;
    vprintln!(args.verbosity >= 2, "...finished hashing '{}' in: {:.2?}", file_name.to_str().unwrap(), start.elapsed());
    Ok(chunk_hashes)
}

#[cfg(feature = "ai_priority")]
fn load_model() -> CModule {
    let model = match CModule::load(Path::new("models/chunk_priority_model.pt")) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("ERROR: Failed to load model: {:?}", e);
            std::process::exit(1);
        }
    };

    model
}

// Scan a directory, collecting fingerprints and subdirectory paths
/// Scans the given directory and returns a list of fingerprints representing
/// the content and structure of the directory at the time of scanning.
/// Applies optional AI-based prioritization for hashing if enabled.
fn scan_directory(root: &Path, args: &Args) -> io::Result<Vec<Fingerprint>> {
    let start = Instant::now();

    #[cfg(feature = "ai_priority")]
    let model = load_model();

    let (prioritized_files, dirs) = collect_paths_with_priority(root, #[cfg(feature = "ai_priority")] &model);
    let fingerprints = hash_files_to_fingerprints(&prioritized_files, root, &args);

    let elapsed = start.elapsed();
    vprintln!(args.verbosity >= 1, "Scanned directory in {:.2?}", elapsed);

    let mut all = fingerprints;
    all.extend(dirs);
    Ok(all)
}


/// Traverses a directory tree, identifying all files and directories,
/// and assigns priority scores for sorting (AI-based if enabled).
fn collect_paths_with_priority(
    root: &Path,
    #[cfg(feature = "ai_priority")] model: &CModule,
) -> (Vec<(f32, PathBuf)>, Vec<Fingerprint>) {
    let mut prioritized_files = Vec::new();
    let mut dirs = Vec::new();

    for entry in WalkDir::new(root).into_iter() {
        let entry = match entry {
            Ok(e) => e,
            Err(e) => {
                eprintln!("ERROR: Failed to access path: {}", e);
                continue;
            }
        };
        let path = entry.path().to_path_buf();
        let relative_path = match path.strip_prefix(root) {
            Ok(p) => p.to_path_buf(),
            Err(e) => {
                eprintln!("ERROR: Failed to get relative path: {:?}", e);
                continue;
            }
        };

        if entry.file_type().is_dir() {
            dirs.push(Fingerprint {
                relative_path,
                is_dir: true,
                chunks: None,
            });
        } else if entry.file_type().is_file() {
            #[cfg(feature = "ai_priority")]
            let score = predict_priority(model, &path, root);

            #[cfg(not(feature = "ai_priority"))]
            let score = 0.0;

            prioritized_files.push((score, path));
        }
    }

    prioritized_files.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    (prioritized_files, dirs)
}


/// Processes a list of prioritized file paths by computing their content hashes
/// and producing corresponding Fingerprint structs.
fn hash_files_to_fingerprints(prioritized_files: &[(f32, PathBuf)], root: &Path, args: &Args) -> Vec<Fingerprint> {
    use indicatif::{ProgressBar, ProgressStyle};

    let progress = if args.verbosity == 1 {
        let pb = ProgressBar::new(prioritized_files.len() as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} files hashed")
            .unwrap()
            .progress_chars("##-"));
        Some(pb)
    } else {
        None
    };

    let fingerprints: Vec<Fingerprint> = prioritized_files.par_iter().filter_map(|(_, path)| {
        let relative_path = match path.strip_prefix(root) {
            Ok(p) => p.to_path_buf(),
            Err(e) => {
                eprintln!("ERROR: Failed to get relative path: {:?}", e);
                if let Some(pb) = &progress { pb.inc(1); }
                return None;
            }
        };

        let chunks = match chunk_and_hash_file(path, args) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("ERROR: chunk_and_hash_file failed: {:?}", e);
                if let Some(pb) = &progress { pb.inc(1); }
                return None;
            }
        };

        if let Some(pb) = &progress { pb.inc(1); }

        Some(Fingerprint {
            relative_path,
            is_dir: false,
            chunks: Some(chunks),
        })
    }).collect();

    if let Some(pb) = progress {
        pb.finish_with_message("File hashing complete");
    }

    fingerprints
}



/// Compares two sets of directory fingerprints and produces a list of operations
/// (create, delete, copy) needed to synchronize them.
fn diff_states(reference_state: &Vec<Fingerprint>, current_state: &Vec<Fingerprint>, args: &Args) -> Vec<DiffOperation> {
    let start = Instant::now();

    let reference_map: HashMap<&PathBuf, &Fingerprint> = reference_state.iter()
        .map(|f| (&f.relative_path, f)).collect();
    let current_map: HashMap<&PathBuf, &Fingerprint> = current_state.iter()
        .map(|f| (&f.relative_path, f)).collect();

    let mut operations: Vec<DiffOperation> = Vec::new();

    for (path, reference_fingerprint) in &reference_map {
        match current_map.get(path) {
            None => operations.push(DiffOperation { op: "delete".into(), path: path.display().to_string() }),
            Some(current_fingerprint) => {
                if reference_fingerprint.chunks != current_fingerprint.chunks {
                    operations.push(DiffOperation { op: "copy".into(), path: path.display().to_string() });
                }
            }
        }
    }

    for (path, _) in &current_map {
        if !reference_map.contains_key(path) {
            operations.push(DiffOperation { op: "create".into(), path: path.display().to_string() });
        }
    }

    let elapsed = start.elapsed();
    vprintln!(args.verbosity >= 1, "Computed diff in {:.2?}", elapsed);

    operations
}


/// Outputs the diff operations to either a human-readable format or a JSON file.
fn output_operations(operations: &[DiffOperation], json_output_file: Option<String>, verbosity: u8) -> io::Result<()> {
    if let Some(path) = json_output_file {
        let file = FileSystem::create(format!("{}.json", &path))?;
        serde_json::to_writer_pretty(file, &operations)?;
        println!("Sucessfully wrote operations to {}", path);
    } else {
        vprintln!(verbosity >= 1, "--------------------------------------");
        for op in operations {
            println!("{} `{}`", op.op, op.path);
        }
        vprintln!(verbosity >= 1, "--------------------------------------");
    }
    Ok(())
}


/// Serializes and saves the directory fingerprint state to a file.
fn save_state(state: Vec<Fingerprint>, file_name: String, verbosity: u8) -> io::Result<()> {
    let file = FileSystem::create(file_name)?;
    serde_json::to_writer(file, &state)?;
    vprintln!(verbosity >= 1, "Directory state saved.");

    Ok(())
}

/// Entry point: orchestrates scanning, diffing, and updating state.
fn main() -> io::Result<()> {
    let args = Args::parse();

    let directory_path = Path::new(&args.target);
    if !directory_path.exists() {
        eprintln!("ERROR: The specified directory '{}' does not exist.", directory_path.display());
        std::process::exit(1);
    }

    let file_name = format!("{}.json", args.state);

    if fs::exists(&file_name)? {
        vprintln!(args.verbosity >= 1, "Deserializing reference state...");
        let start = Instant::now();

        let file = FileSystem::open(&file_name).map_err(|e| {
            eprintln!("ERROR: Failed to open reference state file '{}': {}", file_name, e);
            e
        })?;
        let reference_state: Vec<Fingerprint> = serde_json::from_reader(file).map_err(|e| {
            eprintln!("ERROR: Failed to parse reference state '{}': {}", file_name, e);
            e
        })?;
        vprintln!(args.verbosity >= 1, "...done deserializing in {:.2?}", start.elapsed());

        let current_state = match scan_directory(directory_path, &args) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("ERROR: Failed to scan directory '{}': {}", directory_path.display(), e);
                return Err(e);
            }
        };
        let operations = diff_states(&reference_state, &current_state, &args);
        output_operations(&operations, args.json, args.verbosity)?;
        save_state(current_state, file_name, args.verbosity)?;
    } else {
        vprintln!(args.verbosity >= 1, "No reference state found, creating new state...");
        let state = match scan_directory(directory_path, &args) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("ERROR: Failed to scan directory '{}': {}", directory_path.display(), e);
                return Err(e);
            }
        };
        save_state(state, file_name, args.verbosity)?;
    }

    Ok(())
}
