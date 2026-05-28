# Audio Identification Pipeline

## Summary

## Architecture

# Code Execution 

This folder contains a small audio identification pipeline built around constellation maps, hash generation, and inverted-index matching.

The current implementation is split across two main Python files:

- [audio_fingerprinting.py](audio_fingerprinting.py)
- [main.py](main.py)

The pipeline currently uses a CQT front-end in the production code path, although the notebook work in this folder also explores STFT, Mel spectrogram, and MFCC variants.

## Overview

The workflow is:

1. Load an audio file.
2. Compute a time-frequency representation.
3. Detect local peaks to build a constellation map.
4. Generate hashes from peak pairs with bounded target-zone constraints.
5. Store the hashes in an inverted index keyed by hash tuple.
6. Run query audio through the same process and count hash matches.

The query result is a ranked list of database files ordered by number of matching hashes.

## Files

### [audio_fingerprinting.py](audio_fingerprinting.py)

Contains the feature extraction and hashing logic:

- `get_coordinates_from_audio_file(...)`
	- Loads audio with `librosa` at 22050 Hz.
	- Computes the configured spectrogram representation.
	- Runs `peak_local_max(...)` to extract constellation map coordinates.
- `generate_hashes(...)`
	- Sorts peak coordinates by time.
	- Builds anchor-target hashes from nearby peaks.
	- Applies bounded target-zone constraints using `min_dt`, `max_dt`, `min_df`, and `max_df`.
- `get_file_matches(...)`
	- Compares query hashes against the inverted index.
	- Counts votes per database filename.
- `denoise_query(...)`
	- Applies low-rank NMF reconstruction to a query spectrogram before peak picking.

### [main.py](main.py)

Contains the command-line entry point and indexing/query orchestration:

- `fingerprintBuilder(...)`
	- Walks the database folder.
	- Extracts coordinates and hashes for each audio file.
	- Writes the fingerprint index to `.pkl` or `.json`.
- `audioIdentification(...)`
	- Loads the fingerprint index.
	- Processes each query file.
	- Ranks database files by match count.
- `main()`
	- Exposes the `--database`, `--fingerprints`, `--queryset`, `--output`, and `--debug` CLI flags.

## Current Production Configuration

The production defaults live in `config_params` inside [main.py](main.py).

Key values:

- `fan_out = 20`
- `min_distance = 5`
- `threshold_rel = 0.02`
- `threshold_abs = None`
- `exclude_border = False`
- `num_peaks = 0.25`
- `min_dt = 1`
- `max_dt = 200`
- `min_df = 0`
- `max_df = 100`

These control peak density, hash fan-out, and the bounded target zone used when pairing peaks.

## Hash Format

The current hash key is a 3-tuple:

- anchor frequency bin
- target frequency bin
- time difference between anchor and target

Each hash is stored with the anchor time as the posting value.

The inverted index stores multiple postings per hash:

```python
{(anchor_k, target_k, dt): [(file_name, anchor_time), ...]}
```

## Command-Line Usage

### Build fingerprints

```bash
python main.py --database database_recordings --fingerprints fingerprints.pkl
```

### Run identification

```bash
python main.py --queryset query_recordings --fingerprints fingerprints.pkl --output output.txt
```

### Debug mode

Add `--debug` to stop after the first few files and print extra intermediate information:

```bash
python main.py --database database_recordings --fingerprints fingerprints.pkl --debug
python main.py --queryset query_recordings --fingerprints fingerprints.pkl --output output.txt --debug
```

## Output Files

### Fingerprints

- `.pkl` stores the inverted index as a Python dictionary.
- `.json` stores the same structure in a serializable form.

### Query Results

The output file contains one line per query file:

```text
query_file.wav    ranked_match_1.wav    ranked_match_2.wav    ranked_match_3.wav
```

## Notebook Work

The notebook `parameter-testing.ipynb` explores:

- spectrogram comparisons
- peak picking settings
- fan-out tuning
- triplet construction
- Phase 4 hash packing

That notebook is useful for understanding why the production parameters were chosen.

## Notes

- The code currently uses `librosa` and `scikit-image` for feature extraction and peak detection.
- The fingerprinting pipeline is tuned for a single-track database/query matching assignment, not for large-scale retrieval.
- CQT has been the best-performing representation in the recorded tests so far.
