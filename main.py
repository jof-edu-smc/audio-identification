import argparse
import os
import json
import pickle
from tqdm import tqdm 
from audio_fingerprinting import *

config_params = {
    # Limit the number of targets emitted from each anchor peak.
        "fan_out": 7,  # Number of target peaks to pair with each anchor peak
    # Use a denser neighborhood to keep more peaks per file.
        "min_distance": 5,  # Minimum distance between peaks in time frames
    # Keep moderately low thresholding so the CQT remains expressive.
        "threshold_rel": 0.02,  # Relative threshold for peak detection
    # Do not force an absolute threshold cutoff.
        "threshold_abs": None,  # Absolute threshold for peak detection
    # Allow peaks near the spectrogram borders to remain available.
        "exclude_border": False,  # Whether to exclude peaks near the borders of the spectrogram
    # Keep a fractional peak budget relative to spectrogram size.
        "num_peaks": 0.25,  # Maximum number of peaks to detect as a fraction of total spectrogram size
    # Minimum time gap allowed between anchor and target peaks.
        "min_dt": 1,  # Minimum time difference (in frames) between anchor and target peaks
    # Maximum time gap allowed between anchor and target peaks.
        "max_dt": 200,  # Maximum time difference (in frames) between anchor and target peaks
    # Reject pairs that are too close in frequency.
        "min_df": 0,  # Minimum frequency bin difference between anchor and target peaks
    # Reject pairs that are too far apart in frequency.
        "max_df": 100,  # Maximum frequency bin difference between anchor and target peaks
    }
# print(f"Using configuration: {config_params}")

def to_json_safe_inverted_list(inverted_list):
    """Convert an inverted index with tuple keys into a JSON-serializable list.

    Each hash tuple key ``(anchor_k, target_k, n_diff)`` is stored as a list,
    while postings remain ``(song_id, anchor_time)`` pairs.

    Args:
        inverted_list (dict): Inverted index mapping hash tuples to posting
            lists of ``(song_id, anchor_time)`` pairs.

    Returns:
        list[dict]: JSON-safe records with ``hash`` and ``value`` fields.
    """
    json_ready = []
    for hash_key, value in inverted_list.items():
        anchor_k, tgt_k, n_diff = hash_key
        json_ready.append(
            {
                "hash": [int(anchor_k), int(tgt_k), int(n_diff)],
                "value": [(song_id, k_diff) for song_id, k_diff in value],
            }
        )
    return json_ready            

def fingerprintBuilder(database_path='database_recordings', 
                       fingerprints_path='fingerprints.pkl', 
                       debug=False,
                    ):
    """Build a fingerprint inverted index from database audio files.

    The function scans ``database_path`` for ``.wav`` files, extracts
    fingerprint hashes per file, and saves the resulting index to either
    ``.pkl`` or ``.json`` according to ``fingerprints_path``.

    Args:
        database_path (str): Directory containing database ``.wav`` files.
        fingerprints_path (str): Output path for saved fingerprints. Supports
            ``.pkl`` and ``.json``.
        debug (bool): If True, prints extra diagnostics and stops early.

    Returns:
        None: Writes fingerprints to disk. Returns ``None`` early if
            ``database_path`` does not exist.
    """
    if debug:
        # Print the build configuration when debugging is enabled.
        print(f"Database Path: {database_path}")
        print(f"Fingerprints Path: {fingerprints_path}")
    
    fan_out = config_params["fan_out"]
    
    try:
        db = sorted(os.listdir(database_path))
    except FileNotFoundError:
        print(f"Database path {database_path} does not exist. Please provide a valid path.")
        return None
    
    num_files = len(db)
    if debug: 
        print(f"Number of files in the path are: {len(db)}")
    inverted_list = {}
    
    for file_num in tqdm(range(num_files), desc="Processing audio files", unit="file"):
        file = db[file_num]
        if file.endswith(".wav"):
            audio_path = os.path.join(database_path, file)
            coordinates, stft = get_coordinates_from_audio_file(audio_path, config=config_params, debug=debug)
    
            hashes = generate_hashes(
                coordinates,
                stft,
                fan_out=fan_out,
                min_dt=config_params["min_dt"],
                max_dt=config_params["max_dt"],
                min_df=config_params["min_df"],
                max_df=config_params["max_df"],
                debug=debug,
            )
            
            for hash, anchor_n in hashes:
                if hash not in inverted_list:
                    inverted_list[hash] = [] 
                inverted_list[hash].append((file, anchor_n)) 
            
            if debug:
                print(f"Displaying Built Inverted List for {file}: {len(inverted_list)}")
                raise StopIteration("Debug mode: Stopping after processing 1 file.")
            
    root, ext = os.path.splitext(fingerprints_path)
    if debug:
        print(f"Root: {root} and {ext}")
    if ext == '.json':
        print(f"Saving fingerprints to JSON file: {fingerprints_path}")
        with open(fingerprints_path, 'w', encoding='utf-8') as f:
            json.dump(to_json_safe_inverted_list(inverted_list), f, indent=2)
    elif ext == '.pkl': 
        print(f"Saving fingerprints to Pickle file: {fingerprints_path}")
        with open(fingerprints_path, 'wb') as f:
            pickle.dump(inverted_list, f)
    elif not ext:
        print(f"No file extension provided. Defaulting to JSON format for: {fingerprints_path}")
        path = fingerprints_path + '.json'
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(to_json_safe_inverted_list(inverted_list), f, indent=2)
    else: 
        raise ValueError('Only .pkl and .json are supported for fingerprint saving')
        
def audioIdentification(queryset_path='query_recordings', 
                        fingerprints_path='fingerprints.pkl', 
                        output_path='output.txt', 
                        debug=False
                    ):
    """Identify query audio files by matching against stored fingerprints.

    Loads a previously built fingerprint index, generates query hashes for each
    ``.wav`` file in ``queryset_path``, ranks matching database files, and
    writes top predictions to ``output_path``.

    Args:
        queryset_path (str): Directory containing query ``.wav`` files.
        fingerprints_path (str): Path to stored fingerprints (``.pkl`` or
            JSON format).
        output_path (str): File where ranked top-3 matches are written.
        debug (bool): If True, prints detailed matching diagnostics.

    Returns:
        dict | None: Returns a mapping of query files to ranked matches
            (or detailed debug output). Returns ``None`` if ``queryset_path``
            does not exist.
    """
    try: 
        queryset_files = sorted(os.listdir(queryset_path))
        fan_out = config_params["fan_out"]
    except FileNotFoundError:
        print(f"Queryset path {queryset_path} does not exist. Please provide a valid path.")
        return None
    
    if debug:
        print(f"Number of files in the queryset path are: {len(queryset_files)}")
    
    if fingerprints_path.endswith('.pkl'):
        fingerprints = pickle.load(open(fingerprints_path, 'rb'))
    else:
        fingerprints_json = json.load(open(fingerprints_path, 'r'))
        fingerprints = {}
        for entry in fingerprints_json:
            hash_tuple = tuple(entry['hash'])
            fingerprints[hash_tuple] = entry['value']
    
    if debug:
        print(f"Loaded {len(fingerprints)} hashes from fingerprints")
        detailed_output = {}
    
    output = {}
    i = 0
    for file in tqdm(queryset_files, desc="Processing query files", unit="file"):
        if file.endswith(".wav"):
            audio_path = os.path.join(queryset_path, file)
            query_id = file.split(".")[1] if "." in file else file 
            
            coordinates, stft = get_coordinates_from_audio_file(audio_path, config=config_params, debug=debug)
            query_hashes = generate_hashes(
                coordinates,
                stft,
                fan_out=fan_out,
                min_dt=config_params["min_dt"],
                max_dt=config_params["max_dt"],
                min_df=config_params["min_df"],
                max_df=config_params["max_df"],
                debug=debug,
            )
            
            if debug:
                # Report query-level hashing statistics.
                print(f"Processing query file: {file}, Extracted ID: {query_id}")
                print(f"Generated {len(query_hashes)} hashes for {file}")
            
            match_counts = get_file_matches(query_hashes, fingerprints)
            sorted_matches = sorted(match_counts.items(), key=lambda x: x[1], reverse=True)
            
            if debug:
                print(f"Sorted Matches for {file}: {sorted_matches}")  # Show top 5 matches

            output[file] = [f for f, _ in sorted_matches]
            
            if debug:
                detailed_output[file] = {
                    "total_query_hashes": len(query_hashes),
                    "total_matches": sum(match_counts.values()),
                    "matches": dict(sorted_matches),
                }
                
                for db_file, count in sorted_matches[:10]:  
                    print(f"  {db_file}: {count} matches")
                    raise StopIteration("Debug mode: Stopping after processing 3 query files.")
                i += 1
    
    with open(output_path, 'w') as f:
        if debug:
            for query_file, results_summary in detailed_output.items():
                print(f"Writing detailed results for {query_file}")
                print(f"\n{query_file}:\n")
                print(f"  Total Query Hashes: {results_summary['total_query_hashes']}\n")
                print(f"  Total Matches Found: {results_summary['total_matches']}\n")
                print("  Matching Songs:\n")
                for db_file, match_count in results_summary['matches'].items():
                    match_percentage = (
                        (match_count / results_summary['total_query_hashes']) * 100
                        if results_summary['total_query_hashes'] > 0 else 0
                    )
                    print(f"    {db_file}: {match_count} matches ({match_percentage:.2f}%)\n")
            return detailed_output
        else:
            for query_file, results in output.items():
                output_results = query_file
                for r in results[:3]:
                    output_results += '\t' + r
                f.write(f"{output_results}\n")
            return output

def main() -> None:
    """Run the command-line entry point for fingerprint building/querying.

    Supports two modes:
    1) Build fingerprints with ``--database`` and ``--fingerprints``.
    2) Run identification with ``--queryset``, ``--fingerprints``, and
       ``--output``.

    Args:
        None: Command-line arguments are parsed from ``sys.argv``.

    Returns:
        None: Executes side-effecting build/query steps and exits.
    """
    # Create the command-line interface for building fingerprints and running queries.
    parser = argparse.ArgumentParser(description="Audio identification pipeline")
    parser.add_argument("--database", required=False, help="Path to the database")
    parser.add_argument("--fingerprints", required=False, help="Path to fingerprints ONLY SUPPORTS .pkl or .json for debug")
    parser.add_argument("--queryset", required=False, help="Path to query set")
    parser.add_argument("--output", required=False, help="Path to output")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode, Will only run on the first file in the database and print out debug information")
    args = parser.parse_args() 
    
    if args.database and args.fingerprints:
        fingerprints_parent = os.path.dirname(args.fingerprints)
        if fingerprints_parent and not os.path.exists(fingerprints_parent):
            os.makedirs(fingerprints_parent)

        database_parent = os.path.dirname(args.database)
        if database_parent and not os.path.exists(args.database):
            os.makedirs(args.database)
            
        fingerprintBuilder(args.database, args.fingerprints, debug=args.debug)
    
    if args.queryset and args.fingerprints and args.output:
        queryset_parent = os.path.dirname(args.queryset)
        if queryset_parent and not os.path.exists(queryset_parent):
            os.makedirs(queryset_parent)
            
        fingerprints_parent = os.path.dirname(args.fingerprints)
        if fingerprints_parent and not os.path.exists(fingerprints_parent):
            raise ValueError(f"Fingerprints path {args.fingerprints} does not exist. Please run the fingerprint builder first.")
        
        audioIdentification(args.queryset, args.fingerprints, args.output, debug=args.debug)
        
    elif not (args.database and args.fingerprints) and not (args.queryset and args.fingerprints and args.output):
        raise ValueError("For audio identification, please provide --queryset, --fingerprints, and --output arguments. For fingerprint building, please provide --database and --fingerprints arguments.")

if __name__ == "__main__":
    main()