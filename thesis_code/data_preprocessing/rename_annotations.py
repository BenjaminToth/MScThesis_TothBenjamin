#!/usr/bin/env python3
"""
CSV Annotation Renaming Tool
=============================
Batch find-and-replace tool for renaming patterns in CSV annotation files.

Configuration
-------------
INPUT_FOLDER : Path
    Source folder containing CSV files to process (searched recursively)
OUTPUT_FOLDER : Path
    Destination folder where modified CSVs will be saved
SEARCH_STRING : str
    Text to find in each CSV file
REPLACE_STRING : str
    Text to replace with
"""
from pathlib import Path
from tqdm import tqdm

INPUT_FOLDER = Path("./data/raw/annotations/Chris/")
OUTPUT_FOLDER = Path("./data/preprocessed/annotations/Chris/")
SEARCH_STRING = ",seiz,"
REPLACE_STRING = ",Chris,"

ENCODING = "utf-8"

def main() -> None:
    """
    Process CSV files recursively, finding and replacing text patterns.
    
    Processes all CSV files in subdirectories of INPUT_FOLDER (excluding root-level files).
    For each file, replaces SEARCH_STRING with REPLACE_STRING and saves to OUTPUT_FOLDER
    with the same directory structure preserved.
    """
    if not INPUT_FOLDER.exists() or not INPUT_FOLDER.is_dir():
        raise SystemExit(f"Input folder does not exist or is not a directory: {INPUT_FOLDER}")

    total_files_processed = 0
    files_with_changes = 0
    total_replacement_count = 0

    for source_file in tqdm(INPUT_FOLDER.rglob("*.csv")):
        if source_file.parent == INPUT_FOLDER:
            continue

        relative_path = source_file.relative_to(INPUT_FOLDER)
        destination_file = OUTPUT_FOLDER / relative_path
        destination_file.parent.mkdir(parents=True, exist_ok=True)

        replacement_count = 0
        file_was_changed = False

        with source_file.open("r", encoding=ENCODING, errors="replace", newline="") as input_file, \
             destination_file.open("w", encoding=ENCODING, newline="") as output_file:
            for line in input_file:
                if SEARCH_STRING in line:
                    count_in_line = line.count(SEARCH_STRING)
                    replacement_count += count_in_line
                    file_was_changed = True
                    line = line.replace(SEARCH_STRING, REPLACE_STRING)
                output_file.write(line)

        total_files_processed += 1
        total_replacement_count += replacement_count
        if file_was_changed:
            files_with_changes += 1

    print("\nDone.")
    print(f"CSV files processed:  {total_files_processed}")
    print(f"Files changed:        {files_with_changes}")
    print(f"Total replacements:   {total_replacement_count}")
    print(f"Output folder:        {OUTPUT_FOLDER}")

if __name__ == "__main__":
    main()
