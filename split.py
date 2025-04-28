import os
import json
import hashlib

data_dir = 'data/Stock_news'
PROMPTS_FILE = os.path.join(data_dir, 'filtered_prompts.jsonl')
PROCESSED_FILE = os.path.join(data_dir, 'processed_prompts.jsonl') # File with hashes of completed prompts

# New configuration
N_PARTS = 5  # Number of parts to split the remaining workload into
OUTPUT_DIR = os.path.join(data_dir, 'split_workload') # Directory to save the split files
OUTPUT_FILENAME_TEMPLATE = "remaining_prompts_part_{part_num}.jsonl"

# --- End Configuration ---

def calculate_prompt_hash(prompt_dict: dict) -> str:
    """Calculates the SHA256 hash of a prompt dictionary consistently."""
    # Ensure the hashing is identical to the original script's method
    # The original script added '_hash' *after* calculating it based on the original prompt.
    # So, we calculate the hash based on the prompt *without* the '_hash' key if it exists.
    prompt_copy = prompt_dict.copy()
    prompt_copy.pop('_hash', None) # Remove '_hash' key if present before hashing
    return hashlib.sha256(json.dumps(prompt_copy, sort_keys=True).encode()).hexdigest()

def split_remaining_workload(prompts_path: str, processed_hashes_path: str, output_dir: str, n_parts: int):
    """
    Identifies unprocessed prompts and splits them into n files.
    """
    # 1. Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # 2. Load processed prompt hashes
    processed_hashes = set()
    if os.path.isfile(processed_hashes_path):
        try:
            with open(processed_hashes_path, 'r', encoding='utf-8') as pf:
                for line in pf:
                    line = line.strip()
                    if line:
                        processed_hashes.add(line)
            print(f"Loaded {len(processed_hashes)} processed prompt hashes from {processed_hashes_path}")
        except Exception as e:
            print(f"Warning: Could not read processed hashes file {processed_hashes_path}: {e}")
            print("Assuming no prompts have been processed yet.")
    else:
        print(f"Processed hashes file not found ({processed_hashes_path}). Assuming no prompts processed yet.")

    # 3. Identify remaining prompts
    remaining_prompts_lines = []
    total_prompts_count = 0
    if not os.path.isfile(prompts_path):
        print(f"Error: Prompts file not found at {prompts_path}")
        return

    print(f"Reading prompts from {prompts_path} and identifying remaining work...")
    with open(prompts_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            total_prompts_count += 1
            try:
                prompt_dict = json.loads(line)
                prompt_hash = calculate_prompt_hash(prompt_dict)

                if prompt_hash not in processed_hashes:
                    remaining_prompts_lines.append(line) # Store the original line

            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON line in {prompts_path}: {e} -> '{line[:100]}...'")
            except Exception as e:
                 print(f"Error processing line: {e} -> '{line[:100]}...'")


    num_remaining = len(remaining_prompts_lines)
    print(f"Total prompts in source file: {total_prompts_count}")
    print(f"Number of prompts already processed: {len(processed_hashes)}")
    print(f"Number of prompts remaining: {num_remaining}")

    if num_remaining == 0:
        print("No remaining prompts to split.")
        return

    if n_parts <= 0:
        print("Error: Number of parts (n_parts) must be greater than 0.")
        return

    # 4. Calculate split sizes
    base_size = num_remaining // n_parts
    remainder = num_remaining % n_parts
    print(f"Splitting remaining {num_remaining} prompts into {n_parts} parts.")
    print(f"Base size per part: {base_size}, Remainder: {remainder}")

    # 5. Write split files
    start_index = 0
    for i in range(n_parts):
        part_num = i + 1
        part_size = base_size + (1 if i < remainder else 0)
        end_index = start_index + part_size

        output_filename = OUTPUT_FILENAME_TEMPLATE.format(part_num=part_num)
        output_filepath = os.path.join(output_dir, output_filename)

        part_lines = remaining_prompts_lines[start_index:end_index]

        if not part_lines:
            print(f"Part {part_num}: No prompts to write (this might happen if n_parts > num_remaining).")
            continue

        try:
            with open(output_filepath, 'w', encoding='utf-8') as outfile:
                for line in part_lines:
                    outfile.write(line + '\n')
            print(f"Written {len(part_lines)} prompts to {output_filepath}")
        except IOError as e:
            print(f"Error writing to file {output_filepath}: {e}")

        start_index = end_index # Move start index for the next part

    print("Workload splitting complete.")

if __name__ == '__main__':
    split_remaining_workload(
        prompts_path=PROMPTS_FILE,
        processed_hashes_path=PROCESSED_FILE,
        output_dir=OUTPUT_DIR,
        n_parts=N_PARTS
    )
