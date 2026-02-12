import argparse
import gc
import os
import uuid
from typing import Any, Dict, Iterator, List

import datasets
import numpy as np
import pandas as pd
from tqdm import tqdm

from olmo.config import TrainConfig
from olmo.tokenizer import Tokenizer


def tokenize_batch(texts: List[str], tokenizer: Tokenizer, max_length: int = 1024) -> List[List[int]]:
    """Tokenize a batch of texts and return token lists."""
    tokenized_batch = []

    for text in texts:
        # Tokenize the text
        tokens = tokenizer.encode(text, add_special_tokens=True)

        # Truncate if necessary
        if len(tokens) > max_length:
            tokens = tokens[:max_length]

        tokenized_batch.append(tokens)

    return tokenized_batch


def save_part(
    concatenated_tokens: np.ndarray,
    metadata: List[Dict],
    output_dir: str,
    part_num: int,
):
    """Save a part of tokenized data to .npy and .csv.gz files.

    Args:
        concatenated_tokens: 1D array of concatenated tokens in uint16 format
        metadata: List of dicts with start, end, document_id, source_file, original_length
        output_dir: Directory to save files
        part_num: Part number for filename
    """

    if len(concatenated_tokens) == 0:
        return

    # Save .npy file as 1D concatenated array
    npy_path = os.path.join(output_dir, f"part-{part_num:05d}.npy")
    concatenated_tokens.tofile(npy_path)  # Save as raw binary file like the standard format

    # Create metadata DataFrame with proper column names (no header in standard format)
    df_metadata = pd.DataFrame(metadata)

    # Save .csv.gz file without header (matching standard format)
    csv_path = os.path.join(output_dir, f"part-{part_num:05d}.csv.gz")
    df_metadata.to_csv(csv_path, index=False, header=False, compression="gzip")

    print(
        f"Saved part {part_num}: {len(metadata):,} sequences, {concatenated_tokens.nbytes / 1024**2:.1f} MB, {len(concatenated_tokens):,} tokens"
    )


def tokenize_dataset(
    dataset_name: str,
    dataset_config: str,
    split: str,
    output_dir: str,
    tokenizer: Tokenizer,
    batch_size: int = 1000,
    max_length: int = 1024,
    max_tokens: int = 10_000_000_000,  # 10BT (10 billion tokens)
    max_tokens_per_part: int = 50_000_000,  # ~50M tokens per part file for memory efficiency
    streaming: bool = True,
    text_field: str = "text",
):
    """
    Tokenize data from datasets.load_dataset() and save as concatenated .npy and .csv.gz files.

    Args:
        dataset_name: Name of the dataset (e.g., "allenai/c4")
        dataset_config: Dataset configuration (e.g., "en")
        split: Dataset split (e.g., "train")
        output_dir: Directory to save output files
        tokenizer: OLMo tokenizer instance
        batch_size: Number of texts to process at once
        max_length: Maximum sequence length
        max_tokens: Maximum total tokens to process (roughly 10BT)
        max_tokens_per_part: Maximum tokens per part file (for memory management)
        streaming: Whether to use streaming (recommended for large datasets)
        text_field: Field name containing the text data
    """

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    print(f"Loading dataset: {dataset_name}")
    print(f"Config: {dataset_config}, Split: {split}, Streaming: {streaming}")

    dataset = datasets.load_dataset(dataset_name, dataset_config, split=split, streaming=streaming)

    print(f"Dataset loaded successfully!")
    if not streaming:
        print(f"Dataset size: {len(dataset):,} examples")

    # Initialize counters
    total_tokens = 0
    total_sequences = 0
    part_num = 0
    current_part_tokens = 0

    # Storage for current part - use lists for efficiency
    current_concatenated_tokens = []
    current_metadata = []
    current_start_idx = 0

    # Progress bar
    pbar = tqdm(desc="Tokenizing", unit="sequences")

    try:
        # Process data in batches
        current_batch = []

        for item in dataset:
            # Extract text
            text = item.get(text_field, "")
            if not text:
                continue

            current_batch.append(text)

            # Process batch when full
            if len(current_batch) >= batch_size:
                tokenized_batch = tokenize_batch(current_batch, tokenizer, max_length)

                for i, tokens in enumerate(tokenized_batch):
                    if total_tokens >= max_tokens:
                        print(f"\nReached maximum token limit of {max_tokens:,}")
                        break

                    # Calculate start and end indices for this sequence
                    start_idx = current_start_idx + len(current_concatenated_tokens)
                    end_idx = start_idx + len(tokens)

                    # Generate document ID (UUID-like format to match standard)
                    document_id = f"<urn:uuid:{uuid.uuid4()}>"

                    # Add tokens to concatenated sequence
                    current_concatenated_tokens.extend(tokens)

                    # Add metadata
                    current_metadata.append(
                        {
                            "start": start_idx,
                            "end": end_idx,
                            "document_id": document_id,
                            "source_file": f"{dataset_name}/{dataset_config}",
                            "original_length": len(current_batch[i]),
                        }
                    )

                    total_tokens += len(tokens)
                    total_sequences += 1
                    current_part_tokens += len(tokens)
                    pbar.update(1)
                    pbar.set_postfix(
                        {
                            "total_tokens": f"{total_tokens:,}",
                            "part": part_num,
                            "part_tokens": f"{current_part_tokens:,}",
                        }
                    )

                    # Save part when it reaches the token limit
                    if current_part_tokens >= max_tokens_per_part:
                        # Convert to numpy array with uint16 dtype
                        concatenated_array = np.array(current_concatenated_tokens, dtype=np.uint16)
                        save_part(concatenated_array, current_metadata, output_dir, part_num)

                        # Reset for next part
                        current_concatenated_tokens = []
                        current_metadata = []
                        current_start_idx = 0  # Reset start index for new part
                        current_part_tokens = 0
                        part_num += 1
                        gc.collect()  # Force garbage collection

                current_batch = []

                if total_tokens >= max_tokens:
                    break

        # Process remaining batch
        if current_batch and total_tokens < max_tokens:
            tokenized_batch = tokenize_batch(current_batch, tokenizer, max_length)

            for i, tokens in enumerate(tokenized_batch):
                if total_tokens >= max_tokens:
                    break

                # Calculate start and end indices
                start_idx = current_start_idx + len(current_concatenated_tokens)
                end_idx = start_idx + len(tokens)

                # Generate document ID
                document_id = f"<urn:uuid:{uuid.uuid4()}>"

                # Add tokens and metadata
                current_concatenated_tokens.extend(tokens)
                current_metadata.append(
                    {
                        "start": start_idx,
                        "end": end_idx,
                        "document_id": document_id,
                        "source_file": f"{dataset_name}/{dataset_config}",
                        "original_length": len(current_batch[i]),
                    }
                )

                total_tokens += len(tokens)
                total_sequences += 1
                current_part_tokens += len(tokens)
                pbar.update(1)

        # Save final part if there's data
        if current_concatenated_tokens:
            concatenated_array = np.array(current_concatenated_tokens, dtype=np.uint16)
            save_part(concatenated_array, current_metadata, output_dir, part_num)

    finally:
        pbar.close()

    print(f"\nTokenization complete!")
    print(f"Total sequences processed: {total_sequences:,}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Number of parts created: {part_num + 1}")
    print(f"Output directory: {output_dir}")


def load_tokenized_data(npy_path: str, csv_path: str = None):
    """
    Load tokenized data from .npy file and optionally metadata from .csv.gz file.

    Args:
        npy_path: Path to .npy file (concatenated 1D array)
        csv_path: Optional path to .csv.gz metadata file

    Returns:
        tuple: (token_data, metadata_df) if csv_path provided, else just token_data
    """
    # Load token data as 1D concatenated array (standard format uses fromfile)
    token_data = np.fromfile(npy_path, dtype=np.uint16)

    if csv_path:
        # Load metadata (no header in standard format)
        metadata_df = pd.read_csv(
            csv_path,
            compression="gzip",
            header=None,
            names=["start", "end", "document_id", "source_file", "original_length"],
        )
        return token_data, metadata_df

    return token_data


def get_sequence_from_concatenated(token_data: np.ndarray, start: int, end: int) -> np.ndarray:
    """
    Extract a specific sequence from concatenated token data.

    Args:
        token_data: 1D concatenated token array
        start: Start index of sequence
        end: End index of sequence

    Returns:
        Sequence tokens as numpy array
    """
    return token_data[start:end]


def main():
    parser = argparse.ArgumentParser(description="Tokenize data from datasets.load_dataset() and save as .npy files")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="allenai/c4",
        help="Dataset name (default: allenai/c4)",
    )
    parser.add_argument("--dataset_config", type=str, default="en", help="Dataset config (default: en)")
    parser.add_argument("--split", type=str, default="train", help="Dataset split (default: train)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size for processing")
    parser.add_argument("--max_length", type=int, default=256000, help="Maximum sequence length")
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=10_000_000_000,
        help="Maximum total tokens (default: 10B)",
    )
    parser.add_argument(
        "--max_tokens_per_part",
        type=int,
        default=50_000_000,
        help="Maximum tokens per part file (default: 50M)",
    )
    parser.add_argument(
        "--no_streaming",
        action="store_true",
        help="Disable streaming (not recommended for large datasets)",
    )
    parser.add_argument(
        "--text_field",
        type=str,
        default="text",
        help="Field name containing text data (default: text)",
    )
    parser.add_argument(
        "--tokenizer_identifier",
        type=str,
        default="gpt2",
        help="Tokenizer identifier (default: gpt2)",
    )

    args = parser.parse_args()

    # Load configuration and tokenizer
    print("Loading configuration and tokenizer...")
    tokenizer = Tokenizer.from_pretrained(args.tokenizer_identifier)

    print(f"Dataset: {args.dataset_name}/{args.dataset_config}")
    print(f"Split: {args.split}")
    print(f"Output directory: {args.output_dir}")
    print(f"Max tokens target: {args.max_tokens:,}")
    print(f"Streaming: {not args.no_streaming}")

    # Start tokenization
    tokenize_dataset(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.split,
        output_dir=args.output_dir,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        max_tokens=args.max_tokens,
        max_tokens_per_part=args.max_tokens_per_part,
        streaming=not args.no_streaming,
        text_field=args.text_field,
    )


if __name__ == "__main__":
    main()
