import pandas as pd
from typing import List
from pathlib import Path


def process_stock_price(pool: List[str], data_dir: Path, output_path: Path):
    stock_list = []
    for symbol in pool:
        try:
            file_path = data_dir / f'{symbol}.csv'
            stock_price = pd.read_csv(file_path)
            stock_price['Stock_symbol'] = symbol
            stock_list.append(stock_price)
        except (FileNotFoundError, pd.errors.EmptyDataError):
            print(f"Could not find or read file for symbol: {symbol}")

    price_data = pd.concat(stock_list, ignore_index=True)
    price_data.to_parquet(output_path)
    return price_data

def capitalize_files(data_dir: Path):
    """
    Capitalize all filenames in the given directory using only pathlib.
    
    Args:
        data_dir (Path): Path to the directory containing files to be capitalized.
    
    Returns:
        int: Number of files capitalized.
    """
    count = 0
    for file_path in data_dir.iterdir():
        if file_path.is_file():
            stem = file_path.stem.upper()
            suffix = file_path.suffix
            
            new_filename = stem + suffix
            new_file_path = data_dir / new_filename
            
            file_path.rename(new_file_path)
            count += 1
    return count
