import pickle
import pandas as pd
from pathlib import Path
from typing import List

def process_pool(df: pd.DataFrame, output_path: Path) -> List[str]:
    """
    Processes the pool DataFrame and saves the 'Stock_symbol' column as a list to a pickle file.

    Args:
        df (pd.DataFrame): The input DataFrame.
        output_path (Path): The path to save the pickle file.
    """
    stock_symbols = df['Stock_symbol'].astype(str).tolist()

    with open(output_path, 'wb') as f:
        pickle.dump(stock_symbols, f)

    return stock_symbols
