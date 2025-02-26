import pickle
import pandas as pd
from typing import List
from pathlib import Path

def load_stock_pool(path: Path) -> List[str]:
    if path.suffix == '.pkl':
        with open(path, 'rb') as file:
            symbols = pickle.load(file)
        return symbols
    elif path.suffix == '.csv':
        df = pd.read_csv(path)
        symbols = df['Stock_symbol'].as_type(str).to_list()
        return symbols

    return []
