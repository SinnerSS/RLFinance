import pandas as pd
from pathlib import Path

def load_price_data(pool: pd.DataFrame, data_dir: Path):

    stock_symbols = pool['Stock_symbol'].tolist()
    
    stock_list = []
    for symbol in stock_symbols:
        file_path = data_dir / f'{symbol}.csv'
        stock_price = pd.read_csv(file_path)
        stock_price['Stock_symbol'] = symbol
        stock_list.append(stock_price)

    price_data = pd.concat(stock_list, ignore_index=True)

    return price_data
