import pandas as pd
from pathlib import Path

def load_price_data(pool: pd.DataFrame, data_dir: Path):

    tics = pool['tic'].tolist()
    
    stock_list = []
    for tic in tics:
        file_path = data_dir / f'{tic}.csv'
        stock_price = pd.read_csv(file_path)
        stock_price['tic'] = tic
        stock_list.append(stock_price)

    price_data = pd.concat(stock_list, ignore_index=True)

    return price_data
