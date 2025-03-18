from pathlib import Path
from typing import List, Dict
import pandas as pd

def load_price_data(tics: List[str], data_dir: Path) -> Dict[str, pd.DataFrame]:
    open_dict = {}
    adj_close_dict = {}

    for tic in tics:
        file_path = data_dir / f'{tic}.csv'
        df = pd.read_csv(file_path)
        
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)
        
        if 'adj close' in df.columns:
            df.rename(columns={'adj close': 'adj_close'}, inplace=True)
        
        open_dict[tic] = df['open']
        adj_close_dict[tic] = df['adj_close']
    
    open_df = pd.DataFrame(open_dict)
    adj_close_df = pd.DataFrame(adj_close_dict)
    
    return {
        'open': open_df,
        'adj_close': adj_close_df
    }
