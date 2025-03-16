import pandas as pd
from tqdm import tqdm
from pathlib import Path
from collections import Counter

from config import Config

def news_by_stock(news: pd.DataFrame) -> Counter:

    all_counts = Counter()
    
    counts = news['Stock_symbol'].value_counts().to_dict()
    all_counts.update(counts)

    return all_counts


def capitalize_files(data_dir: Path):
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

def main():
    cf = Config()

    post_path = cf.data_path / 'post'
    price_path = cf.data_path / 'Stock_price/full_history'
    news_path = cf.data_path / 'Stock_news/nasdaq_exteral_data.csv'

    all_counts = Counter()
    for chunk in tqdm(pd.read_csv(news_path, chunksize=10000), desc='Counting news by stock'):
        all_counts.update(news_by_stock(chunk))

    news_counts = pd.DataFrame(all_counts.items(), columns=['tic', 'count'])
    news_counts.to_csv(post_path / 'news_counts.csv', index=False)

    capitalize_files(price_path)

if __name__ == '__main__':
    main()
