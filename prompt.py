import os
import pandas as pd
import tiktoken
import json
from tqdm import tqdm

from config import Config

# Configuration
INPUT_CSV = 'data/Stock_news/nasdaq_exteral_data.csv'
OUTPUT_PROMPTS = 'data/Stock_news/llm_feature_prompts.jsonl'
SUMMARY_FIELDS = ['Lexrank_summary', 'Luhn_summary', 'Lsa_summary']
CHUNK_SIZE = 10000       # rows per chunk
MODEL_NAME = 'gpt-4'     # for token counting

SYSTEM_CONTENT = (
    "You are an expert financial news annotator.\n"
    "Analyze the following news headline or summary for the provided stock symbol and output a JSON object with the following keys:\n"
    "- sentiment_score: number between -5 (very negative) and +5 (very positive), with 0 as neutral.\n"
    "- news_type: one of ['earnings', 'merger_acquisition', 'IPO', 'dividends', 'regulation', 'macro', 'monetary_policy', 'analyst_rating', 'management_change', 'legal', 'ESG', 'product_strategy', 'other'].\n"
    "Only output valid JSON."
)

# Helper: choose best summary or title
NEWS_TYPES = [
    'earnings', 'merger_acquisition', 'IPO', 'dividends',
    'regulation', 'macro', 'monetary_policy', 'analyst_rating',
    'management_change', 'legal', 'ESG', 'product_strategy', 'other'
]

# Select best text from row
def choose_text(row: pd.Series) -> str:
    for field in SUMMARY_FIELDS:
        txt = row.get(field, '')
        if isinstance(txt, str) and txt.strip():
            return txt.strip()
        try:
            txt = row.get('Article_title', '').strip()
        except Exception as e:
            print(e)
    return txt 
# Main streaming function
def main():
    config = Config()
    enc = tiktoken.encoding_for_model(MODEL_NAME)
    total_prompts = 0
    token_sum = 0
    token_min = float('inf')
    token_max = 0

    os.makedirs(os.path.dirname(OUTPUT_PROMPTS), exist_ok=True)
    with open(OUTPUT_PROMPTS, 'w') as f_out:
        for chunk in tqdm(pd.read_csv(INPUT_CSV, chunksize=CHUNK_SIZE)):
            chunk['Date'] = pd.to_datetime(chunk['Date'])
            chunk = chunk[(chunk['Date'] >= config.start_train) & (chunk['Date'] <= config.end_test)].copy()
            if chunk.empty:
                continue
            for _, row in chunk.iterrows():
                text = choose_text(row)
                if not text:
                    continue
                messages = [
                    {'role': 'system', 'content': SYSTEM_CONTENT},
                    {'role': 'user', 'content': f"Text: '{text}'\nStock: {row.get('Stock_symbol')}"}
                ]
                date_val = row['Date']
                if isinstance(date_val, pd.Timestamp):
                    date_str = date_val.strftime('%Y-%m-%d')
                else:
                    date_str = str(date_val)
                record = {
                    'date': date_str,
                    'stock_symbol': row.get('Stock_symbol'),
                    'messages': messages
                }
                f_out.write(json.dumps(record) + '\n')

                count = len(enc.encode(SYSTEM_CONTENT)) + len(enc.encode(text))
                total_prompts += 1
                token_sum += count
                token_min = min(token_min, count)
                token_max = max(token_max, count)

    if total_prompts == 0:
        print("No prompts generated.")
    else:
        avg_tokens = token_sum / total_prompts
        print(f"Feature prompts saved to {OUTPUT_PROMPTS}")
        print(f"Total prompts: {total_prompts}")
        print(f"Average tokens per prompt: {avg_tokens:.2f}")
        print(f"Min tokens: {token_min}")
        print(f"Max tokens: {token_max}")

if __name__ == '__main__':
    main()
