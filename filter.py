import os
import json
import tiktoken

# Configuration
INPUT_PROMPTS = 'data/Stock_news/llm_feature_prompts.jsonl'
OUTPUT_FILTERED = 'data/Stock_news/filtered_prompts.jsonl'
FILTER_SYMBOLS = [
    'AAPL','ADBE','ADI','ADP','ADSK','AEP','ALGN','AMAT','AMD',
    'AMGN','AMZN','ANSS','ASML','AVGO','AZN','BIIB','BKNG',
    'BKR','CDNS','CHTR','CMCSA','COST','CPRT','CSCO','CSGP',
    'CSX','CTAS','CTSH','DLTR','DXCM','EA','EBAY','ENPH','EXC',
    'FANG','FAST','FTNT','GILD','GOLD','GOOG','GOOGL','HON',
    'IDXX','ILMN','INTC','INTU','ISRG','JD','KDP','KHC','KLAC',
    'LRCX','LULU','MAR','MCHP','MDLZ','MELI','META','MNST',
    'MRVL','MSFT','MU','NFLX','NVDA','NXPI','ODFL','ON','ORLY',
    'PANW','PAYX','PCAR','PEP','PYPL','QCOM','REGN','ROST',
    'SBUX','SIRI','SNPS','TEAM','TMUS','TSLA','TXN','VRSK',
    'VRTX','WBA','WBD','WDAY','XEL'
]
MODEL_NAME = 'gpt-4'

# Function: filter prompts by symbol
def filter_prompts():
    os.makedirs(os.path.dirname(OUTPUT_FILTERED), exist_ok=True)
    total_in = total_out = 0
    with open(INPUT_PROMPTS, 'r') as fin, open(OUTPUT_FILTERED, 'w') as fout:
        for line in fin:
            total_in += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            symbol = str(rec.get('stock_symbol','')).strip().upper()
            if symbol in FILTER_SYMBOLS:
                fout.write(json.dumps(rec) + '\n')
                total_out += 1
    print(f"Filtered {total_out}/{total_in} prompts into {OUTPUT_FILTERED}")
    return total_out

# Function: compute token statistics on filtered prompts
def stats_filtered():
    enc = tiktoken.encoding_for_model(MODEL_NAME)
    counts = []
    with open(OUTPUT_FILTERED, 'r') as f:
        for line in f:
            rec = json.loads(line)
            system, user = rec.get('messages', '')
            counts.append(len(enc.encode(system.get('content'))) + len(enc.encode(user.get('content'))))
    if not counts:
        print("No filtered prompts to analyze.")
        return
    total = len(counts)
    avg = sum(counts) / total
    mn = min(counts)
    mx = max(counts)
    sorted_counts = sorted(counts)
    def pct(p):
        idx = int(p * total)
        idx = min(idx, total-1)
        return sorted_counts[idx]
    print(f"Total filtered prompts: {total}")
    print(f"Average tokens: {avg:.2f}")
    print(f"Min tokens: {mn}")
    print(f"Max tokens: {mx}")
    print(f"25th percentile: {pct(0.25)}")
    print(f"50th percentile: {pct(0.50)}")
    print(f"75th percentile: {pct(0.75)}")
    print(f"90th percentile: {pct(0.90)}")
    print(f"95th percentile: {pct(0.95)}")

if __name__ == '__main__':
    filter_prompts()
    stats_filtered()
