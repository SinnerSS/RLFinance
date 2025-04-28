import os
import json
import csv
import asyncio
import hashlib
from tqdm import tqdm
from openai import AsyncOpenAI

# File paths
data_dir = 'data/Stock_news/split_workload'
PROMPTS_FILE = os.path.join(data_dir, 'remaining_prompts_part_3.jsonl')
RESPONSES_FILE = os.path.join(data_dir, 'deepseek_responses_part_3.csv')
PROCESSED_FILE = os.path.join(data_dir, 'processed_prompts.jsonl')

# Batch & retry settings
BATCH_SIZE = 100        # prompts per batch
CONCURRENCY = 25        # simultaneous requests
environment = os.getenv('DEEPSEEK_API_KEY')
RETRY_LIMIT = 3
BACKOFF_BASE = 1       # seconds
INTER_BATCH_SLEEP = 2  # seconds between batches

# Deepseek client
client = AsyncOpenAI(
    api_key=environment,
    base_url="https://api.deepseek.com"
)
semaphore = asyncio.Semaphore(CONCURRENCY)

async def fetch_annotation(payload: dict) -> dict:
    """Fetch sentiment & type with retry/backoff."""
    async with semaphore:
        backoff = BACKOFF_BASE
        for attempt in range(1, RETRY_LIMIT + 1):
            try:
                response = await client.chat.completions.create(
                    model="deepseek-chat",
                    messages=payload['messages'],
                    temperature=1.0,
                    response_format={'type': 'json_object'}
                )
                data = json.loads(response.choices[0].message.content)
                return {
                    'date': payload.get('date'),
                    'stock_symbol': payload.get('stock_symbol'),
                    'sentiment_score': data.get('sentiment_score'),
                    'news_type': data.get('news_type'),
                    'prompt_hash': payload.get('_hash')
                }
            except Exception as e:
                if attempt == RETRY_LIMIT:
                    print(f"Failed after {RETRY_LIMIT} attempts for hash={payload.get('_hash')}: {e}")
                    return {
                        'date': payload.get('date'),
                        'stock_symbol': payload.get('stock_symbol'),
                        'sentiment_score': None,
                        'news_type': None,
                        'prompt_hash': payload.get('_hash')
                    }
                else:
                    print(f"Attempt {attempt} failed for hash={payload.get('_hash')}, retrying in {backoff}s…")
                    await asyncio.sleep(backoff)
                    backoff *= 2

async def process_file(prompts_path: str, responses_path: str, processed_path: str):
    # Ensure data directory exists
    os.makedirs(os.path.dirname(prompts_path), exist_ok=True)

    # Load all prompts
    prompts = []
    with open(prompts_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                prompt = json.loads(line)
                # compute a unique hash for each prompt
                h = hashlib.sha256(json.dumps(prompt, sort_keys=True).encode()).hexdigest()
                prompt['_hash'] = h
                prompts.append(prompt)
            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON line: {e}")

    # Load processed hashes
    processed = set()
    if os.path.isfile(processed_path):
        with open(processed_path, 'r', encoding='utf-8') as pf:
            for line in pf:
                line = line.strip()
                if line:
                    processed.add(line)

    # Prepare CSV
    is_new_csv = not os.path.isfile(responses_path) or os.path.getsize(responses_path) == 0
    csvfile = open(responses_path, 'a', newline='', encoding='utf-8')
    writer = csv.DictWriter(csvfile, fieldnames=['date', 'stock_symbol', 'sentiment_score', 'news_type', 'prompt_hash'])
    if is_new_csv:
        writer.writeheader()
        csvfile.flush()
        os.fsync(csvfile.fileno())

    # Open processed file for appending
    pf_append = open(processed_path, 'a', encoding='utf-8')

    try:
        for i in tqdm(range(0, len(prompts), BATCH_SIZE)):
            batch = [p for p in prompts[i:i + BATCH_SIZE] if p['_hash'] not in processed]
            if not batch:
                continue

            tasks = [fetch_annotation(p) for p in batch]
            results = await asyncio.gather(*tasks)

            for row in results:
                h = row['prompt_hash']
                if h in processed:
                    continue
                writer.writerow(row)
                csvfile.flush(); os.fsync(csvfile.fileno())

                # append hash to processed file
                pf_append.write(h + '\n')
                pf_append.flush(); os.fsync(pf_append.fileno())
                processed.add(h)

            print(f"Completed batch {i // BATCH_SIZE + 1} / {(len(prompts)-1)//BATCH_SIZE + 1}")
            await asyncio.sleep(INTER_BATCH_SLEEP)

    except KeyboardInterrupt:
        print("Interrupted by user, shutting down and saving progress.")
    finally:
        csvfile.close()
        pf_append.close()

if __name__ == '__main__':
    asyncio.run(process_file(PROMPTS_FILE, RESPONSES_FILE, PROCESSED_FILE))

