from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
import pickle

from cs336_basics.pretokenization_example import find_chunk_boundaries

train_path = "data\TinyStoriesV2-GPT4-train.txt"
test_path = "data\TinyStoriesV2-GPT4-valid.txt"

import regex as re

PAT = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)

def count_chunk(file_path: str, start: int, end: int) -> Counter[tuple[bytes, ...]]:
    with open(file_path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8")
    
    counts = Counter()
    
    for document in text.split("<|endoftext|>"):
        for match in PAT.finditer:
            word = match.group(0)
            pieces = tuple(bytes([b]) for b in word.encode("utf-8"))
            counts[pieces] += 1
    
    return counts

def get_pair_count(chunk_count:Counter[tuple[bytes, ...]]):
    pair_count:Counter[tuple[bytes, bytes]] = Counter()
    
    for word_chunk, count in chunk_count.items():
        for i in range(len(word_chunk) - 1):
            pair = (word_chunk[i], word_chunk[i+1])
            pair_count[pair] += count
    
    return pair_count

def get_max_pair_count(chunk_count:Counter[tuple[bytes, ...]], pool:ProcessPoolExecutor, max_workers:int=4):
    items = list[tuple[tuple[bytes, ...], int]](chunk_count.items())
    if len(items) == 0: return None
    chunk_size = (len(items) + max_workers - 1) // max_workers
    
    jobs = [Counter(dict(items[i:i+chunk_size])) for i in range(0, len(items), chunk_size)]
    
    total_pair_counts:Counter[tuple[bytes, bytes]] = Counter()
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        for pair_counts in pool.map(get_pair_count, jobs):
            total_pair_counts.update(pair_counts)
    
    return max(total_pair_counts, key=lambda pair:(total_pair_counts[pair], pair), default=None)

def merged_target_pair(chunk_count:Counter[tuple[bytes, ...]], target_pair:tuple[bytes, bytes]) -> Counter[tuple[bytes, bytes]]:
    new_chunk_count = Counter()
    for word, frequent in chunk_count.items():
        i = 0
        new_word = []
        while i < len(word) - 1:
            if (word[i],word[i+1]) == target_pair:
                new_word.append(word[i] + word[i+1])
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        
        if i == len(word) - 1: #handle the last if there's remain chr
            new_word.append(word[i])
        
        new_chunk_count[tuple(new_word)] += frequent
    
    return new_chunk_count

def train_tokenizer(input_path:str, vocab_size:int, special_tokens:list[str], num_processes:int=4
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, special_tokens)

    merged:list[tuple[bytes, bytes]] = []
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for token in special_tokens:
        vocab[len(vocab)] = token.encode()

    #pretokenizing - count chunk count
    jobs_chunk_count = [(input_path, start, end) for start, end in zip(boundaries[:-1], boundaries[1:])]
    current_chunk_counts = Counter()

    with ProcessPoolExecutor(max_workers=num_processes) as pool:
        for chunk_counts in pool.map(count_chunk, *zip(*jobs_chunk_count)):
            current_chunk_counts.update(chunk_counts)
    
    print(f"pre token num : {len(current_chunk_counts)}")

    with open("total_counts.pkl", "wb") as f:
        pickle.dump(current_chunk_counts, f)
    #end of pretokenizing
    
    #start of pair count
    while len(vocab) < vocab_size:
        new_chunk_count = Counter()
        
        max_frequent_pair = get_max_pair_count(current_chunk_counts, num_processes)
        if max_frequent_pair == None:
            break
        merged.append(max_frequent_pair)

        vocab[len(vocab)] = max_frequent_pair[0] + max_frequent_pair[1]
        chunk_items = list(current_chunk_counts.items())
        chunk_size = (len(chunk_items) + num_processes - 1) // num_processes
        jobs = [
            Counter(dict(chunk_items[i:i + chunk_size]))
            for i in range(0, len(chunk_items), chunk_size)
        ]
        with ProcessPoolExecutor(max_workers=num_processes) as pool:
            for chunk_count in pool.map(merged_target_pair, jobs, repeat(max_frequent_pair)):
                new_chunk_count.update(chunk_count)
        
        current_chunk_counts = new_chunk_count

    return (vocab, merged)
        
def main():
    file_path = "data/TinyStoriesV2-GPT4-train.txt"
    num_processes = 4
    result = train_tokenizer(file_path, vocab_size=10000, special_tokens=["<|endoftext|>"], num_processes=4)
    
    

if __name__ == "__main__":
    main()