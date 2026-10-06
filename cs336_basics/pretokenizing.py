from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import pickle
from cs336_basics.pretokenization_example import find_chunk_boundaries

train_path = "data\TinyStoriesV2-GPT4-train.txt"
test_path = "data\TinyStoriesV2-GPT4-valid.txt"

def count_chunk(file_path: str, start: int, end: int) -> Counter[tuple[bytes, ...]]:
    with open(file_path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8")
    
    counts = Counter()
    
    for document in text.split("<|endoftext|>"):
        for word in document.split():
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

def get_max_pair_count(chunk_count:Counter[tuple[bytes, ...]], max_workers:int=4):
    items = list[tuple[bytes, ...]](chunk_count)
    chunk_size = (len(items) + max_workers - 1) // max_workers
    
    jobs = [Counter(dict(items[i:i+chunk_size]) for i in range(0, len(items), chunk_size))]
    
    total_pair_counts:Counter[tuple[bytes, bytes]] = Counter()
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        for pair_counts in pool.map(get_pair_count, jobs):
            total_pair_counts.update(pair_counts)
    
    return max(total_pair_counts, key=lambda pair:(total_pair_counts[pair], pair), default=None) #????
            
def main():
    file_path = "data/TinyStoriesV2-GPT4-train.txt"
    num_processes = 4
    result = train_tokenizer(file_path, vocab_size=10000, special_tokens=["<|endoftext|>"], num_processes=4)

def train_tokenizer(input_path:str, vocab_size:int, special_tokens:list[str], num_processes:int=4
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
    
    #pretokenizing - count chunk count
    jobs_chunk_count = [(input_path, start, end) for start, end in zip(boundaries[:-1], boundaries[1:])]
    total_counts = Counter()
    
    with ProcessPoolExecutor(max_workers=num_processes) as pool:
        for chunk_counts in pool.map(count_chunk, jobs_chunk_count):
            total_counts.update(chunk_counts)
    
    print(f"pre token num : {len(total_counts)}")

    with open("total_counts.pkl", "wb") as f:
        pickle.dump(total_counts, f)
    #end of pretokenizing
    
    #start of pair count
    iter_num = 10
    for i in range(iter_num):
        
    
    
    

if __name__ == "__main__":
    main()