from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import pickle


from cs336_basics.pretokenization_example import find_chunk_boundaries

def count_chunk(file_path: str, start: int, end: int) -> None:
    with open(file_path, "rb") as f:
        f.seek(start)
        text = f.read(end - start)
    
    counts = Counter()
    
    for document in text.split("<|endoftext|>"):
        for word in document.split():
            pieces = tuple(bytes[b] for b in word.encode("utf-8"))
            counts += 1
    
    return counts

def main():
    file_path = "data/TinyStoriesV2-GPT4-train.txt"
    num_processes = 4
    
    with open(file_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
    
    jobs = [(file_path, start, end) for start, end in zip(boundaries[:-1], boundaries[1:])]
    
    total_counts = Counter()
    
    with ProcessPoolExecutor(max_workers=num_processes) as pool:
        for chunk_counts in pool.map(count_chunk, jobs):
            total_counts.update(chunk_counts)
    
    print(f"pre token num : {len(total_counts)}")

    with open("total_counts.pkl", "wb") as f:
        pickle.dump(total_counts, f)

if __name__ == "__main__":
    main()