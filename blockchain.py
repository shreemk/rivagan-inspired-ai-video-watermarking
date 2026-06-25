#blockchain.py

import hashlib, json, time

def generate_blockchain_record(video_path):
    with open(video_path, "rb") as f:
        hash_val = hashlib.sha256(f.read()).hexdigest()

    record = {
        "video": video_path,
        "hash": hash_val,
        "algorithm": "SHA-256",
        "timestamp": time.time()
    }

    with open("blockchain_ledger.json", "a") as f:
        f.write(json.dumps(record) + "\n")

    return hash_val
