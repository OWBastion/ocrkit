from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    labeled = Path("datasets/labeled/samples.jsonl")
    if not labeled.exists():
        print("no labeled samples found at datasets/labeled/samples.jsonl")
        return

    total = 0
    with labeled.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            json.loads(line)
            total += 1

    print(json.dumps({"total_samples": total}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
