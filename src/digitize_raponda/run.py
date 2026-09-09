# src/run_async.py

import json
import asyncio
from pathlib import Path
from provider import LLMProvider
from tqdm.asyncio import tqdm

PAGES_DIR = Path("data/intermediate/pages")
OUTPUT_DIR = Path("data/intermediate/json")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ERROR_LOG = Path("data/intermediate/error_log.txt")

BATCH_SIZE = 50


def log_error(msg: str):
    with open(ERROR_LOG, "a") as f:
        f.write(msg + "\n")


async def process_pdf(provider: LLMProvider, pdf_path: Path, existing_jsons: set):
    page_id = pdf_path.stem

    if page_id in existing_jsons:
        return f"skip:{page_id}"

    try:
        result = await provider.digitize_pdf(pdf_path)

        if result is None:
            log_error(f"{page_id}: digitization failed")
            return f"fail:{page_id}"

        out_file = OUTPUT_DIR / f"{page_id}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        return f"ok:{page_id}"

    except Exception as e:
        log_error(f"{page_id}: {str(e)}")
        return f"error:{page_id}"


def chunked(iterable, size):
    for i in range(0, len(iterable), size):
        yield iterable[i : i + size]


async def main_async():
    provider = LLMProvider(model="gpt-5-mini", temperature=0.0)

    pdf_files = sorted(PAGES_DIR.glob("page*.pdf"))
    json_files = {p.stem for p in OUTPUT_DIR.glob("page*.json")}

    print(f"Found {len(pdf_files)} PDFs")
    print(f"Found {len(json_files)} existing JSONs")

    batches = list(chunked(pdf_files, BATCH_SIZE))

    for batch_idx, batch in enumerate(batches, start=1):
        print(f"\n🚀 Processing batch {batch_idx}/{len(batches)} "
              f"({len(batch)} pages)")

        tasks = [
            process_pdf(provider, pdf_path, json_files)
            for pdf_path in batch
        ]

        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            await coro

    print("\n🎉 All batches completed.")


if __name__ == "__main__":
    asyncio.run(main_async())
