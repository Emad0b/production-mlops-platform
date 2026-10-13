"""Measure actual HTTP latency against an already-approved local model."""

import argparse, asyncio, json, time
from pathlib import Path
import httpx


async def run(url, requests, concurrency):
    semaphore = asyncio.Semaphore(concurrency)
    times = []
    errors = []
    payload = {
        "temperature": 82,
        "vibration": 4.2,
        "pressure": 5.3,
        "rpm": 1850,
        "operating_hours": 8200,
    }
    async with httpx.AsyncClient(timeout=30) as client:

        async def one():
            async with semaphore:
                start = time.perf_counter()
                try:
                    result = await client.post(url + "/api/predict", json=payload)
                    result.raise_for_status()
                    times.append((time.perf_counter() - start) * 1000)
                except Exception as error:
                    errors.append(str(error))

        await one()
        times.clear()
        errors.clear()
        start = time.perf_counter()
        await asyncio.gather(*(one() for _ in range(requests)))
        elapsed = time.perf_counter() - start
    values = sorted(times)
    report = {
        "requests": requests,
        "concurrency": concurrency,
        "successful": len(times),
        "failed": len(errors),
        "p50_http_ms": values[int((len(values) - 1) * 0.5)] if values else None,
        "p95_http_ms": values[int((len(values) - 1) * 0.95)] if values else None,
        "throughput_rps": len(times) / elapsed,
        "elapsed_seconds": elapsed,
        "errors": errors[:3],
    }
    print(json.dumps(report, indent=2))
    Path("reports").mkdir(exist_ok=True)
    Path("reports/load-test.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8200")
    p.add_argument("--requests", type=int, default=200)
    p.add_argument("--concurrency", type=int, default=8)
    args = p.parse_args()
    if not 1 <= args.requests <= 10000 or not 1 <= args.concurrency <= 64:
        p.error("Use 1–10,000 requests and 1–64 concurrency")
    asyncio.run(run(args.url, args.requests, args.concurrency))
