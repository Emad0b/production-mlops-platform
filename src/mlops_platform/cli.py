import argparse, json
from .data.ingest import simulate, save_dataset
from .training.train import train
from .registry.promote import promote, rollback
from .storage import models, db, sql, audit, state
from .registry.canary import start_canary
from .pipelines.retraining import inspect_batch
from .monitoring.drift import monitor


def main():
    parser = argparse.ArgumentParser(prog="mlops_platform")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rows", type=int, default=4000)
    p.add_argument("--shifted", action="store_true")
    p = sub.add_parser("train")
    p.add_argument("dataset_id")
    p = sub.add_parser("promote")
    p.add_argument("model_id")
    p.add_argument("--approve", action="store_true")
    p = sub.add_parser("rollback")
    p.add_argument("--approve", action="store_true")
    sub.add_parser("models")
    p = sub.add_parser("monitor")
    p.add_argument("reference_id")
    p.add_argument("current_id")
    p = sub.add_parser("approve-data")
    p.add_argument("dataset_id")
    p.add_argument("--approve", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        result = save_dataset(
            simulate(args.rows, args.seed, args.shifted),
            {
                "generator": "simulator-v1",
                "seed": args.seed,
                "shifted": args.shifted,
                "rows": args.rows,
            },
        )
    elif args.command == "train":
        result = train(args.dataset_id)
    elif args.command == "promote":
        result = (
            start_canary(args.model_id, approved=args.approve)
            if state("champion")
            else promote(args.model_id, approved=args.approve)
        )
    elif args.command == "rollback":
        result = rollback(args.approve)
    elif args.command == "models":
        result = models()
    elif args.command == "monitor":
        result = inspect_batch(args.reference_id, args.current_id)
    elif args.command == "approve-data":
        if not args.approve:
            parser.error("--approve is required")
        from .data.ingest import load_dataset

        load_dataset(args.dataset_id)
        with db() as conn:
            sql(
                conn,
                "UPDATE state SET value=? WHERE key=?",
                (args.dataset_id, "approved_dataset"),
            )
        audit("dataset_approved", {"dataset": args.dataset_id})
        result = {"approved": args.dataset_id}
    print(json.dumps(result, indent=2))
    if isinstance(result, dict) and result.get("passed") is False:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
