"""Score one customer from the command line.

    python predict.py examples/sample_customer.json
    python predict.py --json '{"gender": "Female", ...}'
    cat customer.json | python predict.py -

Prints the prediction as JSON on stdout. Exit code 2 means the payload was
invalid or unreadable, 1 means the model could not be loaded or run.
"""
import argparse
import json
import logging
import sys
from pathlib import Path

from pydantic import ValidationError

from churn.config import MODEL_DIR
from churn.inference import ChurnPredictor


def read_payload(args):
    if args.json:
        return json.loads(args.json)
    raw = sys.stdin.buffer.read() if args.payload == "-" else Path(args.payload).read_bytes()
    # given bytes, json works out UTF-8 / UTF-8 with BOM / UTF-16, which is what
    # Notepad and PowerShell ("> file.json", piped Get-Content) produce on Windows
    return json.loads(raw)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("payload", nargs="?", help="path to a JSON file, or '-' for stdin")
    parser.add_argument("--json", help="the JSON payload as a string")
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    args = parser.parse_args(argv)
    if not args.payload and not args.json:
        parser.error("give a JSON file, '-' for stdin, or --json '<payload>'")

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    try:
        payload = read_payload(args)
    except (OSError, ValueError, RecursionError) as exc:  # ValueError covers bad JSON and bad encodings
        print(f"could not read the payload: {exc}", file=sys.stderr)
        return 2

    try:
        predictor = ChurnPredictor.load(args.model_dir)
    except Exception as exc:  # missing, corrupt or incompatible model files
        print(f"could not load the model: {exc}", file=sys.stderr)
        return 1

    try:
        result = predictor.predict(payload)
    except ValidationError as exc:
        print("invalid customer payload:", file=sys.stderr)
        for err in exc.errors():
            field = ".".join(str(part) for part in err["loc"]) or "<payload>"
            print(f"  {field}: {err['msg']}", file=sys.stderr)
        return 2

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
