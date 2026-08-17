import argparse
import json

from .load import run_forecast_weather, run_historical


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="gridpulse-ingest", description="GridPulse data ingestion"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    hist = sub.add_parser("historical", help="Pull historical demand, generation, and weather")
    hist.add_argument("--start", required=True, metavar="YYYY-MM-DD")
    hist.add_argument("--end",   required=True, metavar="YYYY-MM-DD")
    hist.add_argument("--db",    default="gridpulse.duckdb")

    fcast = sub.add_parser("forecast", help="Pull 48h weather forecast")
    fcast.add_argument("--db", default="gridpulse.duckdb")

    args = parser.parse_args()

    if args.cmd == "historical":
        result = run_historical(args.start, args.end, args.db)
    else:
        result = run_forecast_weather(args.db)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
