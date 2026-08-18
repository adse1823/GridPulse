import argparse
import json

from .load import ALL_REGIONS, run_forecast_weather, run_historical


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="gridpulse-ingest", description="GridPulse data ingestion"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    hist = sub.add_parser("historical", help="Pull historical demand, generation, and weather")
    hist.add_argument("--start", required=True, metavar="YYYY-MM-DD")
    hist.add_argument("--end", required=True, metavar="YYYY-MM-DD")
    hist.add_argument("--db", default="gridpulse.duckdb")
    hist.add_argument(
        "--regions", nargs="+", default=ALL_REGIONS,
        metavar="REGION",
        help=f"EIA respondent codes to ingest (default: all — {ALL_REGIONS})",
    )

    fcast = sub.add_parser("forecast", help="Pull 48h weather forecast")
    fcast.add_argument("--db", default="gridpulse.duckdb")
    fcast.add_argument(
        "--regions", nargs="+", default=ALL_REGIONS,
        metavar="REGION",
        help=f"Regions to fetch forecast for (default: all — {ALL_REGIONS})",
    )

    args = parser.parse_args()

    if args.cmd == "historical":
        result = run_historical(args.start, args.end, args.db, args.regions)
    else:
        result = run_forecast_weather(args.db, args.regions)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
