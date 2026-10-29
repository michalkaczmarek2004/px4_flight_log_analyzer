import argparse
from pathlib import Path

from src import metrics, parser, preprocessing, report, validation


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze a PX4 .ulg flight log.")
    ap.add_argument("log", help="path to a .ulg file")
    ap.add_argument("--topics", action="store_true", help="list every topic and instance in the log")
    ap.add_argument("--export", metavar="TOPIC", help="save one topic as CSV")
    ap.add_argument("--instance", type=int, default=0, help="instance of --export topic (default 0)")
    ap.add_argument("--out", default="output", help="directory for exported CSV files (default: output)")
    args = ap.parse_args()

    # Listing topics needs the whole log; metrics only need a few topics.
    wanted = list(metrics.REQUIRED_TOPICS)
    if args.export:
        wanted.append(args.export)
    topics = parser.load_topics(args.log, None if args.topics else wanted)

    if args.topics:
        report.print_topics(parser.list_topics(topics))
        print()

    if args.export:
        df = parser.get_topic(topics, args.export, args.instance)
        out = Path(args.out) / f"{Path(args.log).stem}_{parser.topic_key(args.export, args.instance)}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"Exported {args.export} instance {args.instance} ({len(df)} rows) to {out}\n")

    # Validate only the topics the metrics use; with --topics this would flag every unrelated topic.
    required = {name: df for name, df in topics.items() if name in metrics.REQUIRED_TOPICS}
    issues = validation.validate(required)
    report.print_validation(issues)
    print()
    if validation.has_errors(issues):
        print("Some metrics may be missing or wrong because of the errors above.\n")

    # Validation reports on the raw data; metrics use the cleaned data.
    clean = preprocessing.preprocess(required)
    report.print_masked(clean)
    print()
    report.print_metrics(metrics.compute_metrics(clean))


if __name__ == "__main__":
    main()
