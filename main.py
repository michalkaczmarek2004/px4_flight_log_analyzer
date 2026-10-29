import argparse

from src.parser import load_ulog


def print_summary(path: str) -> None:
    log = load_ulog(path)

    print(f"File:        {log.path.name}")
    print(f"Duration:    {log.duration_s:.1f} s")
    print(f"UTC start:   {log.utc_start or 'unknown (no GPS time)'}")
    print(f"Dropouts:    {log.dropouts_s:.3f} s")
    print(f"Hardware:    {log.info.get('ver_hw', '?')}")
    print(f"Firmware:    {log.firmware_version}")
    print(f"System:      {log.info.get('sys_name', '?')}")
    print(f"Parameters:  {len(log.parameters)}")

    print(f"\nTopics ({len(log.topics)}):")
    for name, df in sorted(log.topics.items()):
        print(f"  {name:<40} {len(df):>7} samples")

    print(f"\nMessages ({len(log.messages)}):")
    for msg in log.messages:
        print(f"  [{msg.timestamp_s:8.2f}s] {msg.level:<8} {msg.text}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Print a summary of a PX4 .ulg flight log.")
    ap.add_argument("log", help="path to a .ulg file")
    print_summary(ap.parse_args().log)
