from typing import Dict
import csv
import os


def val_metrics_logger(metrics: Dict[str, float], log_pth: str) -> None:
    """

    :param log_pth: Path to the log files.
    :param metrics: Metrics to be logged
    :return: None.
    """
    header = False if os.path.exists(log_pth) else True
    fieldnames = metrics.keys()
    with open(log_pth, "a") as log_file:
        log_writer: csv.DictWriter = csv.DictWriter(log_file, fieldnames)
        log_writer.writeheader() if header else None
        log_writer.writerow(metrics)


def main():
    log_pth = "../loggs/log.csv"
    metrics = {
        "metric 1": 50,
        "metric 2": 4.1,
        "metric 3": "test",
    }
    val_metrics_logger(metrics, log_pth)


if __name__ == "__main__":
    main()
