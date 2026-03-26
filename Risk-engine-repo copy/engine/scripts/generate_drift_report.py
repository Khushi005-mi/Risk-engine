import argparse
import logging
import pandas as pd

from engine.monitoring.psi import PSI


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--reference_data", required=True)
    parser.add_argument("--current_data", required=True)
    parser.add_argument("--output_path", required=True)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    ref = pd.read_csv(args.reference_data)
    cur = pd.read_csv(args.current_data)

    psi_calculator = PSI()

    report = {}

    for col in ref.columns:
        if col in cur.columns:
            psi_value = psi_calculator.calculate(ref[col], cur[col])
            report[col] = psi_value

    report_df = pd.DataFrame(list(report.items()), columns=["feature", "psi"])

    report_df.to_csv(args.output_path, index=False)

    print("Drift report generated")


if __name__ == "__main__":
    main()