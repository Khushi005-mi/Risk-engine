import argparse
import logging
import pandas as pd

from engine.orchestration.inference_pipeline import InferencePipeline


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--model_name", default="credit_model")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Load data
    df = pd.read_csv(args.input_path)

    # Run inference
    pipeline = InferencePipeline()
    result = pipeline.run(df, args.model_name)

    # Save output
    result.to_csv(args.output_path, index=False)

    print(f"Inference completed. Saved to {args.output_path}")


if __name__ == "__main__":
    main()
    