import argparse
import logging
import pandas as pd

from engine.orchestration.training_pipeline import TrainingPipeline


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--model_name", default="credit_model")
    parser.add_argument("--version", required=True)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Load data
    df = pd.read_csv(args.data_path)

    # Run pipeline
    pipeline = TrainingPipeline()

    result = pipeline.run(
        df=df,
        target_column=args.target,
        model_name=args.model_name,
        version=args.version
    )

    print("Training completed:")
    print(result)


if __name__ == "__main__":
    main()