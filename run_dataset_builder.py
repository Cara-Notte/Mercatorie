from pathlib import Path

from src.dataset_builder.run_dataset_pipeline import run_dataset_pipeline, save_pipeline_outputs


def main() -> None:
    raw_dir = Path("data/raw_samples")
    out_dir = Path("data/processed")

    for raw_file in sorted(raw_dir.glob("*.xlsx")):
        yearly_out = out_dir / raw_file.stem
        outputs = run_dataset_pipeline(raw_file)
        save_pipeline_outputs(outputs, yearly_out)
        print(f"Processed {raw_file} -> {yearly_out}")


if __name__ == "__main__":
    main()