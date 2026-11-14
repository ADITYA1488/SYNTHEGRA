from pathlib import Path

from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_DIR = Path(__file__).resolve().parent
TEST_DATASET = PROJECT_DIR / "test_dataset.csv"


def main():
    print("Loading dataset...")
    ds = load_dataset(
        "bitext/bitext-customer-support-llm-chatbot-training-dataset"
    )
    df = ds["train"].to_pandas()

    df["group_key"] = (
        df["instruction"].astype(str).str.strip().str.lower()
        .str.replace(r"\s+", " ", regex=True)
    )

    groups = (
        df[["group_key", "intent"]]
        .drop_duplicates("group_key")
        .sort_values("group_key")
        .reset_index(drop=True)
    )

    remaining, test_groups = train_test_split(
        groups,
        test_size=0.15,
        random_state=42,
        stratify=groups["intent"]
    )

    train_groups, val_groups = train_test_split(
        remaining,
        test_size=0.15 / 0.85,
        random_state=42,
        stratify=remaining["intent"]
    )

    test_df = df[
        df["group_key"].isin(set(test_groups["group_key"]))
    ].copy()

    print("\nDataset rows:", len(df))
    print("Unique groups:", len(groups))
    print("Test groups:", len(test_groups))
    print("Test rows:", len(test_df))

    assert len(test_df) == 4016, "Test split size mismatch!"

    test_df.drop(columns=["group_key"]).to_csv(
        TEST_DATASET,
        index=False
    )

    print("\nSaved:", TEST_DATASET)
    print("Exact test split reconstructed successfully.")


if __name__ == "__main__":
    main()
