import pandas as pd
import os 
import json
from processing_demo import save_cache, normalize_review, process_for_svm, process_for_phobert, process_for_qwen
from tqdm import tqdm
# Bật tqdm cho pandas
tqdm.pandas()
PROGRESS_FILE = "progress.json" # luu tien do xu ly

def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Không tìm thấy file tại {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data = pd.DataFrame(data)
    mapping_columns = {
        "reviewId": "id",
        "title": "game",
        "userName": "usergame",
        "score": "rating",
        "content": "review",
        "at": "timestamp",
        "appVersion": "versiongame"
    }
    data = data.rename(columns=mapping_columns)
    data = data.dropna(subset=["review"])
    if "versiongame" in data.columns:
        data["versiongame"] = data["versiongame"].fillna("unknown")
    else:
        data["versiongame"] = "unknown"
    cols = ["id", "game", "usergame", "rating", "review", "timestamp", "versiongame"]
    data = data[cols]
    return data

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"last_index": -1}

def save_progress(idx):
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump({"last_index": idx}, f)

if __name__ == "__main__":
    input_path = "lien_quan_theo_ti_le.json"
    output_path = "processed.csv"
    df = load_data(input_path)
    BATCH_SIZE = 50
    progress = load_progress()
    start_index = progress["last_index"] + 1
    if start_index >= len(df):
        print("Da hoan tat tien xu ly")
    else:
        for start in range(start_index, len(df), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(df))
            df_batch = df.iloc[start:end].copy()

            print(f"\nProcessing batch {start} -> {end}/{len(df)}")
            df_batch["normalized"] = df_batch["review"].progress_apply(normalize_review)

            print(f"\nBatch {start}-{end}")
            print("Total:", len(df_batch))
            print("Valid:", df_batch["normalized"].notna().sum())
            print("Fail:", df_batch["normalized"].isna().sum())

            # luu fail review do api loi
            failed = df_batch[df_batch["normalized"].isna()]
            if not failed.empty:
                is_failed_header = not os.path.exists("failed.csv")
                failed.to_csv("failed.csv", mode='a', header=is_failed_header, index=False, encoding="utf-8-sig")

            # chi xu ly review hop le
            df_valid = df_batch[df_batch["normalized"].notna()].copy()
            if not df_valid.empty:
                df_valid["svm_review"] = df_valid["normalized"].apply(process_for_svm)
                df_valid["phobert_review"] = df_valid["normalized"].apply(process_for_phobert)
                df_valid["qwen_review"] = df_valid["normalized"].apply(process_for_qwen)
                is_header = not os.path.exists(output_path)
                df_valid.to_csv(output_path, mode='a', header=is_header, index=False, encoding="utf-8-sig")
            save_progress(end - 1)
    print(f"Da luu file processed.csv")
