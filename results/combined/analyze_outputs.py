from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# -------- CONFIG --------
# Enter run directory name as DATE_DIR (date / time)
# the ones we have: Path("2026-01-15") / "22-04-31" or Path("2026-01-17") / "16-03-43" or Path("2026-01-18") / "02-06-25"
DATE_DIR = Path("2026-01-18") / "02-06-25"
BASE_DIR = Path("cluster_outputs") / DATE_DIR
OUT_CSV = f"results_{DATE_DIR.as_posix().replace('/', '_')}.csv"
OUT_OK_CSV = f"results_{DATE_DIR.as_posix().replace('/', '_')}_OK_runs_with_val.csv"  # output csv of only runs with validation metrics
# ------------------------

rows = []
for run_dir in sorted(
    BASE_DIR.iterdir(),
    key=lambda p: (not p.name.isdigit(), int(p.name) if p.name.isdigit() else p.name),
):  # sorted by run_id
    if not run_dir.is_dir():
        continue

    run_id = run_dir.name

    try:
        # load overrides.yaml for metadata
        overrides_path = run_dir / ".hydra" / "overrides.yaml"
        if not overrides_path.exists():
            raise FileNotFoundError("missing overrides.yaml")

        with open(overrides_path, "r") as f:
            overrides = yaml.safe_load(f)

        # converting overrides list to dict
        meta = {}
        for item in overrides:
            if "=" in item:
                k, v = item.split("=", 1)
                meta[k] = v

        dataset = meta.get("dataset", None)
        embedding = meta.get("embedding", None)
        preprocessing = meta.get("image_preprocessing", None)
        seed = meta.get("seed", None)
        image_width = meta.get("training.image_width", None)

        # loading results.csv
        results_path = run_dir / "results.csv"
        if not results_path.exists():
            raise FileNotFoundError("missing results.csv")

        df = pd.read_csv(results_path)

        # extracting final validation metrics
        val_acc = None
        val_loss = None

        if "val_acc" in df.columns:
            valid = df["val_acc"].dropna()
            if not valid.empty:
                val_acc = float(
                    valid.iloc[0]
                )  # takes first validation metrics, as we ran each for one epoch, but some have 2

        if "val_loss" in df.columns:
            valid = df["val_loss"].dropna()
            if not valid.empty:
                val_loss = float(valid.iloc[0])

        # check if validation actually ran by looking for an .npz file
        has_npz = any(run_dir.glob("epoch_*_raw_val_results.npz"))

        if val_acc is not None:
            status = "OK"
        elif has_npz:
            status = "NO_LOGGED_VAL"
        else:
            status = "NO_VAL"

    except Exception as e:
        dataset = embedding = preprocessing = seed = image_width = None
        val_acc = val_loss = None
        status = "FAILED"

    rows.append(
        {
            "date": DATE_DIR.as_posix(),
            "run_id": run_id,
            "dataset": dataset,
            "embedding": embedding,
            "preprocessing": preprocessing,
            "image_width": image_width,
            "seed": seed,
            "val_acc": val_acc,
            "val_loss": val_loss,
            "status": status,
        }
    )

# saving formatted rows to csv
out_df = pd.DataFrame(rows)
out_df.to_csv(OUT_CSV, index=False)

out_df_ok = out_df[out_df["status"] == "OK"].copy()
out_df_ok.to_csv(OUT_OK_CSV, index=False)

print(f"Saved {len(out_df)} runs to {OUT_CSV}")
print(out_df["status"].value_counts())
