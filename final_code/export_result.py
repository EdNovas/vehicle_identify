import pandas as pd
from pathlib import Path
import datetime

# 1. Define your local path (using "r" for raw string to handle Windows backslashes)
RESULTS_DIR = Path(r"C:\Users\wdm17\Desktop\vehicle_identify\ultralytics_runs_results\ultralytics_runs")

def get_comparison_table(root_dir):
    rows = []
    
    # Check if directory exists
    if not root_dir.exists():
        print(f"Error: The directory {root_dir} does not exist.")
        return None

    print(f"Scanning {root_dir} for results.csv files...\n")

    # 2. Iterate through all subdirectories looking for 'results.csv'
    # We use rglob('*') to find any file named results.csv recursively, 
    # or simple glob if they are just one level deep. 
    # Using glob('*/results.csv') assumes structure: root/model_name/results.csv
    csv_files = list(root_dir.glob("*/results.csv"))

    if not csv_files:
        print("No results.csv files found! Check your folder structure.")
        return None

    for csv_path in csv_files:
        model_name = csv_path.parent.name  # Use folder name as model name
        
        try:
            # Read CSV and strip whitespace from column names (YOLO csvs often have spaces)
            df = pd.read_csv(csv_path)
            df.columns = df.columns.str.strip()
            
            # Check if empty
            if df.empty:
                print(f"[SKIP] {model_name}: File is empty.")
                continue

            # 3. Identify Key Columns
            # YOLOv8/11 standard columns usually include these names
            map50_95_col = 'metrics/mAP50-95(B)'
            map50_col = 'metrics/mAP50(B)'
            prec_col = 'metrics/precision(B)'
            recall_col = 'metrics/recall(B)'
            time_col = 'time'  # Cumulative time in seconds (usually)
            
            if map50_95_col not in df.columns:
                print(f"[SKIP] {model_name}: Could not find mAP columns.")
                continue

            # 4. Find the "Best" Epoch (Max mAP50-95)
            best_row_idx = df[map50_95_col].idxmax()
            best_row = df.loc[best_row_idx]
            
            # 5. Calculate Training Time
            # Taking the max time value found in the file (assuming cumulative seconds)
            # If 'time' column is missing, we skip time calculation
            if time_col in df.columns:
                total_seconds = df[time_col].max()
                # Convert to readable format (Hours:Minutes:Seconds)
                train_time_str = str(datetime.timedelta(seconds=int(total_seconds)))
            else:
                train_time_str = "N/A"

            # 6. Append data
            rows.append({
                "Model": model_name,
                "mAP50-95": best_row[map50_95_col],
                "mAP50": best_row[map50_col],
                "Precision": best_row[prec_col],
                "Recall": best_row[recall_col],
                "Best Epoch": int(best_row['epoch']),
                "Total Epochs": int(df['epoch'].max()),
                "Train Time": train_time_str
            })
            
        except Exception as e:
            print(f"[ERROR] Could not process {model_name}: {e}")

    # 7. Create DataFrame and Sort
    if not rows:
        return None

    comp_df = pd.DataFrame(rows)
    # Sort by mAP50-95 descending (best first)
    comp_df = comp_df.sort_values("mAP50-95", ascending=False)
    
    return comp_df

# Run the function
df_final = get_comparison_table(RESULTS_DIR)

if df_final is not None:
    # formatting for cleaner float output
    pd.options.display.float_format = '{:,.4f}'.format
    
    print("\n=== Final Model Comparison Table ===")
    print(df_final.to_string(index=False))
    
    # Optional: Save to CSV in the same folder
    output_path = RESULTS_DIR / "final_model_comparison.csv"
    df_final.to_csv(output_path, index=False)
    print(f"\n[Saved] Comparison table saved to: {output_path}")