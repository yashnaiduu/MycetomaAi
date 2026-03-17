import os
import json
from pathlib import Path

def generate_kaggle_metadata(data_dir: str, dataset_name: str, username: str):
    """
    Generates the kaggle.json metadata file required to push a folder as a dataset.
    """
    metadata = {
        "title": dataset_name,
        "id": f"{username}/{dataset_name.lower().replace(' ', '-')}",
        "licenses": [{"name": "CC0-1.0"}]
    }
    
    metadata_path = Path(data_dir) / "dataset-metadata.json"
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print(f"✅ Kaggle metadata generated at: {metadata_path}")
    print(f"Dataset ID will be: {metadata['id']}")

if __name__ == "__main__":
    # Project paths
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    
    # --- IMPORTANT: Replace with your actual Kaggle username ---
    KAGGLE_USERNAME = "supporoot" 
    DATASET_TITLE = "Mycetoma AI Pretraining Data"
    
    generate_kaggle_metadata(
        data_dir=str(DATA_DIR),
        dataset_name=DATASET_TITLE,
        username=KAGGLE_USERNAME
    )
    
    print("\nNext Steps:")
    print("1. Ensure you have your Kaggle API token installed at ~/.kaggle/kaggle.json")
    print("2. Run: kaggle datasets create -p data/ --dir-mode zip")
