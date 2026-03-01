import os
import subprocess
import pandas as pd
import json
from typing import List, Dict, Any, Optional

class KaggleManager:
    """
    Manages interaction with Kaggle Competitions, Datasets, and Kernels.
    Allows the agent to search, download data, submit solutions,
    upload datasets, and push notebooks.
    """
    def __init__(self, username: str = "hichambedrani", token: str = "KGAT_7972aa3c1ae3f10a452943afc4b51193"):
        self.username = username
        self.token = token
        self._setup_config()

    def _setup_config(self):
        kaggle_dir = os.path.expanduser("~/.kaggle")
        os.makedirs(kaggle_dir, exist_ok=True)
        config_path = os.path.join(kaggle_dir, "kaggle.json")
        with open(config_path, "w") as f:
            json.dump({"username": self.username, "key": self.token}, f)
        os.chmod(config_path, 0o600)

    def list_competitions(self, search: str = "") -> List[Dict[str, str]]:
        cmd = ["kaggle", "competitions", "list", "--csv"]
        if search:
            cmd.extend(["-s", search])

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error listing competitions: {result.stderr}")
            return []

        from io import StringIO
        df = pd.read_csv(StringIO(result.stdout))
        return df.to_dict('records')

    def download_data(self, competition: str, path: str = "data"):
        os.makedirs(path, exist_ok=True)
        cmd = ["kaggle", "competitions", "download", "-c", competition, "-p", path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Successfully downloaded data for {competition}")
            import zipfile
            zip_path = os.path.join(path, f"{competition}.zip")
            if os.path.exists(zip_path):
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(path)
                os.remove(zip_path)
        else:
            print(f"Error downloading data: {result.stderr}")

    def submit(self, competition: str, file_path: str, message: str):
        cmd = ["kaggle", "competitions", "submit", "-c", competition, "-f", file_path, "-m", message]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Successfully submitted to {competition}")
        else:
            print(f"Error submitting: {result.stderr}")

    def create_dataset(self, folder_path: str, title: str, slug: str):
        """Creates or updates a dataset on Kaggle."""
        metadata_path = os.path.join(folder_path, "dataset-metadata.json")
        metadata = {
            "title": title,
            "id": f"{self.username}/{slug}",
            "licenses": [{"name": "CC0-1.0"}]
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Check if dataset exists
        check_cmd = ["kaggle", "datasets", "status", f"{self.username}/{slug}"]
        check_res = subprocess.run(check_cmd, capture_output=True, text=True)

        if "NotFound" in check_res.stderr or check_res.returncode != 0:
            cmd = ["kaggle", "datasets", "create", "-p", folder_path, "-u"]
        else:
            cmd = ["kaggle", "datasets", "version", "-p", folder_path, "-m", "Auto-update from Hiro Agent"]

        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(f"Dataset error: {result.stderr}")

    def push_notebook(self, folder_path: str, slug: str, title: str, code_file: str, dataset_slugs: List[str] = []):
        """Pushes a notebook to Kaggle."""
        metadata_path = os.path.join(folder_path, "kernel-metadata.json")

        # Ensure the code file exists in the folder
        if not os.path.exists(os.path.join(folder_path, code_file)):
            print(f"Error: {code_file} not found in {folder_path}")
            return

        metadata = {
            "id": f"{self.username}/{slug}",
            "title": title,
            "code_file": code_file,
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": "true",
            "enable_tpu": "false",
            "enable_internet": "true",
            "dataset_sources": [f"{self.username}/{ds}" if "/" not in ds else ds for ds in dataset_slugs],
            "competition_sources": [],
            "kernel_sources": []
        }

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        cmd = ["kaggle", "kernels", "push", "-p", folder_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(f"Notebook error: {result.stderr}")
