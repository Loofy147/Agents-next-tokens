import os
import subprocess
import pandas as pd
from typing import List, Dict, Any, Optional

class KaggleManager:
    """
    Manages interaction with Kaggle Competitions.
    Allows the agent to search, download data, and submit solutions.
    """
    def __init__(self, username: str = "hichambedrani", token: str = "KGAT_7972aa3c1ae3f10a452943afc4b51193"):
        self.username = username
        self.token = token
        self._setup_config()

    def _setup_config(self):
        kaggle_dir = os.path.expanduser("~/.kaggle")
        os.makedirs(kaggle_dir, exist_ok=True)
        config_path = os.path.join(kaggle_dir, "kaggle.json")
        import json
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

        # Parse CSV output manually or via pandas
        from io import StringIO
        df = pd.read_csv(StringIO(result.stdout))
        return df.to_dict('records')

    def download_data(self, competition: str, path: str = "data"):
        os.makedirs(path, exist_ok=True)
        cmd = ["kaggle", "competitions", "download", "-c", competition, "-p", path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Successfully downloaded data for {competition}")
            # Unzip if necessary
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
