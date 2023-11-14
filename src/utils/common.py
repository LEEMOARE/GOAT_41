import git
import json
from pathlib import Path
from typing import Union
import zipfile

def get_repo_root() -> str:
    repo = git.Repo('.', search_parent_directories=True)
    return repo.working_tree_dir

def load_json(path:Union[str, Path]):
    if Path(path).suffix == '.zip':
        with zipfile.ZipFile(path, 'r') as zip_ref:
            for file in zip_ref.namelist():
                if file.endswith('.json'):
                    return json.load(zip_ref.open(file))
    else:
        with open(path, 'r') as f:
            return json.load(f)
        
    


