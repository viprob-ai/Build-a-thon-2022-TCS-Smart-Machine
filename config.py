

from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

from pathlib import PosixPath

class Dataset(SimpleNamespace): 
    name: str
    data_path: PosixPath
    train_path: PosixPath
    test_path: PosixPath


