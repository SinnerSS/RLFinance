import pandas as pd
from pathlib import Path
from typing import Literal
from dataclasses import dataclass, field

@dataclass
class Config:
    pool_size: int = 300
    pool_strategy: Literal['mixed', 'top', 'bot'] = 'mixed'

    data_path: Path = field(default_factory=lambda: Path('data').resolve()) 
    result_path: Path = field(default_factory=lambda: Path('result').resolve())

    start_train: pd.Timestamp = field(default_factory=lambda: pd.to_datetime('2016-01-01'))
    end_train: pd.Timestamp = field(default_factory=lambda: pd.to_datetime('2020-12-31'))
    start_test: pd.Timestamp = field(default_factory=lambda: pd.to_datetime('2021-01-01'))
    end_test: pd.Timestamp = field(default_factory=lambda: pd.to_datetime('2022-12-31'))

    def __post_init__(self):
        assert self.data_path.is_dir(), 'Data directory do not exists!'
        self.result_path.mkdir(parents=True, exist_ok=True)
