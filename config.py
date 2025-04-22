import pandas as pd
from pathlib import Path
from typing import Literal
from dataclasses import dataclass, field

@dataclass
class Config:
    pool_size: int = 300
    pool_strategy: Literal['mixed', 'top', 'bot'] = 'mixed'

    result_path: Path = field(default_factory=lambda: Path('result').resolve())

    start_train: str = '2016-01-01'
    end_train: str = '2020-12-31'
    start_val: str = '2021-01-01'
    end_val: str = '2021-12-31'
    start_test: str = '2022-01-01'
    end_test: str = '2023-12-28'


    def __post_init__(self):
        self.result_path.mkdir(parents=True, exist_ok=True)
        self.cwd = Path.cwd()
