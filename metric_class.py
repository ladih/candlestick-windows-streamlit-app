from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ModelTradesMetrics:
    name: str
    threshold: float
    n_trades: int
    n_correct: int
    mean_return: float
    hitrate: float
    sharpe: float
    indices: List[int]
    returns: List[float]
    perm_pval_mean: Optional[float] = None
    perm_pval_hitrate: Optional[float] = None
    perm_pval_sharpe: Optional[float] = None
    boot_p_val_mean: Optional[float] = None
