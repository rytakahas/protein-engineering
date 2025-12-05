
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class Variant:
    pos: int
    wt: str
    mut: str
    score: float

@dataclass
class DMSSet:
    uniprot: str
    variants: List[Variant]

@dataclass
class StabilitySet:
    uniprot: str
    ddg_values: List[float]
