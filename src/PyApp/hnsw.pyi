from enum import auto, Enum
from IHnsw import IHnsw
import numpy as np

class Space(Enum):
	EUCLIDEAN = auto()
	INNER_PRODUCT = auto()

class ChmOrigIndexFloat32(IHnsw):
	...

class HnswlibIndexFloat32(IHnsw):
	...