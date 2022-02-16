from enum import auto, Enum
from IHnsw import IHnsw
import numpy as np

class Space(Enum):
	EUCLIDEAN = auto()
	INNER_PRODUCT = auto()

class BruteforceIndexFloat32(IHnsw):
	...

class ChmOptimIndexFloat32(IHnsw):
	...

class ChmOrigIndexFloat32(IHnsw):
	...

class HnswlibIndexFloat32(IHnsw):
	...
