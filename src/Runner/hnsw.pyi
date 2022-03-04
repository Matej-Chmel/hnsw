from enum import auto, Enum
import IHnsw
import numpy as np

class Space(Enum):
	ANGULAR = auto()
	EUCLIDEAN = auto()
	INNER_PRODUCT = auto()

class BruteforceIndexFloat32(IHnsw.Index):
	...

class ChmOptimIndexFloat32(IHnsw.Index):
	...

class ChmOrigIndexFloat32(IHnsw.Index):
	...

class HnswlibIndexFloat32(IHnsw.Index):
	...

def getRecallUInt64(correctLabels: np.ndarray, testedLabels: np.ndarray) -> float: ...
