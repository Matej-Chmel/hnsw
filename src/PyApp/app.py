from dataclasses import dataclass
import hnsw
from IHnsw import IHnsw, KnnResults
import numpy as np

@dataclass
class Setup:
	dim: int = 16
	efConstruction: int = 200
	elementCount: int  = 10000
	k: int  = 10
	M: int  = 16
	seed: int  = 100
	space: hnsw.Space = hnsw.Space.EUCLIDEAN

	def __post_init__(self):
		self.queryCount = max(1, self.elementCount // 100)
		self.elements = generateData(self.elementCount, self.dim)
		self.queries = generateData(self.queryCount, self.dim)

def generateData(count: int, dim: int) -> np.ndarray:
	return np.float32(np.random.random((count, dim)))

def run(cls, setup: Setup):
	print(f"Building index {cls.__name__}.")
	index: IHnsw = cls(setup.space, setup.dim)
	index.init_index(setup.elementCount, setup.M, setup.efConstruction, setup.seed)
	index.add_items(setup.elements)

	print("Index built.\nSearching.")
	res = index.knn_query(setup.queries, setup.k)
	print("Search completed.")
	return res

def areEqual(refRes: KnnResults, subRes: KnnResults):
	return np.array_equal(refRes[0], subRes[0]) and np.array_equal(refRes[1], subRes[1])

def main():
	setup = Setup()
	refRes = run(hnsw.HnswlibIndexFloat32, setup)
	subRes = run(hnsw.ChmOrigIndexFloat32, setup)
	print(f"Results equal: {areEqual(refRes, subRes)}.")

if __name__ == "__main__":
	main()
