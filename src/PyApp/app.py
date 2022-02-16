from dataclasses import dataclass
import hnsw
from IHnsw import IHnsw, KnnResults
import numpy as np

N = "\n"

@dataclass
class Setup:
	dataSeed: int = 200
	dim: int = 16
	efConstruction: int = 200
	elementCount: int = 1000
	k: int = 10
	M: int = 16
	seed: int = 100
	space: hnsw.Space = hnsw.Space.EUCLIDEAN

	def __post_init__(self):
		self.queryCount = 1 # max(1, self.elementCount // 100)
		np.random.seed(self.dataSeed)
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
	print("Search completed.\n")
	return res

def areEqual(refRes: KnnResults, subRes: KnnResults):
	print(subRes, end="\n\n")
	return np.array_equal(refRes[0], subRes[0]) and np.array_equal(refRes[1], subRes[1])

def checkAreEqual(refRes: KnnResults, subRes: KnnResults):
	print(f"Results equal: {areEqual(refRes, subRes)}.{N}")

def main():
	setup = Setup()
	refRes = run(hnsw.HnswlibIndexFloat32, setup)
	print(refRes, end="\n\n")
	checkAreEqual(refRes, run(hnsw.ChmOrigIndexFloat32, setup))
	checkAreEqual(refRes, run(hnsw.ChmOptimIndexFloat32, setup))
	checkAreEqual(refRes, run(hnsw.BruteforceIndexFloat32, setup))

if __name__ == "__main__":
	main()
