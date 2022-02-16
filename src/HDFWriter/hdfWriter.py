import h5py as hdf
import hnsw
import numpy as np
from pathlib import Path

def generateData(count: int, dim: int, seed: int):
	np.random.seed(seed)
	return np.float32(np.random.random((count, dim)))

def getNeighbors(K: int, space: hnsw.Space, test: np.ndarray, train: np.ndarray):
	index = hnsw.BruteforceIndexFloat32(space, train.shape[1])
	index.init_index(train.shape[0])
	index.add_items(train)
	return index.knn_query(test, K)[0]

def run(dim: int, K: int, seed: int, space: hnsw.Space, testCount: int, trainCount: int):
	train = generateData(trainCount, dim, seed)
	test = generateData(testCount, dim, seed + 1)
	neighbors = getNeighbors(K, space, test, train)
	datasetsDir = Path(__file__).parent.parent.parent / "datasets"
	datasetsDir.mkdir(exist_ok=True)
	write(neighbors, test, train, datasetsDir / f"d{dim}_tr{trainCount}_k{K}_te{testCount}_s{seed}_{spaceToStr(space)}.hdf5")

def spaceToStr(s: hnsw.Space):
	return str(s).split(".")[1].lower()

def write(neighbors: np.ndarray, test: np.ndarray, train: np.ndarray, path: Path):
	with hdf.File(path, "w") as f:
		f.create_dataset("neighbors", data=neighbors)
		f.create_dataset("test", data=test)
		f.create_dataset("train", data=train)

	print(f"Written {path}.")

def main():
	run(128, 10, 100, hnsw.Space.EUCLIDEAN, 200, 20000)

if __name__ == "__main__":
	main()
