import hnsw
import numpy as np

def generateData(count: int, dim: int) -> np.ndarray:
	return np.float32(np.random.random((count, dim)))

def main():
	dim = 16
	efConstruction = 200
	elementCount = 10000
	k = 10
	M = 16
	queryCount = max(1, elementCount // 100)
	seed = 100

	elements = generateData(elementCount, dim)
	index = hnsw.HnswlibIndexFloat(hnsw.Space.EUCLIDEAN, dim)
	index.init_index(elementCount, M, efConstruction, seed)
	index.add_items(elements)

	queries = generateData(queryCount, dim)
	labels, distances = index.knn_query(queries, k)

	print(type(elements))
	print(type(queries))
	print(labels)
	print(distances)

if __name__ == "__main__":
	main()
