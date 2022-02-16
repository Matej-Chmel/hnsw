from dataclasses import dataclass, field
import hnsw
import IHnsw
import json
import numpy as np
from pathlib import Path
import time

@dataclass
class Setup:
	dim: int
	efConstruction: int
	efs: list[int]
	elements: np.ndarray
	k: int
	M: int
	queries: np.ndarray
	seed: int
	space: hnsw.Space

@dataclass
class SearchResult:
	execTimeNS: int
	recall: float

@dataclass
class HnswResult:
	name: str
	buildTimeNS: int = 0
	initTimeNS: int = 0
	searches: list[SearchResult] = field(default_factory=list)

	def toDict(self, efs: list[int]):
		return {
			"buildTimeNS": self.buildTimeNS,
			"initTimeNS": self.initTimeNS,
			"searches": [
				{
					"ef": efs[i],
					"execTimeNS": self.searches[i].execTimeNS,
					"recall": self.searches[i].recall
				}
				for i in range(len(efs))
			]
		}

def generateData(dim: int, elementCount: int, queryCount: int, seed: int):
	np.random.seed(seed)
	return np.float32(np.random.random((elementCount, dim))), np.float32(np.random.random((queryCount, dim)))

def getRecall(refLabels: np.ndarray, subLabels: np.ndarray):
	correct = 0

	for i in range(refLabels.shape[0]):
		for label in subLabels[i]:
			for correctLabel in refLabels[i]:
				if label == correctLabel:
					correct += 1
					break

	return float(correct) / (refLabels.shape[0] * refLabels.shape[1])

def buildIndex(index: IHnsw.Index, setup: Setup):
	beginTime = time.perf_counter_ns()
	index.add_items(setup.elements)
	endTime = time.perf_counter_ns()
	return endTime - beginTime

def initIndex(cls, setup: Setup) -> tuple[IHnsw.Index, int]:
	beginTime = time.perf_counter_ns()
	index = cls(setup.space, setup.dim)
	index.init_index(setup.elements.shape[0], setup.M, setup.efConstruction, setup.seed)
	endTime = time.perf_counter_ns()
	return index, endTime - beginTime

def runBruteforce(setup: Setup) -> tuple[np.ndarray, dict]:
	res = {}
	index, res["initTimeNS"] = initIndex(hnsw.BruteforceIndexFloat32, setup)
	res["buildTimeNS"] = buildIndex(index, setup)

	beginTime = time.perf_counter_ns()
	labels, distances = index.knn_query(setup.queries, setup.k)
	endTime = time.perf_counter_ns()

	res["searchTimeNS"] = endTime - beginTime
	return labels, res

def runHNSW(cls, name: str, setup: Setup, refLabels: np.ndarray):
	res = HnswResult(name)

	index, res.initTimeNS = initIndex(cls, setup)
	res.buildTimeNS = buildIndex(index, setup)

	for i in range(len(setup.efs)):
		beginTime = time.perf_counter_ns()
		index.set_ef(setup.efs[i])
		labels, distances = index.knn_query(setup.queries, setup.k)
		endTime = time.perf_counter_ns()
		res.searches.append(SearchResult(endTime - beginTime, getRecall(refLabels, labels)))

	return res

def write(bfRes: dict, results: list[HnswResult], efs: list[int], path: Path):
	d = {"bruteforce": bfRes}

	for r in results:
		d[r.name] = r.toDict(efs)

	with path.open("w", encoding="utf-8") as f:
		json.dump(d, f, indent=4, sort_keys=True)

def main():
	dim = 16
	elements, queries = generateData(dim, 10000, 100, 100)
	setup = Setup(dim, 200, [*range(10, 30), *range(100, 2000, 100)], elements, 10, 16, queries, 101, hnsw.Space.EUCLIDEAN)

	refLabels, bfRes = runBruteforce(setup)

	results = [
		runHNSW(hnsw.HnswlibIndexFloat32, "hnswlib", setup, refLabels),
		runHNSW(hnsw.ChmOrigIndexFloat32, "chm-original", setup, refLabels),
		runHNSW(hnsw.ChmOptimIndexFloat32, "chm-optimized", setup, refLabels)
	]

	outDir = Path(__file__).parent.parent.parent / "out"
	outDir.mkdir(exist_ok=True)
	write(bfRes, results, setup.efs, outDir / "results.json")

if __name__ == "__main__":
	main()
