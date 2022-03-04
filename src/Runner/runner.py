from dataclasses import dataclass, field
import h5py as hdf
import hnsw
import IHnsw
import json
import numpy as np
import pandas
from pathlib import Path
import time

@dataclass
class Setup:
	efConstruction: int
	efs: list[int]
	M: int
	path: Path
	seed: int
	space: hnsw.Space

	def __post_init__(self):
		self.neighbors, self.test, self.train = readData(self.path)

	def toDict(self):
		return {
			"datasetName": self.path.stem,
			"dim": self.train.shape[1],
			"efConstruction": self.efConstruction,
			"K": self.neighbors.shape[1],
			"M": self.M,
			"seed": self.seed,
			"space": str(self.space).split(".")[1].lower(),
			"testCount": self.test.shape[0],
			"trainCount": self.train.shape[0]
		}

@dataclass
class SearchResult:
	recall: float
	timeNS: int

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
					"recall": self.searches[i].recall,
					"timeNS": self.searches[i].timeNS
				}
				for i in range(len(efs))
			]
		}

def buildIndex(index: IHnsw.Index, name: str, setup: Setup):
	print(f"Building index {name}...")
	beginTime = time.perf_counter_ns()
	index.add_items(setup.train)
	endTime = time.perf_counter_ns()
	ns = endTime - beginTime
	print(f"Index {name} built in {timeDeltaNS(ns)}.", end="\n\n")
	return ns

def getRecall(labels: np.ndarray, name: str, setup: Setup):
	print(f"Computing recall of index {name}...")
	beginTime = time.perf_counter_ns()
	recall = hnsw.getRecallUInt64(setup.neighbors, labels)
	endTime = time.perf_counter_ns()
	print(f"Recall computed in {timeDeltaNS(endTime - beginTime)}.", end="\n\n")
	return recall

def initIndex(cls, name: str, setup: Setup) -> tuple[IHnsw.Index, int]:
	print(f"Initializing index {name}...")
	beginTime = time.perf_counter_ns()
	index = cls(setup.space, setup.train.shape[1])
	index.init_index(setup.train.shape[0], setup.M, setup.efConstruction, setup.seed)
	endTime = time.perf_counter_ns()
	ns = endTime - beginTime
	print(f"Index {name} initialized in {timeDeltaNS(ns)}.", end="\n\n")
	return index, ns

def timeDeltaNS(ns: int):
	return pandas.Timedelta(nanoseconds=ns)

def readData(p: Path):
	with hdf.File(p, "r") as f:
		return f["neighbors"][:], f["test"][:], f["train"][:]

def runHNSW(cls, name: str, setup: Setup):
	res = HnswResult(name)

	index, res.initTimeNS = initIndex(cls, name, setup)
	res.buildTimeNS = buildIndex(index, name, setup)

	for ef in setup.efs:
		labels, ns = searchIndex(ef, index, name, setup)
		res.searches.append(SearchResult(getRecall(labels, name, setup), ns))

	return res

def searchIndex(ef: int, index: IHnsw.Index, name: str, setup: Setup):
	print(f"Searching index {name} with ef={ef}...")
	beginTime = time.perf_counter_ns()
	index.set_ef(ef)
	labels, _ = index.knn_query(setup.test, setup.neighbors.shape[1])
	endTime = time.perf_counter_ns()
	ns = endTime - beginTime
	print(f"Search completed in {timeDeltaNS(ns)}.", end="\n\n")
	return labels, ns

def writeResults(results: list[HnswResult], setup: Setup, path: Path):
	d = {r.name: r.toDict(setup.efs) for r in results}
	d["setup"] = setup.toDict()

	with path.open("w", encoding="utf-8") as f:
		json.dump(d, f, indent=4, sort_keys=True)

def main():
	# datasetName = "d16_tr20000_k10_te10_s100_euclidean"
	# datasetName = "d128_tr20000_k100_te200_s100_euclidean"
	# datasetName = "d128_tr20000_k100_te200_s100_angular"
	datasetName = "lastfm-64-dot"
	# datasetName = "sift-128-euclidean"

	slnDir = Path(__file__).parents[2]
	setup = Setup(
		200, [*range(10, 30), *range(100, 2000, 100)], 16,
		slnDir / "datasets" / f"{datasetName}.hdf5",
		200, hnsw.Space.ANGULAR
	)

	results = [
		runHNSW(hnsw.HnswlibIndexFloat32, "hnswlib", setup),
		# runHNSW(hnsw.ChmOrigIndexFloat32, "chm-original", setup),
		runHNSW(hnsw.ChmOptimIndexFloat32, "chm-optimized", setup)
	]

	outDir = slnDir / "out"
	outDir.mkdir(exist_ok=True)
	writeResults(results, setup, outDir / "results.json")

if __name__ == "__main__":
	main()
