from dataclasses import dataclass
import datetime as dt
from io import TextIOWrapper
import IndexModule as m
from matplotlib import pyplot as plt

N = "\n"

@dataclass
class BuildResults:
	hnswlib: m.IndexBuildRes = None
	chmOriginal: m.IndexBuildRes = None
	chmBitset: m.IndexBuildRes = None
	chmNoCache: m.IndexBuildRes = None
	chmPreAllocNeighbors: m.IndexBuildRes = None
	chmKeptHeaps: m.IndexBuildRes = None
	chmOptimized: m.IndexBuildRes = None

DIM = 128
EF_CONSTRUCTION = 200
ELEM_MAX = 1.0
ELEM_MIN = 0.0;
K = 10
M = 16
NODE_COUNT = 10000
NODE_SEED = 1000
USE_EUCLID = True

def build(cfg: m.HnswCfg, kind: m.HnswKind, nodes: m.ICoords, settings: m.HnswSettings, name: str, f: TextIOWrapper):
	res = m.Index(m.HnswType(cfg, kind, settings)).build(nodes)
	f.write(
		f"[{name}]{N}"
		f"Init: {res.init}{N}"
		f"Accumulated: {res.queryTime.accumulated}{N}"
		f"Average: {res.queryTime.avg}{N}"
		f"Total: {res.total}{N}{N}"
	)
	return res

def run():
	nodes = m.RndCoords(NODE_COUNT, DIM, ELEM_MIN, ELEM_MAX, NODE_SEED)
	cfg = m.HnswCfg(DIM, EF_CONSTRUCTION, M, nodes.getCount(DIM), NODE_SEED, USE_EUCLID)
	res = BuildResults()

	with open("build.log", "w", encoding="utf-8") as f:
		res.hnswlib = build(cfg, m.HnswKind.HNSWLIB, nodes, None, "hnswlib", f)
		res.chmOriginal = build(cfg, m.HnswKind.CHM_AUTO, nodes, m.HnswSettings(True, False, False, False), "chm-orig", f)
		res.chmBitset = build(cfg, m.HnswKind.CHM_AUTO, nodes, m.HnswSettings(True, False, True, False), "chm-bitset", f)
		res.chmNoCache = build(cfg, m.HnswKind.CHM_AUTO, nodes, m.HnswSettings(False, False, False, False), "chm-no-cache", f)
		res.chmPreAllocNeighbors = build(cfg, m.HnswKind.CHM_AUTO, nodes, m.HnswSettings(True, False, False, True), "chm-neighbors", f)
		res.chmKeptHeaps = build(cfg, m.HnswKind.CHM_AUTO, nodes, m.HnswSettings(True, True, False, False), "chm-kept-heaps", f)
		res.chmOptimized = build(cfg, m.HnswKind.CHM_AUTO, nodes, m.HnswSettings(False, True, True, True), "chm-optim", f)

	return res

def main():
	with m.cppStdout(stderr=True, stdout=True):
		res = run()

	names = ["hnswlib", "chm-orig", "chm-bitset", "chm-no-cache", "chm-neighbors", "chm-kept-heaps", "chm-optim"]
	times = [
		t / dt.timedelta(microseconds=1)
		for t in [
			res.hnswlib.total, res.chmOriginal.total, res.chmBitset.total, res.chmNoCache.total,
			res.chmPreAllocNeighbors.total, res.chmKeptHeaps.total, res.chmOptimized.total
		]
	]

	fig, ax = plt.subplots(figsize=(12, 7))
	ax.bar(names, times)
	fig.savefig("bar.pdf")
	plt.show()

if __name__ == "__main__":
	main()
