import json
from matplotlib import pyplot as plt
import pandas
from pathlib import Path

N = "\n"
Y_LABEL = "Execution time (ns)"

def algoRecallsToStr(algos: list[str], results: dict, idx: int):
	s = "Recall:\n"

	for a in algos:
		s += f"{a}: {results[a]['searches'][idx]['recall']:.3f}{N}"

	return s

def algoTimesToStr(algos: list[str], results: dict, title: str, attr: str, idx: int = None):
	s = f"{N}{title}:{N}"

	for a in algos:
		ns = results[a][attr]

		if idx is not None:
			ns = ns[idx]["timeNS"]

		s += f"{a}: {timeDeltaNS(ns)}{N}"

	return s

def getAlgos(results: dict):
	algos = list(results.keys())
	algos.remove("setup")
	return algos

def getDatasetName(r: dict):
	return r['setup']['datasetName']

def getSubplots():
	return plt.subplots(figsize=(12, 7))

def plotAttr(results: dict, attr: str, path: Path, title: str):
	fig, ax = getSubplots()
	algos = getAlgos(results)
	times = [results[a][attr] for a in algos]

	plt.bar(algos, times)
	plt.title(f"{title}, {getDatasetName(results)}")
	plt.xlabel("Algorithm")
	plt.ylabel(Y_LABEL)

	fig.savefig(path)
	plt.show()

def plotSearches(results: dict, path: Path):
	fig, ax = getSubplots()
	algos = getAlgos(results)

	for a in algos:
		recalls, times = [], []

		for s in results[a]["searches"]:
			recalls.append(s["recall"])
			times.append(s["timeNS"])

		plt.plot(recalls, times, label=a, marker="o")

	plt.legend()
	plt.title(f"Search, {getDatasetName(results)}")
	plt.xlabel("Recall")
	plt.ylabel(Y_LABEL)

	fig.savefig(path)
	plt.show()

def prettyPrint(path: Path, results: dict):
	s = resultsToStr(results)
	print(s)

	with path.open("w", encoding="utf-8") as f:
		f.write(s)

def resultsToStr(results: dict):
	s = ""
	setup = results["setup"]

	for k, v in setup.items():
		s += f"{upperFirst(k)}: {v}{N}"

	algos = getAlgos(results)
	s += algoTimesToStr(algos, results, "Build times", "buildTimeNS")
	s += algoTimesToStr(algos, results, "Initialization times", "initTimeNS")

	searches = results[algos[0]]["searches"]

	for i in range(len(searches)):
		s += algoTimesToStr(algos, results, f"Search, ef={searches[i]['ef']}", "searches", i)
		s += algoRecallsToStr(algos, results, i)

	return s

def timeDeltaNS(ns: int):
	return pandas.Timedelta(nanoseconds=ns)

def upperFirst(s: str):
	if len(s) < 2:
		return s.upper()
	return s[0].upper() + s[1:]

def main():
	outDir = Path(__file__).parents[2] / "out"

	with (outDir / "results.json").open("r", encoding="utf-8") as f:
		results = json.load(f)

	prettyPrint(outDir / "results.txt", results)
	plotAttr(results, "buildTimeNS", outDir / "buildTime.pdf", "Build time")
	plotAttr(results, "initTimeNS", outDir / "initTime.pdf", "Initialization time")
	plotSearches(results, outDir / "search.pdf")

if __name__ == "__main__":
	main()
