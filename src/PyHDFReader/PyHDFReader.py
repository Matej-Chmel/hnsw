import h5py as hdf
from pathlib import Path

def printInfo(d: hdf.Dataset):
	print(d.shape, d.dtype)

def main():
	slnDir = Path(__file__).parent.parent.parent
	datasetsDir = slnDir / "datasets"

	sift = hdf.File(datasetsDir / "sift-128-euclidean.hdf5", "r")

	for name in ["distances", "neighbors", "test", "train"]:
		printInfo(sift[name])

	outDir = slnDir / "out"
	outDir.mkdir(exist_ok=True)

	with (outDir / "sift-hdf.txt").open("w", encoding="utf-8") as f:
		train = sift["train"]

		for i in range(100):
			f.write(f"{train[0][i]}\n")

if __name__ == "__main__":
	main()
