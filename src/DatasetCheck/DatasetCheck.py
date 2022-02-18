import h5py as hdf
import numpy as np
from pathlib import Path

def check(file: hdf.File, attr: str, datasetsDir: Path, binName: str, type: str = "f4"):
	binData = np.fromfile((datasetsDir / binName).with_suffix(".bin"), f"<{type}")
	hdfData = file[attr][:].flatten()

	for i in range(10):
		print(binData[i], hdfData[i])

	return np.array_equal(binData, hdfData)

def main():
	datasetsDir = Path(__file__).parents[2] / "datasets"

	with hdf.File(datasetsDir / "d128_tr20000_k10_te200_s100_euclidean.hdf5", "r") as f:
		res = check(f, "neighbors", datasetsDir, "neighbors", "i8")
		print(res)
		res = check(f, "train", datasetsDir, "train")
		print(res)
		res = check(f, "test", datasetsDir, "test")
		print(res)

if __name__ == "__main__":
	main()
