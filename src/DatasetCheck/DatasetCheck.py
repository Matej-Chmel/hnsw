import h5py as hdf
import numpy as np
from pathlib import Path

def check(file: hdf.File, attr: str, datasetsDir: Path, binName: str, dtype: np.dtype):
	binData = np.fromfile((datasetsDir / binName).with_suffix(".bin"), dtype=dtype)
	hdfData = file[attr][:].flatten()
	return np.array_equal(binData, hdfData)

def printCheck(file: hdf.File, attr: str, datasetsDir: Path, binName: str, dtype: np.dtype = np.float32):
	print(f'Binary and HDF formats of dataset "{attr}" are equal: {check(file, attr, datasetsDir, binName, dtype)}.')

def main():
	datasetsDir = Path(__file__).parents[2] / "datasets"

	with hdf.File(datasetsDir / "d128_tr20000_k100_te200_s100_euclidean.hdf5", "r") as f:
		printCheck(f, "neighbors", datasetsDir, "neighbors", np.uint64)
		printCheck(f, "train", datasetsDir, "train")
		printCheck(f, "test", datasetsDir, "test")

if __name__ == "__main__":
	main()
