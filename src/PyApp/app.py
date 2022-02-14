import cpp_module as m
import numpy as np

def run(data: list[float]):
	try:
		arr = np.array(data, dtype=np.float32)
		m.multiply(arr, 2.0)
		print(arr)
	except RuntimeError as e:
		print(e)

run([1, 2, 3])
run([
	[1, 2, 3],
	[4, 5, 6]
])
run([[
		[1, 2, 3],
		[4, 5, 6]
	],[
		[1, 2, 3],
		[4, 5, 6]
	]
])
