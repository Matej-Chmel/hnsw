import cpp_module as m
import numpy as np

def run(data: list[float]):
	try:
		with m.cppStdout(stderr=True, stdout=True):
			m.print(np.array(data))
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
