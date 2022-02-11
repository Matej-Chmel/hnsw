#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <ios>
#include <vector>

namespace fs = std::filesystem;

int main() {
	const auto slnDir = fs::path(SOLUTION_DIR);
	const auto datasetsDir = slnDir / "datasets";
	const auto dataPath = datasetsDir / "SIFT1M.bin";
	std::vector<float> data;

	{
		std::ifstream s(dataPath, std::ios::binary);

		s.seekg(0, std::ios::end);
		const auto fileSize = s.tellg();
		s.seekg(0, std::ios::beg);

		data.resize(fileSize / sizeof(float));
		s.read(reinterpret_cast<std::ifstream::char_type*>(data.data()), fileSize);
	}

	const auto outDir = slnDir / "out";

	if(!fs::exists(outDir))
		fs::create_directories(outDir);

	std::ofstream out(outDir / "SIFT1M.txt");
	const auto origPrecision = out.precision();

	for(size_t i = 0; i < 100; i++) {
		const auto val = data[i];

		if(std::floorf(val) == val)
			out << std::setprecision(1) << std::fixed << val;
		else {
			out.unsetf(std::ios_base::fixed);
			out << std::setprecision(origPrecision) << val;
		}

		out << '\n';
	}

	return 0;
}
