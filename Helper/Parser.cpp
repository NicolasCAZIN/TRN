#include "stdafx.h"
#include "Parser.h"

void TRN::Helper::Parser::place_cells_model(const std::string &filename, const float &radius_threshold, 
	std::vector<float> &x, std::vector<float> &y, std::vector<float> &K)
{
	std::ifstream stream(filename.c_str());
	static const std::string separators = ";,\t ";
	//std::cout << stream.is_open() << std::endl;
	strtk::for_each_line(stream, [&](const std::string& line)
	{
		float cx, cy, radius;
		if (strtk::parse_columns(line, separators, strtk::column_list(3, 4, 5), cx, cy, radius))
		{
			x.push_back(cx);
			y.push_back(cy);
			auto width = std::logf(radius_threshold) / (radius * radius);
			K.push_back(width);

			//std::cout << cx << "\t" << cy << "\t" << width << std::endl;
		}
	});

}