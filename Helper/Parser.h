#pragma once

#include "helper_global.h"

namespace TRN
{
	namespace Helper
	{
		namespace Parser
		{
			void HELPER_EXPORT place_cells_model(const std::string &filename, const float &radius_threshold, std::vector<float> &x, std::vector<float> &y, std::vector<float> &K);
		};
	
	};
};