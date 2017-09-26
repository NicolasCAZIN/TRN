#pragma once

#include "trn4cpp_global.h"

#include "Basic.h"

namespace TRN4CPP
{
	namespace Simulation
	{
		void TRN4CPP_EXPORT  	declare(const std::string &label, const std::vector<float> &elements, const std::size_t rows, const std::size_t &cols, const std::string &tag = DEFAULT_TAG);
		void TRN4CPP_EXPORT		compute(const std::string &scenario_filename);
	};
};


