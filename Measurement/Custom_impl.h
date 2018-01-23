#pragma once

#include "Custom.h"

class TRN::Measurement::Custom::Handle
{
public:
	std::function<void(const unsigned long long &evaluation_id, const std::vector<float> &primed, const std::vector<float> &predicted, const std::vector<float> &expected, const std::size_t &preamble, const std::size_t &pages, const std::size_t &rows, const std::size_t &cols)> functor;
};
