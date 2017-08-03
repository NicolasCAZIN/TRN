#pragma once

#include "Custom.h"

class TRN::Measurement::Custom::Handle
{
public:
	std::function<void(const std::vector<float> &predicted, const std::vector<float> &expected, const std::size_t &pages, const std::size_t &rows, const std::size_t &cols)> functor;
};
