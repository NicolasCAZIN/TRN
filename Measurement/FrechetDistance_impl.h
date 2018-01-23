#pragma once

#include "FrechetDistance.h"

class TRN::Measurement::FrechetDistance::Handle
{
public:
	std::function<void(const unsigned long long &evaluation_id, const std::vector<float> &values, const std::size_t &rows, const std::size_t &cols)> functor;
};
