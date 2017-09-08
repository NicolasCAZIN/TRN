#pragma once

#include "MeanSquareError.h"

class TRN::Measurement::MeanSquareError::Handle
{
public :
	std::function<void(const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const std::size_t &cols)> functor;
};
