#pragma once

#include "Matrix.h"

class TRN::Core::Matrix::Handle
{
	public:
		std::size_t rows;
		std::size_t cols;


		std::size_t stride;
		float *elements;
		bool ownership;
};