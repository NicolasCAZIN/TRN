#pragma once

#include "Matrix.h"

class TRN::Core::Matrix::Handle
{
	public:
		std::size_t pages;
		std::size_t rows;
		std::size_t cols;
		std::size_t width;
		std::size_t height;

		std::size_t stride;
		float *elements;
		bool ownership;
};