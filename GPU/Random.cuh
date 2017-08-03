#pragma once
#include <cuda.h>

void random_uniform(const cudaStream_t &stream,
	const unsigned long &seed,
	const float &a, const float &b, const float &sparsity,
	const std::size_t &batch_size, const std::size_t &rows, const std::size_t &cols,
	float **x, const std::size_t &x_stride);


void random_gaussian(const cudaStream_t &stream,
	const unsigned long &seed,
	const float &mu, const float &sigma,
	const std::size_t &batch_size, const std::size_t &rows, const std::size_t &cols,
	float **x, const std::size_t &x_stride);


