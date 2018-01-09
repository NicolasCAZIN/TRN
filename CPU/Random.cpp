#include "stdafx.h"
#include "Random.h"

#include <sstream>

template< typename T >
void throw_on_error(T result, char const *const func, const char *const file, int const line)
{
	if (result)
	{
		const int code = static_cast<unsigned int>(result);
		std::stringstream ss;
		ss << "code " << code  << " file " << file << " (" << line << ") in function " << " \"" << func << " \"";
		std::string file_and_line_func;
		ss >> file_and_line_func;

		throw std::runtime_error(file_and_line_func);
	}
}

#define check_vsl(val)           throw_on_error ( (val), #val, __FILE__, __LINE__ )


void TRN::CPU::Random::uniform_implementation(const unsigned long &seed, float **ptr, const std::size_t &batch_size, const std::size_t *rows, const std::size_t *cols, const std::size_t *strides, const bool &blank_diagonal, const float &a, const float &b, const float &sparsity)
{
	if (std::fabs(b - a) > 0.0f)
	{
		std::vector<VSLStreamStatePtr> streams(batch_size);

		for (int batch = 0; batch < batch_size; batch++)
			check_vsl(vslNewStream(&streams[batch], VSL_BRNG_MT19937, seed + batch));

#pragma omp parallel for
		for (int batch = 0; batch < batch_size; batch++)
		{
			for (int k = 0; k < rows[batch]; k++)
			{
				auto mid = std::lrint((float)cols[batch] * (1.0 - sparsity));
				auto row = &ptr[batch][k * strides[batch]];

				check_vsl(vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, streams[batch], mid, row, a, b));
				std::fill(row + mid, row + cols[batch], 0.0f);
				std::random_shuffle(row, row + cols[batch]);
				if (blank_diagonal && rows[batch] == cols[batch])
					row[k] = 0.0f;
			}
		}

		for (int batch = 0; batch < batch_size; batch++)
			check_vsl(vslDeleteStream(&streams[batch]));
	}
}

void TRN::CPU::Random::uniform(unsigned long &seed, float **ptr, const std::size_t &batch_size, const std::size_t *rows, const std::size_t *cols, const std::size_t *strides, const bool &blank_diagonal, const float &a, const float &b, const float &sparsity)
{
	TRN::CPU::Random::uniform_implementation( seed, ptr, batch_size, rows, cols, strides, blank_diagonal, a, b, sparsity);
	for (int batch = 0; batch < batch_size; batch++)
	{
		seed += rows[batch] * cols[batch];
	}
}

void TRN::CPU::Random::gaussian(unsigned long &seed, float **ptr, const std::size_t &batch_size, const std::size_t *rows, const std::size_t *cols, const std::size_t *strides, const bool &blank_diagonal, const float &mu, const float &sigma, const float &sparsity)
{
	if (sigma > 0.0f)
	{
		std::vector<VSLStreamStatePtr> streams(batch_size);

		for (int batch = 0; batch < batch_size; batch++)
			check_vsl(vslNewStream(&streams[batch], VSL_BRNG_MT19937, seed + batch));
#pragma omp parallel for
		for (int batch = 0; batch < batch_size; batch++)
		{
			for (int k = 0; k < rows[batch]; k++)
			{
				auto mid = std::lrint((float)cols[batch] * (1.0 - sparsity));
				auto row = &ptr[batch][k * strides[batch]];

				check_vsl(vsRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, streams[batch], mid, row, mu, sigma));
				std::fill(row + mid, row + cols[batch], 0.0f);
				std::random_shuffle(row, row + cols[batch]);

				if (blank_diagonal && rows[batch] == cols[batch])
					row[k] = 0.0f;
			}

		}
		for (int batch = 0; batch < batch_size; batch++)
		{
			check_vsl(vslDeleteStream(&streams[batch]));
			seed += rows[batch] * cols[batch];
		}
	}
}

std::shared_ptr<TRN::CPU::Random> TRN::CPU::Random::create()
{
	return std::make_shared<TRN::CPU::Random>();
}