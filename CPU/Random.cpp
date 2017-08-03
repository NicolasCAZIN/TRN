#include "stdafx.h"
#include "Random.h"



void TRN::CPU::Random::uniform_implementation(const unsigned long &seed, float **ptr, const std::size_t &batch_size, const std::size_t *rows, const std::size_t *cols, const std::size_t *strides, const float &a, const float &b, const float &sparsity)
{
	std::uniform_real_distribution<float> dist(a, b);
	auto seed_base = seed;

#pragma omp parallel for
	for (int batch = 0; batch < batch_size; batch++)
	{
		for (int k = 0; k < rows[batch]; k++)
		{
			auto mid = std::lrint((float)cols[batch] * (1.0 - sparsity));
			auto row = &ptr[batch][k * strides[batch]];
			std::default_random_engine random_engine(seed + k*cols[batch] + batch * cols[batch] * rows[batch]);
			auto gen = std::bind(dist, random_engine);
			std::generate(row, row + mid, gen);
			std::random_shuffle(row, row + cols[batch]);
		}
	}
}

void TRN::CPU::Random::uniform(unsigned long &seed, float **ptr, const std::size_t &batch_size, const std::size_t *rows, const std::size_t *cols, const std::size_t *strides, const float &a, const float &b, const float &sparsity)
{
	TRN::CPU::Random::uniform_implementation( seed, ptr, batch_size, rows, cols, strides, a, b, sparsity);
	for (int batch = 0; batch < batch_size; batch++)
	{
		seed += rows[batch] * cols[batch];
	}
}

void TRN::CPU::Random::gaussian(unsigned long &seed, float **ptr, const std::size_t &batch_size, const std::size_t *rows, const std::size_t *cols, const std::size_t *strides, const float &mu, const float &sigma)
{
	std::normal_distribution<float> dist(mu, sigma);

#pragma omp parallel for
	for (int batch = 0; batch < batch_size; batch++)
	{
		for (int k = 0; k < rows[batch]; k++)
		{
			auto row = &ptr[batch][k * strides[batch]];
			std::default_random_engine random_engine(seed + k*cols[batch] + batch * cols[batch] * rows[batch]);
			auto gen = std::bind(dist, random_engine);
			std::generate(row, row + cols[batch], gen);
		}
	}
	for (int batch = 0; batch < batch_size; batch++)
	{
		seed += rows[batch] * cols[batch];
	}
}

std::shared_ptr<TRN::CPU::Random> TRN::CPU::Random::create()
{
	return std::make_shared<TRN::CPU::Random>();
}