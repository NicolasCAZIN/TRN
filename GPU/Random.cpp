#include "stdafx.h"
#include "Random_impl.h"
#include "Random.cuh"


TRN::GPU::Random::Random(const std::shared_ptr<TRN::GPU::Context> context) :
	handle(std::make_unique<Handle>())
{
	handle->context = context;

}

TRN::GPU::Random::~Random()
{
	handle.reset();
}



void TRN::GPU::Random::uniform(unsigned long &seed, float **ptr, const std::size_t &batch_size, const std::size_t *rows, const std::size_t *cols, const std::size_t *strides, const float &a, const float &b, const float &sparsity)
{
	random_uniform(handle->context->get_stream(), seed, a, b, sparsity, batch_size, *rows, *cols, ptr, *strides);

	for (int batch = 0; batch < batch_size; batch++)
	{
		seed += rows[batch] * cols[batch];
	}
}
void TRN::GPU::Random::gaussian(unsigned long &seed, float **ptr, const std::size_t &batch_size, const std::size_t *rows, const std::size_t *cols, const std::size_t *strides, const float &mu, const float &sigma)
{
	random_gaussian(handle->context->get_stream(), seed, mu, sigma, batch_size, *rows, *cols, ptr, *strides);
	
	for (int batch = 0; batch < batch_size; batch++)
	{
		seed += rows[batch] * cols[batch];
	}
}

std::shared_ptr<TRN::GPU::Random> TRN::GPU::Random::create(const std::shared_ptr<TRN::GPU::Context> context)
{
	return std::make_shared<TRN::GPU::Random>(context);
}