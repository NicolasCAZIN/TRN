#include "stdafx.h"
#include "Gaussian.h"

class TRN::Initializer::Gaussian::Handle
{
public:
	const float mu;
	const float sigma;
	const float sparsity;
public:
	Handle(const float &mu, const float &sigma, const float &sparsity) :
		mu(mu),
		sigma(sigma),
		sparsity(sparsity)
	{
	}
};

TRN::Initializer::Gaussian::Gaussian(const std::shared_ptr<TRN::Backend::Driver> &driver,const float &mu, const float &sigma, const float &sparsity) :
	TRN::Core::Initializer(driver),
	handle(std::make_unique<Handle>(mu, sigma, sparsity))
{
}

TRN::Initializer::Gaussian::~Gaussian()
{
	handle.reset();
}

void TRN::Initializer::Gaussian::initialize(unsigned long &seed, std::shared_ptr<TRN::Core::Batch> &batch, const bool &blank_diagonal)
{
	implementor->get_random()->gaussian(seed, batch->get_elements(), batch->get_size(), batch->get_rows(), batch->get_cols(), batch->get_strides(), blank_diagonal, handle->mu, handle->sigma, handle->sparsity);
}

std::shared_ptr<TRN::Initializer::Gaussian> TRN::Initializer::Gaussian::create(const std::shared_ptr<TRN::Backend::Driver> &driver, const float &mu, const float &sigma, const float &sparsity)
{
	return std::make_shared<TRN::Initializer::Gaussian>(driver, mu, sigma, sparsity);
}
