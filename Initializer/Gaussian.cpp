#include "stdafx.h"
#include "Gaussian.h"

class TRN::Initializer::Gaussian::Handle
{
public:
	const float mu;
	const float sigma;

public:
	Handle(const float &mu, const float &sigma) :
		mu(mu),
		sigma(sigma)
	{
	}
};

TRN::Initializer::Gaussian::Gaussian(const std::shared_ptr<TRN::Backend::Driver> &driver,const float &mu, const float &sigma) :
	TRN::Core::Initializer(driver),
	handle(std::make_unique<Handle>(mu, sigma))
{
}

TRN::Initializer::Gaussian::~Gaussian()
{
	handle.reset();
}

void TRN::Initializer::Gaussian::initialize(unsigned long &seed, std::shared_ptr<TRN::Core::Batch> &batch)
{
	implementor->get_random()->gaussian(seed, batch->get_elements(), batch->get_size(), batch->get_rows(), batch->get_cols(), batch->get_strides(), handle->mu, handle->sigma);
}

std::shared_ptr<TRN::Initializer::Gaussian> TRN::Initializer::Gaussian::create(const std::shared_ptr<TRN::Backend::Driver> &driver, const float &mu, const float &sigma)
{
	return std::make_shared<TRN::Initializer::Gaussian>(driver, mu, sigma);
}
