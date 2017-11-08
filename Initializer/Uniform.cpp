#include "stdafx.h"
#include "Uniform.h"

class TRN::Initializer::Uniform::Handle
{
public:
	const float a;
	const float b;
	const float sparsity;

public:
	Handle( const float &a, const float &b, const float &sparsity) :
		a(a),
		b(b),
		sparsity(sparsity)
	{
	}
};

TRN::Initializer::Uniform::Uniform(const std::shared_ptr<TRN::Backend::Driver> &driver, const float &a, const float &b, const float &sparsity) :
	TRN::Core::Initializer(driver),
	handle(std::make_unique<TRN::Initializer::Uniform::Handle>( a, b, sparsity))
{
}

TRN::Initializer::Uniform::~Uniform()
{
	handle.reset();
}

void TRN::Initializer::Uniform::initialize(unsigned long &seed, std::shared_ptr<TRN::Core::Batch> &batch, const bool &blank_diagonal)
{
	implementor->get_random()->uniform(seed,  batch->get_elements(), batch->get_size(), batch->get_rows(), batch->get_cols(), batch->get_strides(), blank_diagonal, handle->a, handle->b, handle->sparsity);
}

std::shared_ptr<TRN::Initializer::Uniform> TRN::Initializer::Uniform::create(const std::shared_ptr<TRN::Backend::Driver> &driver, const float &a, const float &b, const float &sparsity)
{
	return std::make_shared<TRN::Initializer::Uniform>(driver,  a, b, sparsity);
}
