#include "stdafx.h"
#include "Initializer.h"
#include "Initializer/Gaussian.h"
#include "Initializer/Uniform.h"
#include "Initializer/Custom.h"

std::shared_ptr<TRN::Core::Initializer> TRN::Model::Initializer::Uniform::create(const std::shared_ptr<TRN::Backend::Driver> &driver,  const float &a, const float &b, const float &sparsity)
{
	return TRN::Initializer::Uniform::create(driver, a, b, sparsity);
}

std::shared_ptr<TRN::Core::Initializer> TRN::Model::Initializer::Gaussian::create(const std::shared_ptr<TRN::Backend::Driver> &driver,  const float &mu, const float &sigma, const float &sparsity)
{
	return TRN::Initializer::Gaussian::create(driver, mu, sigma, sparsity);
}

std::shared_ptr<TRN::Core::Initializer> TRN::Model::Initializer::Custom::create(const std::shared_ptr<TRN::Backend::Driver> &driver,
	const std::function<void(const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &request,
	std::function<void(const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &reply)
{
	return TRN::Initializer::Custom::create(driver, request, reply);
}