#include "stdafx.h"
#include "FrechetDistance_impl.h"

TRN::Measurement::FrechetDistance::FrechetDistance(const std::shared_ptr<TRN::Backend::Driver> &driver, const std::function<void(const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const std::size_t &cols)> &functor) :
	TRN::Core::Measurement::Implementation(driver),
	handle(std::make_unique<Handle>())
{
	handle->functor = functor;
}


void TRN::Measurement::FrechetDistance::compute(const std::size_t &trial, const std::size_t &evaluation, const std::shared_ptr<TRN::Core::Matrix> &primed, const std::shared_ptr<TRN::Core::Batch> &predicted, const std::shared_ptr<TRN::Core::Matrix> &expected, const std::shared_ptr<TRN::Core::Matrix> &error)
{
	

	std::vector<float> values;
	std::size_t rows;
	std::size_t cols;

	error->to(values, rows, cols);
	handle->functor(trial, evaluation, values, rows, cols);
}

std::shared_ptr<TRN::Measurement::FrechetDistance> TRN::Measurement::FrechetDistance::create(const std::shared_ptr<TRN::Backend::Driver> &driver, const std::function<void(const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const std::size_t &cols)> &functor)
{
	return std::make_shared<TRN::Measurement::FrechetDistance>(driver, functor);
}