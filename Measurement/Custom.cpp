#include "stdafx.h"

#include "Custom_impl.h"
#include "Core/Measurement_impl.h"

TRN::Measurement::Custom::Custom(const std::shared_ptr<TRN::Backend::Driver> &driver, const std::function<void(const std::vector<float> &predicted, const std::vector<float> &expected, const std::size_t &pages, const std::size_t &rows, const std::size_t &cols)> &functor) :
	TRN::Core::Measurement::Implementation(driver),
	handle(std::make_unique<Handle>())
{
	handle->functor = functor;
}

void TRN::Measurement::Custom::compute(const std::shared_ptr<TRN::Core::Batch> &predicted, const std::shared_ptr<TRN::Core::Matrix> &expected, const std::shared_ptr<TRN::Core::Matrix> &error)
{
	std::vector<float> predicted_values;
	std::size_t batch_size;
	std::vector<std::size_t> predicted_rows;
	std::vector<std::size_t> predicted_cols;

	std::vector<float> expected_values;
	std::size_t expected_rows;
	std::size_t expected_cols;

	expected->to(expected_values, expected_rows, expected_cols);
	predicted->to(predicted_values, batch_size, predicted_rows, predicted_cols);
	

	handle->functor(predicted_values, expected_values, batch_size, expected_rows, expected_cols);
}

std::shared_ptr<TRN::Measurement::Custom> TRN::Measurement::Custom::create(const std::shared_ptr<TRN::Backend::Driver> &driver, const std::function<void(const std::vector<float> &predicted, const std::vector<float> &expected, const std::size_t &pages, const std::size_t &rows, const std::size_t &cols)> &functor)
{
	return std::make_shared<TRN::Measurement::Custom>(driver, functor);
}