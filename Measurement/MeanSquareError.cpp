#include "stdafx.h"
#include "MeanSquareError_impl.h"

TRN::Measurement::MeanSquareError::MeanSquareError(const std::shared_ptr<TRN::Backend::Driver> &driver, const std::function<void(const unsigned long long &evaluation_id, const std::vector<float> &values, const std::size_t &rows, const std::size_t &cols)> &functor) :
	TRN::Core::Measurement::Implementation(driver),
	handle(std::make_unique<Handle>())
{
	handle->functor = functor;
}

void TRN::Measurement::MeanSquareError::compute(const unsigned long long &evaluation_id, const std::shared_ptr<TRN::Core::Matrix> &primed, const std::shared_ptr<TRN::Core::Batch> &predicted, const std::shared_ptr<TRN::Core::Matrix> &expected, const std::shared_ptr<TRN::Core::Matrix> &error)
{
	std::vector<float> error_values;
	std::size_t error_rows;
	std::size_t error_cols;
	
	implementor->get_algorithm()->mean_square_error(
		error->get_cols(),
		(const float **)predicted->get_elements(), predicted->get_rows(), predicted->get_cols(), predicted->get_strides(),
		expected->get_elements(), expected->get_rows(), expected->get_cols(), expected->get_stride(),
		error->get_elements(), error->get_rows(), error->get_cols(), error->get_stride()
	);



	error->to(error_values, error_rows, error_cols);
	handle->functor(evaluation_id, error_values, error_rows, error_cols);
}

std::shared_ptr<TRN::Measurement::MeanSquareError> TRN::Measurement::MeanSquareError::create(const std::shared_ptr<TRN::Backend::Driver> &driver, const std::function<void(const unsigned long long &evaluation_id, const std::vector<float> &values, const std::size_t &rows, const std::size_t &cols)> &functor)
{
	return std::make_shared<TRN::Measurement::MeanSquareError>(driver, functor);
}