#include "stdafx.h"
#include "FrechetDistance_impl.h"
#include "Helper/Logger.h"

TRN::Measurement::FrechetDistance::FrechetDistance(const std::shared_ptr<TRN::Backend::Driver> &driver, const std::function<void(const unsigned long long &evaluation_id, const std::vector<float> &values, const std::size_t &rows, const std::size_t &cols)> &functor) :
	TRN::Core::Measurement::Implementation(driver),
	handle(std::make_unique<Handle>())
{
	handle->functor = functor;
}


static float norm_2(const float *c_1, const float *c_2, const std::size_t &n_d, const std::size_t &i, const std::size_t &j)
{
	float sum = 0.0f;

	for (std::size_t col = 0; col < n_d; col++)
	{
		float diff = c_1[(i - 1)*n_d + col] - c_2[(j - 1)*n_d + col];

		sum += diff*diff;
	}

	return std::sqrtf(sum);
}


static float recursive_c(float *ca, const std::size_t &n_d,
	const std::size_t &n_1, const std::size_t &n_2,
	const float *c_1, const float *c_2, const std::size_t i, const std::size_t j)
{
	float *ca_ij;
	ca_ij = ca + (i - 1)*n_2 + (j - 1);


	if (*ca_ij > -1.0)
	{
		return *ca_ij;
	}
	else if ((i == 1) && (j == 1))
	{
		*ca_ij = norm_2(c_1, c_2, n_d, 1, 1);
	}
	else if ((i > 1) && (j == 1))
	{
		*ca_ij = std::max(recursive_c(ca, n_d, n_1, n_2, c_1, c_2, i - 1, 1), norm_2(c_1, c_2, n_d, i, 1));
	}
	else if ((i == 1) && (j > 1))
	{
		*ca_ij = std::max(recursive_c(ca, n_d, n_1, n_2, c_1, c_2, 1, j - 1), norm_2(c_1, c_2, n_d, 1, j));
	}
	else if ((i > 1) && (j > 1))
	{
		*ca_ij = std::max(
			std::min(
				std::min(
					recursive_c(ca, n_d, n_1, n_2, c_1, c_2, i - 1, j),
					recursive_c(ca, n_d, n_1, n_2, c_1, c_2, i - 1, j - 1)),
					recursive_c(ca, n_d, n_1, n_2, c_1, c_2, i, j - 1)),
			norm_2(c_1, c_2, n_d, i, j));
	}
	else
	{
		*ca_ij = -std::numeric_limits<float>::max();
	}

	return *ca_ij;
}

static float discrete_frechet_distance(
	float *ca, const std::size_t &cols,
	const float *predicted, const std::size_t &predicted_rows,
	const float *expected, const std::size_t &expected_rows)
{
	return recursive_c(ca, cols, predicted_rows, expected_rows, 
		predicted, expected, predicted_rows, expected_rows);

}


void TRN::Measurement::FrechetDistance::compute(const unsigned long long &evaluation_id,
	const std::shared_ptr<TRN::Core::Matrix> &primed, 
	const std::shared_ptr<TRN::Core::Batch> &predicted, 
	const std::shared_ptr<TRN::Core::Matrix> &expected, 
	const std::shared_ptr<TRN::Core::Matrix> &error)
{
	DEBUG_LOGGER << "Using non vectorized fallback";
	std::vector<float> expected_values;
	std::size_t expected_rows;
	std::size_t expected_cols;
	expected->to(expected_values, expected_rows, expected_cols);

	std::vector<float> predicted_values;
	std::vector<std::size_t> predicted_rows;
	std::vector<std::size_t> predicted_cols;
	std::size_t batch_size;
	predicted->to(predicted_values, batch_size, predicted_rows, predicted_cols);

	std::vector<float> error_values;
	std::size_t error_rows;
	std::size_t error_cols;
	error->to(error_values, error_rows, error_cols);
#pragma omp parallel for
	for (int batch = 0; batch < batch_size; batch++)
	{
		std::vector<float> ca(expected_rows * predicted_rows[batch]);
		std::fill(ca.begin(), ca.end(), -1.0f);
		std::size_t offset = 0;
		if (batch > 0)
			offset  += batch * predicted_rows[batch-1] * predicted_cols[batch-1];
		error_values[batch] = discrete_frechet_distance(
			ca.data(), expected_cols,

			
			&predicted_values[offset], predicted_rows[batch],
			expected_values.data(), expected_rows);
	}


	handle->functor(evaluation_id, error_values, error_rows, error_cols);
}

std::shared_ptr<TRN::Measurement::FrechetDistance> TRN::Measurement::FrechetDistance::create(const std::shared_ptr<TRN::Backend::Driver> &driver, const std::function<void(const unsigned long long &evaluation_id, const std::vector<float> &values, const std::size_t &rows, const std::size_t &cols)> &functor)
{
	return std::make_shared<TRN::Measurement::FrechetDistance>(driver, functor);
}