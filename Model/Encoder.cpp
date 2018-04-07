#include "stdafx.h"
#include "Encoder.h"
#include "Encoder/Model.h"
#include "Encoder/Custom.h"

std::shared_ptr<TRN::Core::Encoder>   TRN::Model::Encoder::Model::create(const std::shared_ptr<TRN::Backend::Driver> &driver,
	const std::size_t &batch_size, const std::size_t &stimulus_size,
	const std::shared_ptr<TRN::Core::Matrix> &cx,
	const std::shared_ptr<TRN::Core::Matrix> &cy,
	const std::shared_ptr<TRN::Core::Matrix> &K)
{
	return TRN::Encoder::Model::create(driver, batch_size, stimulus_size, cx, cy, K);
}

std::shared_ptr<TRN::Core::Encoder>   TRN::Model::Encoder::Custom::create(const std::shared_ptr<TRN::Backend::Driver> &driver,
	const std::size_t &batch_size, const std::size_t &stimulus_size,
	const std::function<void(const unsigned long long &evaluation_id, const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols)> &predicted_position,
	std::function<void(const unsigned long long &evaluation_id, const std::vector<float> &stimulus, const std::size_t &rows, const std::size_t &cols)> &estimated_position,
	std::function<void(const unsigned long long &evaluation_id, const std::vector<float> &stimulus, const std::size_t &rows, const std::size_t &cols)> &perceived_stimulus)
{
	return TRN::Encoder::Custom::create(driver, batch_size, stimulus_size, predicted_position, estimated_position, perceived_stimulus);
}