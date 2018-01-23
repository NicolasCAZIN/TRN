#include "stdafx.h"
#include "Loop.h"
#include "Loop/Copy.h"
#include "Loop/SpatialFilter.h"
#include "Loop/Custom.h"

std::shared_ptr<TRN::Core::Loop> TRN::Model::Loop::Copy::create(const std::shared_ptr<TRN::Backend::Driver> &driver, const std::size_t &batch_size, const std::size_t &stimulus_size)
{
	return TRN::Loop::Copy::create(driver, batch_size, stimulus_size);
}

std::shared_ptr<TRN::Core::Loop> TRN::Model::Loop::SpatialFilter::create(const std::shared_ptr<TRN::Backend::Driver> &driver, const std::size_t &batch_size, const std::size_t &stimulus_size, const unsigned long &seed,
	const std::function<void(const unsigned long long &evaluation_id, const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols)> &predicted_position,
	std::function<void(const unsigned long long &evaluation_id, const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols)> &estimated_position,
	const std::function<void(const unsigned long long &evaluation_id, const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols)> &predicted_stimulus,
	std::function<void(const unsigned long long &evaluation_id, const std::vector<float> &stimulus, const std::size_t &rows, const std::size_t &cols)> &perceived_stimulus,
	const std::size_t &rows, const std::size_t &cols,
	const std::pair<float, float> &x, const std::pair<float, float> &y,
	const std::shared_ptr<TRN::Core::Matrix> &firing_rate_map,
	const float &sigma,
	const float &radius,
	const float &scale,
	const std::string &tag)
{
	return TRN::Loop::SpatialFilter::create(driver, batch_size, stimulus_size, seed, predicted_position, estimated_position, predicted_stimulus, perceived_stimulus, rows, cols, x, y, firing_rate_map, sigma, radius,scale, tag);
}

std::shared_ptr<TRN::Core::Loop> TRN::Model::Loop::Custom::create(const std::shared_ptr<TRN::Backend::Driver> &driver, const std::size_t &batch_size, const std::size_t &stimulus_size,
	const std::function<void(const unsigned long long &evaluation_id, const std::vector<float> &prediction, const std::size_t &rows, const std::size_t &cols)> &request,
	std::function<void(const unsigned long long &evaluation_id, const std::vector<float> &stimulus, const std::size_t &rows, const std::size_t &cols)> &reply
	)
{
	return TRN::Loop::Custom::create(driver, batch_size, stimulus_size, request, reply);
}