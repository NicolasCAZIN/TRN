#include "stdafx.h"
#include "Decoder.h"
#include "Decoder/Linear.h"
#include "Decoder/Map.h"
#include "Decoder/Model.h"

std::shared_ptr<TRN::Core::Decoder> TRN::Model::Decoder::Linear::create(const std::shared_ptr<TRN::Backend::Driver> &driver, const std::size_t &batch_size, const std::size_t &stimulus_size, const std::shared_ptr<TRN::Core::Matrix> &cx, const std::shared_ptr<TRN::Core::Matrix> &cy)
{
	return TRN::Decoder::Linear::create(driver, batch_size, stimulus_size, cx, cy);
}

std::shared_ptr<TRN::Core::Decoder>   TRN::Model::Decoder::Kernel::Model::create(const std::shared_ptr<TRN::Backend::Driver> &driver,
		const std::size_t &batch_size, const std::size_t &stimulus_size,
		const std::size_t &rows, const std::size_t &cols,
		const std::pair<float, float> &arena_x, const std::pair<float, float> &arena_y,
		const float &sigma, const float &radius, const float &angle, const float &scale, const unsigned long &seed,
		const std::shared_ptr<TRN::Core::Matrix> &cx,
		const std::shared_ptr<TRN::Core::Matrix> &cy,
		const std::shared_ptr<TRN::Core::Matrix> &K)
{
	return TRN::Decoder::Model::create(driver, batch_size, stimulus_size, rows, cols, arena_x, arena_y, sigma, radius, angle, scale, seed,  cx, cy, K);
}


std::shared_ptr<TRN::Core::Decoder>   TRN::Model::Decoder::Kernel::Map::create(const std::shared_ptr<TRN::Backend::Driver> &driver,
	const std::size_t &batch_size, const std::size_t &stimulus_size,
	const std::size_t &rows, const std::size_t &cols,
	const std::pair<float, float> &arena_x, const std::pair<float, float> &arena_y,
	const float &sigma, const float &radius, const float &angle, const float &scale, const unsigned long &seed,
	const std::shared_ptr<TRN::Core::Matrix> &firing_rate_map)
{
	return TRN::Decoder::Map::create(driver, batch_size, stimulus_size, rows, cols, arena_x, arena_y, sigma, radius, angle, scale, seed, firing_rate_map);
}
