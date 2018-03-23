#include "stdafx.h"
#include "Reservoir.h"
#include "Reservoir/WidrowHoff.h"


std::shared_ptr<TRN::Core::Reservoir> TRN::Model::Reservoir::WidrowHoff::create(const std::shared_ptr<TRN::Backend::Driver> &driver,
	const std::size_t &stimulus_size, const std::size_t &prediction_size, const std::size_t &reservoir_size, const float &leak_rate,
	const float &initial_state_scale, const float &learning_rate, const unsigned long &seed, const std::size_t &batch_size, const std::size_t &mini_batch_size)
{
	return TRN::Reservoir::WidrowHoff::create(driver, stimulus_size, prediction_size,reservoir_size, leak_rate, initial_state_scale, learning_rate, seed, batch_size, mini_batch_size);
}
			
