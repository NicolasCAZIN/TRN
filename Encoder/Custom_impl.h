#pragma once

#include "Custom.h"

struct TRN::Encoder::Custom::Handle 
{
	std::shared_ptr<TRN::Core::Batch> stimulus;
	std::shared_ptr<TRN::Core::Batch> position;
	std::function<void(const unsigned long long &evaluation_id, const std::vector<float> &stimulus, const std::size_t &rows, const std::size_t &cols)> on_predicted_position;
};

