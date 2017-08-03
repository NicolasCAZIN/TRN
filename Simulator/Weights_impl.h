#pragma once

#include "Weights.h"

class TRN::Simulator::Weights::Handle
{
public:
	bool train;
	bool initialization;
	std::shared_ptr<TRN::Core::Message::Payload<TRN::Core::Message::WEIGHTS>> weights;
	std::function<void(const std::string &phase, const std::string &label, const std::vector<float> &data, const std::size_t &rows, const std::size_t &cols)> functor;
};
