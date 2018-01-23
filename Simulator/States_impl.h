#pragma once

#include "States.h"

class TRN::Simulator::States::Handle
{
public:
	bool train;
	bool prime;
	bool generate;
	std::shared_ptr<TRN::Core::Message::Payload<TRN::Core::Message::STATES>> states;
	std::function<void(const unsigned long long &evaluation_id, const std::string &phase, const std::string &label, const std::size_t &batch, const std::vector<float> &data, const std::size_t &rows, const std::size_t &cols)> functor;
};
