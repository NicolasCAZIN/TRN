#pragma once

#include "Basic.h"

class TRN::Simulator::Basic::Handle
{
public:
	std::map<std::string, std::shared_ptr<TRN::Core::Matrix>> sequences;
	std::map<std::string, std::shared_ptr<TRN::Core::Set>> sets;
	std::shared_ptr<TRN::Core::Initializer> feedforward;
	std::shared_ptr<TRN::Core::Initializer> recurrent;
	std::shared_ptr<TRN::Core::Initializer> feedback;
	std::shared_ptr<TRN::Core::Initializer> readout;
	std::shared_ptr<TRN::Core::Reservoir> reservoir;
	std::shared_ptr<TRN::Core::Loop> loop;
	std::shared_ptr<TRN::Core::Scheduler> scheduler;
	std::queue<TRN::Core::Message::Payload<TRN::Core::Message::SET>> pending;
	std::function<void()> trained;
	std::function<void()> primed;
	std::function<void()> tested;
	std::list<std::shared_ptr<TRN::Core::Measurement::Abstraction>> measurements;

	bool initialized;
};