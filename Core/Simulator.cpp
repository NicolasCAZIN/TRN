#include "stdafx.h"
#include "Simulator.h"


TRN::Core::Simulator::Simulator()
{
	
}
TRN::Core::Simulator::~Simulator()
{

}

void TRN::Core::Simulator::initialize()
{
	auto reservoir = get_reservoir();
	if (!reservoir)
		throw std::runtime_error("Reservoir is not initialized");
	reservoir->TRN::Helper::Observable<TRN::Core::Message::Payload<TRN::Core::Message::TRAINED>>::attach(shared_from_this());
	reservoir->TRN::Helper::Observable<TRN::Core::Message::Payload<TRN::Core::Message::TESTED>>::attach(shared_from_this());
	reservoir->TRN::Helper::Observable<TRN::Core::Message::Payload<TRN::Core::Message::PRIMED>>::attach(shared_from_this());
	reservoir->TRN::Helper::Observable<TRN::Core::Message::Payload<TRN::Core::Message::CONFIGURED>>::attach(shared_from_this());

	auto mutators = get_mutators();
	if (!mutators.empty())
	{
		mutators[mutators.size() - 1]->attach(shared_from_this());
	}
	else
	{
		auto scheduler = get_scheduler();
		if (!scheduler)
			throw std::runtime_error("Scheduler is not initialized");
		scheduler->attach(shared_from_this());
	}

}

void TRN::Core::Simulator::uninitialize()
{
	
}