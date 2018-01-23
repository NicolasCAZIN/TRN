#include "stdafx.h"
#include "Custom_impl.h"



TRN::Mutator::Custom::Custom(const unsigned long &seed, const std::function<void(const unsigned long long &evaluation_id, const unsigned long &seed,  const std::vector<int> &offsets, const std::vector<int> &durations)> &request, std::function<void(const std::size_t &trial, const std::vector<int> &offsets, const std::vector<int> &durations)> &reply) :
	handle(std::make_unique<Handle>())
{
	handle->seed = seed;
	handle->functor = request;
	reply = [&](const unsigned long long &evaluation_id, const std::vector<int> &offsets, const std::vector<int> &durations)
	{
		notify(TRN::Core::Message::Payload<TRN::Core::Message::SCHEDULING>(evaluation_id, TRN::Core::Scheduling::create(offsets, durations)));
	};
}
TRN::Mutator::Custom::~Custom()
{
	handle.reset();
}

		
void TRN::Mutator::Custom::update(const TRN::Core::Message::Payload<TRN::Core::Message::SCHEDULING> &payload)
{
	handle->functor(payload.get_evaluation_id(), handle->seed, payload.get_scheduling()->get_offsets(), payload.get_scheduling()->get_durations());
	handle->seed += payload.get_scheduling()->get_offsets().size() * payload.get_scheduling()->get_durations().size();
}

std::shared_ptr<TRN::Mutator::Custom> TRN::Mutator::Custom::create(const unsigned long &seed, const std::function<void(const unsigned long long &evaluation_id, const unsigned long &seed, const std::vector<int> &offsets, const std::vector<int> &durations)> &request, std::function<void(const unsigned long long &evaluation_id, const std::vector<int> &offsets, const std::vector<int> &durations)> &reply)
{
	return std::make_shared<TRN::Mutator::Custom>(seed, request, reply);
}
