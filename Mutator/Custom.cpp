#include "stdafx.h"
#include "Custom_impl.h"



TRN::Mutator::Custom::Custom(const std::function<void(const std::vector<int> &offsets, const std::vector<int> &durations)> &request, std::function<void(const std::vector<int> &offsets, const std::vector<int> &durations)> &reply) :
	handle(std::make_unique<Handle>())
{
	handle->functor = request;
	reply = [&](const std::vector<int> &offsets, const std::vector<int> &durations)
	{
		notify(TRN::Core::Message::Payload<TRN::Core::Message::SCHEDULING>(TRN::Core::Scheduling::create(offsets, durations)));
	};
}
TRN::Mutator::Custom::~Custom()
{
	handle.reset();
}

		
void TRN::Mutator::Custom::update(const TRN::Core::Message::Payload<TRN::Core::Message::SCHEDULING> &payload)
{
	handle->functor(payload.get_scheduling()->get_offsets(), payload.get_scheduling()->get_durations());
}

std::shared_ptr<TRN::Mutator::Custom> TRN::Mutator::Custom::create(const std::function<void(const std::vector<int> &offsets, const std::vector<int> &durations)> &request, std::function<void(const std::vector<int> &offsets, const std::vector<int> &durations)> &reply)
{
	return std::make_shared<TRN::Mutator::Custom>(request, reply);
}
