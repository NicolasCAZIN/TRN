#include "stdafx.h"
#include "Shuffle_impl.h"
#include "Core/Scheduling.h"

TRN::Mutator::Shuffle::Shuffle(const unsigned long &seed) :
	handle(std::make_unique<Handle>())
{
	handle->seed = seed;
}

TRN::Mutator::Shuffle::~Shuffle()
{
	handle.reset();
}

void TRN::Mutator::Shuffle::update(const TRN::Core::Message::Payload<TRN::Core::Message::SCHEDULING> &payload)
{
	std::default_random_engine engine(handle->seed);
	std::vector<std::vector<int>> indices;
	payload.get_scheduling()->to(indices);
	
	std::shuffle(std::begin(indices), std::end(indices), engine);
	handle->seed += payload.get_scheduling()->get_offsets().size() *  payload.get_scheduling()->get_durations().size();
	notify(TRN::Core::Message::Payload<TRN::Core::Message::SCHEDULING>(payload.get_evaluation_id(), TRN::Core::Scheduling::create(indices)));
}

std::shared_ptr<TRN::Mutator::Shuffle> TRN::Mutator::Shuffle::create(const unsigned long &seed)
{
	return std::make_shared<TRN::Mutator::Shuffle>(seed);
}