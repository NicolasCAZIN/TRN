#include "stdafx.h"
#include "Shuffle.h"
#include "Core/Scheduling.h"

void TRN::Mutator::Shuffle::update(const TRN::Core::Message::Payload<TRN::Core::Message::SCHEDULING> &payload)
{
	std::vector<std::vector<int>> indices;

	payload.get_scheduling()->to(indices);
	std::random_shuffle(std::begin(indices), std::end(indices));

	notify(TRN::Core::Message::Payload<TRN::Core::Message::SCHEDULING>(TRN::Core::Scheduling::create(indices)));
}

std::shared_ptr<TRN::Mutator::Shuffle> TRN::Mutator::Shuffle::create()
{
	return std::make_shared<TRN::Mutator::Shuffle>();
}