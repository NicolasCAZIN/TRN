#include "stdafx.h"
#include "Reverse_impl.h"

TRN::Mutator::Reverse::Reverse(const unsigned long &seed, const float &rate, const std::size_t &size) :
	handle(std::make_unique<Handle>())
{
	handle->rate = rate;
	handle->size = size;
	handle->seed = seed;
}

TRN::Mutator::Reverse::~Reverse()
{
	handle.reset();
}



void TRN::Mutator::Reverse::update(const TRN::Core::Message::Payload<TRN::Core::Message::SCHEDULING> &payload)
{
	std::vector<std::vector<int>> indices;
	std::default_random_engine engine(handle->seed);
	std::uniform_real_distribution<float> rate_distribution(0.0f, 1.0f);

	auto rate_dice = std::bind(rate_distribution, engine);
	payload.get_scheduling()->to(indices);
	std::for_each(std::begin(indices), std::end(indices), [&](std::vector<int> &v) 
	{
		if (rate_dice() <= handle->rate)
		{
			auto size = std::min(handle->size, v.size());
			std::uniform_int_distribution<std::size_t> offset_distribution(0, v.size() - size);

			auto offset = offset_distribution(engine);

			std::reverse(std::begin(v) + offset, std::begin(v) + offset + size);
		}
	});
	handle->seed += payload.get_scheduling()->get_offsets().size() *  payload.get_scheduling()->get_durations().size();

	notify(TRN::Core::Message::Payload<TRN::Core::Message::SCHEDULING>(payload.get_evaluation_id(), TRN::Core::Scheduling::create(indices)));
}

std::shared_ptr<TRN::Mutator::Reverse> TRN::Mutator::Reverse::create(const unsigned long &seed, const float &rate, const std::size_t &size)
{
	return std::make_shared<TRN::Mutator::Reverse>(seed, rate, size);
}