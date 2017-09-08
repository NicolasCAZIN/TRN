#include "stdafx.h"
#include "Punch_impl.h"

TRN::Mutator::Punch::Punch(const unsigned long &seed, const float &rate, const std::size_t &size, const std::size_t &number) :
	handle(std::make_unique<Handle>())
{
	handle->rate = rate;
	handle->size = size;
	handle->number = number;
	handle->seed = seed;
}

TRN::Mutator::Punch::~Punch()
{
	handle.reset();
}



void TRN::Mutator::Punch::update(const TRN::Core::Message::Payload<TRN::Core::Message::SCHEDULING> &payload)
{
	std::default_random_engine engine(handle->seed);
	std::vector<std::vector<int>> indices;
	std::uniform_real_distribution<float> rate_distribution(0.0f, 1.0f);

	auto rate_dice = std::bind(rate_distribution, engine);
	payload.get_scheduling()->to(indices);
	
	std::for_each(std::begin(indices), std::end(indices), [&](std::vector<int> &v)
	{
		if (rate_dice() < handle->rate)
		{
			if (handle->size > v.size())
				throw std::invalid_argument("size of sequence must be <= to " + std::to_string(handle->size));
			std::uniform_int_distribution<std::size_t> rate_distribution(0, v.size() - handle->size);
			for (std::size_t n = 0; n < handle->number; n++)
			{
				auto offset = rate_distribution(engine);

				std::transform(std::begin(v) + offset, std::begin(v) + offset + handle->size, std::begin(v) + offset, [](const int &x) { return -std::abs(x); });

			}
		}
	});
	handle->seed += payload.get_scheduling()->get_offsets().size() *  payload.get_scheduling()->get_durations().size();

	notify(TRN::Core::Message::Payload<TRN::Core::Message::SCHEDULING>(payload.get_trial(), TRN::Core::Scheduling::create(indices)));
}

std::shared_ptr<TRN::Mutator::Punch> TRN::Mutator::Punch::create(const unsigned long &seed, const float &rate, const std::size_t &size, const std::size_t &number)
{
	return std::make_shared<TRN::Mutator::Punch>(seed, rate, size, number);
}