#include "stdafx.h"
#include "Reverse_impl.h"

TRN::Mutator::Reverse::Reverse(const float &rate, const std::size_t &size) :
	handle(std::make_unique<Handle>())
{
	handle->rate = rate;
	handle->size = size;
}

TRN::Mutator::Reverse::~Reverse()
{
	handle.reset();
}



void TRN::Mutator::Reverse::update(const TRN::Core::Message::Payload<TRN::Core::Message::SCHEDULING> &payload)
{
	std::vector<std::vector<int>> indices;
	std::default_random_engine engine;
	std::uniform_real_distribution<float> rate_distribution(0.0f, 1.0f);

	auto rate_dice = std::bind(rate_distribution, engine);
	payload.get_scheduling()->to(indices);
	std::for_each(std::begin(indices), std::end(indices), [&](std::vector<int> &v) 
	{
		if (rate_dice() < handle->rate)
		{
			if (handle->size > v.size())
				throw std::invalid_argument("size of sequence must be <= to " + std::to_string(handle->size));
			std::uniform_int_distribution<std::size_t> offset_distribution(0, v.size() - handle->size);

			auto offset = offset_distribution(engine);

			std::reverse(std::begin(v) + offset, std::begin(v) + offset + handle->size);
		}
	});

	notify(TRN::Core::Message::Payload<TRN::Core::Message::SCHEDULING>(TRN::Core::Scheduling::create(indices)));
}

std::shared_ptr<TRN::Mutator::Reverse> TRN::Mutator::Reverse::create(const float &rate, const std::size_t &size)
{
	return std::make_shared<TRN::Mutator::Reverse>(rate, size);
}