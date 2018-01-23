#include "stdafx.h"
#include "Snippets.h"

class TRN::Scheduler::Snippets::Handle
{
public :
	unsigned int snippets_size;
	unsigned int time_budget;
	std::default_random_engine random_engine;
	std::string tag;
};

TRN::Scheduler::Snippets::Snippets(const unsigned long &seed, const unsigned int &snippets_size, const unsigned int &time_budget, const std::string &tag) :
	handle(std::make_unique<TRN::Scheduler::Snippets::Handle>())
{
	handle->snippets_size = snippets_size;
	handle->time_budget = time_budget;
	handle->tag = tag;
	handle->random_engine = std::default_random_engine(seed);
}

TRN::Scheduler::Snippets::~Snippets()
{
	handle.reset();
}

void TRN::Scheduler::Snippets::update(const TRN::Core::Message::Payload<TRN::Core::Message::SET> &payload)
{

	std::shared_ptr<TRN::Core::Set> set;
	std::vector<int> score;
	std::vector<int> batch_offsets, batch_durations;

	std::vector<float> reward_elements;

	if (handle->tag.empty())
	{
		set = delegate.lock()->retrieve_set(payload.get_label(), payload.get_incoming());
		set->get_scheduling()->to(batch_offsets, batch_durations);
		reward_elements.resize(batch_offsets.size());
		std::fill(reward_elements.begin(), reward_elements.end(), 0.0f);
		std::size_t b = 0;
	
		for (std::size_t k = 0; k < batch_durations.size(); k++)
		{
			auto e = b + batch_durations[k] - 1;
			reward_elements[b] = 1.0f;
			reward_elements[e] = 1.0f;

			b = e + 1;
		}

	}
	else
	{
		std::size_t reward_rows;
		std::size_t reward_cols;

		set = delegate.lock()->retrieve_set(payload.get_label(), handle->tag);
		set->get_sequence()->to(reward_elements, reward_rows, reward_cols);
		set->get_scheduling()->to(batch_offsets, batch_durations);

	

	}
	/*std::size_t b = 0;
	for (std::size_t k = 0; k < batch_durations.size(); k++)
	{
		auto e = b + batch_durations[k] - 1;

	}*/
	std::vector<std::vector<float>> reward_sites;
	std::vector<std::vector<int>> relative_offsets;

	std::size_t b = 0;

	for (std::size_t k = 0; k < batch_durations.size(); k++)
	{
		/*auto e = b + batch_durations[k]-1;
		reward_elements[b] = 1.0f;
		reward_elements[e] = 1.0f;*/
		std::vector<float> batch_reward(batch_durations[k]);
		std::vector<int> batch_offset(batch_durations[k]);

		std::copy(reward_elements.begin() + b, reward_elements.begin() + b + batch_durations[k], batch_reward.begin());
		std::copy(batch_offsets.begin() + b, batch_offsets.begin() + b + batch_durations[k], batch_offset.begin());


		reward_sites.push_back(batch_reward);
		relative_offsets.push_back(batch_offset);
		b += batch_durations[k];
	}

	auto sequence = std::uniform_int<std::size_t>(0, reward_sites.size() -1);

	std::vector<int> offsets;
	std::vector<int> durations;

	while (offsets.size() < handle->time_budget)
	{
		std::size_t s = sequence(handle->random_engine);
		std::discrete_distribution<std::size_t> d(reward_sites[s].begin(), reward_sites[s].end());
		std::size_t r0 = d(handle->random_engine);
		std::size_t r1 = r0;
		while (r0 == r1)
		{
			r1 = d(handle->random_engine);
		} 

		if (r0 > r1)
			std::swap(r0, r1);
		std::uniform_int<std::size_t> offset(r0, r1);

		std::size_t o = offset(handle->random_engine);
		auto duration = handle->snippets_size;
		if (r1 - r0 < handle->snippets_size)
		{
			unsigned int t = reward_sites[s].size() - o;
			duration = std::min(t + 1, handle->snippets_size);
		}
		else
		{
			while (o + duration >= reward_sites[s].size())
			{
				o = offset(handle->random_engine);
			}
			
		}
	
		int extra_steps = offsets.size() + duration - handle->time_budget;
		if (extra_steps > 0)
		{
			if (extra_steps >= handle->snippets_size)
				throw std::runtime_error("");
			duration -= extra_steps;
		}
		

		std::vector<int> snippet(duration);
		std::iota(snippet.begin(), snippet.end(), relative_offsets[s][o]);

		durations.push_back(static_cast<int>(snippet.size()));
		offsets.insert(offsets.begin(), snippet.begin(), snippet.end());

	}
	assert(handle->time_budget == offsets.size());
	notify(TRN::Core::Message::Payload<TRN::Core::Message::SCHEDULING>(payload.get_evaluation_id(), TRN::Core::Scheduling::create(offsets, durations)));
}

std::shared_ptr<TRN::Scheduler::Snippets> TRN::Scheduler::Snippets::create(const unsigned long &seed, const unsigned int  &snippets_size, const unsigned int &time_budget, const std::string &tag)
{
	return std::make_shared<TRN::Scheduler::Snippets>(seed, snippets_size, time_budget, tag);
}