#include "stdafx.h"
#include "Snippets.h"

class TRN::Scheduler::Snippets::Handle
{
public :
	mutable unsigned int snippets_size;
	mutable unsigned int time_budget;
	mutable std::string tag;
};

TRN::Scheduler::Snippets::Snippets(const std::shared_ptr<TRN::Backend::Driver> &driver, const unsigned int &snippets_size, const unsigned int &time_budget, const std::string &tag) :
	TRN::Core::Scheduler(driver),
	handle(std::make_unique<TRN::Scheduler::Snippets::Handle>())
{
	handle->snippets_size = snippets_size;
	handle->time_budget = time_budget;
	handle->tag = tag;
}

TRN::Scheduler::Snippets::~Snippets()
{
	handle.reset();
}

void TRN::Scheduler::Snippets::update(const TRN::Core::Message::Payload<TRN::Core::Message::SET> &payload)
{
	auto snippets_number = handle->time_budget / handle->snippets_size;
	auto remaining_time = handle->time_budget % handle->snippets_size;
	std::vector<unsigned int> durations(snippets_number);
	std::fill(durations.begin(), durations.end(), (unsigned int)handle->snippets_size);
	if (remaining_time)
	{
		durations.push_back(remaining_time);
		snippets_number++;
	}
	std::shared_ptr<TRN::Core::Set> set;
	std::vector<unsigned int> score;
	std::vector<unsigned int> batch_offsets, batch_durations;

	if (handle->tag.empty())
	{
		set = delegate.lock()->retrieve_set(payload.get_label(), payload.get_incoming());

	}
	else
	{
		set = delegate.lock()->retrieve_set(payload.get_label(), handle->tag);
	}



	set->get_scheduling()->to(batch_offsets, batch_durations);
	std::vector<unsigned int> possible_offsets, possible_durations;
	for (std::size_t k = 0; k < set->get_scheduling()->get_repetitions(); k++)
	{
		for (std::size_t offset = 0; offset < batch_durations[k] - handle->snippets_size; offset++)
		{
			possible_offsets.push_back(batch_offsets[k] + offset);
		}
	}

	if (handle->tag.empty())
	{
		score.resize(possible_offsets.size());
		std::fill(score.begin(), score.end(),  score.size());
	}
	else
	{
		std::vector<float> reward_elements;
		std::size_t reward_rows;
		std::size_t reward_cols;
		set->get_sequence()->to(reward_elements, reward_rows, reward_cols);

		for (std::size_t k = 0; k < set->get_scheduling()->get_repetitions(); k++)
		{
			std::vector<float> integral(batch_durations[k]);
			float cumsum = 0.0f;
			for (unsigned int l = 0; l < batch_durations[k]; l++)
			{
				integral[l] = cumsum;
				cumsum += reward_elements[batch_offsets[k] + l];
			}
			for (int l = 0; l < batch_durations[k] - handle->snippets_size; l++)
			{
				score.push_back((integral[l + handle->snippets_size] - integral[l]) * possible_offsets.size());
			}
		}
	}

	std::default_random_engine generator;
	std::discrete_distribution<unsigned int> distribution(score.begin(), score.end());
	auto rnd = std::bind(distribution, generator);
	std::vector<unsigned int> offsets(snippets_number);
	for (std::size_t n = 0; n < snippets_number; n++)
	{
		offsets[n] = possible_offsets[rnd()];
	}
	
	notify(TRN::Core::Scheduling::create(implementor, offsets, durations));
}

std::shared_ptr<TRN::Scheduler::Snippets> TRN::Scheduler::Snippets::create(const std::shared_ptr<TRN::Backend::Driver> &driver, const unsigned int  &snippets_size, const unsigned int &time_budget, const std::string &tag)
{
	return std::make_shared<TRN::Scheduler::Snippets>(driver, snippets_size, time_budget, tag);
}