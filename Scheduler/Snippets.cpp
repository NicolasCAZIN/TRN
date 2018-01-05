#include "stdafx.h"
#include "Snippets.h"

class TRN::Scheduler::Snippets::Handle
{
public :
	unsigned int snippets_size;
	unsigned int time_budget;
	unsigned long seed;
	std::string tag;
};

TRN::Scheduler::Snippets::Snippets(const unsigned long &seed, const unsigned int &snippets_size, const unsigned int &time_budget, const std::string &tag) :
	handle(std::make_unique<TRN::Scheduler::Snippets::Handle>())
{
	handle->snippets_size = snippets_size;
	handle->time_budget = time_budget;
	handle->tag = tag;
	handle->seed = seed;
}

TRN::Scheduler::Snippets::~Snippets()
{
	handle.reset();
}

void TRN::Scheduler::Snippets::update(const TRN::Core::Message::Payload<TRN::Core::Message::SET> &payload)
{
	auto snippets_number = handle->time_budget / handle->snippets_size;
	auto remaining_time = handle->time_budget % handle->snippets_size;
	std::vector<int> durations(snippets_number);
	std::fill(durations.begin(), durations.end(), (int)handle->snippets_size);
	if (remaining_time)
	{
		durations.push_back(remaining_time);
		snippets_number++;
	}
	std::shared_ptr<TRN::Core::Set> set;
	std::vector<int> score;
	std::vector<int> batch_offsets, batch_durations;

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
	auto b = batch_offsets.begin();
	for (std::size_t k = 0; k < set->get_scheduling()->get_durations().size(); k++)
	{
		auto e = b + batch_durations[k] - handle->snippets_size;
		
		possible_offsets.insert(possible_offsets.end(), b, e);
		b = e + handle->snippets_size;
	}

	if (handle->tag.empty())
	{
		score.resize(possible_offsets.size());
		std::fill(score.begin(), score.end(),  1);
	}
	else
	{
		std::vector<float> reward_elements;
		std::size_t reward_rows;
		std::size_t reward_cols;
		set->get_sequence()->to(reward_elements, reward_rows, reward_cols);

		std::size_t c = 0;
		for (std::size_t k = 0; k < set->get_scheduling()->get_durations().size(); k++)
		{
			std::vector<float> integral(batch_durations[k]);
		
			float cumsum = 0.0f;
			for (unsigned int l = 0; l < batch_durations[k]; l++)
			{
				integral[l] = cumsum;
				cumsum += reward_elements[batch_offsets[l + c]];
			}
			if (cumsum == 0.0f)
				throw std::runtime_error("no reward present in sequence " + payload.get_label() + " with tag " + handle->tag);
			for (int l = 0; l < batch_durations[k] - handle->snippets_size; l++)
			{
				score.push_back((integral[l + handle->snippets_size] - integral[l]));
			}
			c += batch_durations[k];
		}
	}

	std::default_random_engine engine(handle->seed);

	std::discrete_distribution<int> score_distribution(score.begin(), score.end());
	auto random_offset = std::bind(score_distribution, engine);
	//std::uniform_real_distribution<float> mutate_distribution(0.0f, 1.0f);
	//auto random_mutation = std::bind(mutate_distribution, engine);
	std::vector<int> offsets;
	for (std::size_t n = 0; n < snippets_number; n++)
	{

		std::vector<int> snippet(handle->snippets_size);
		std::iota(snippet.begin(), snippet.end(), possible_offsets[random_offset()]);
		offsets.insert(offsets.end(), snippet.begin(), snippet.end());
	}
	if (std::any_of(offsets.begin(), offsets.end(), [](const int &offset) {return offset < 0; }))
		throw std::runtime_error("offset can't be negative");
	handle->seed += offsets.size() * durations.size();
	notify(TRN::Core::Message::Payload<TRN::Core::Message::SCHEDULING>(payload.get_trial(), TRN::Core::Scheduling::create(offsets, durations)));
}

std::shared_ptr<TRN::Scheduler::Snippets> TRN::Scheduler::Snippets::create(const unsigned long &seed, const unsigned int  &snippets_size, const unsigned int &time_budget, const std::string &tag)
{
	return std::make_shared<TRN::Scheduler::Snippets>(seed, snippets_size, time_budget, tag);
}