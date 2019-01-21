#include "stdafx.h"
#include "Snippets.h"

struct Scheduling
{
	std::vector<float> reward_elements;
	std::vector<float> replay_likelihood;
	std::vector<int>	batch_offsets;
	std::vector<int>	batch_durations;
};

class TRN::Scheduler::Snippets::Handle
{
public :
	unsigned int snippets_size;
	unsigned int time_budget;
	std::default_random_engine random_engine;
	std::string tag;
	float learn_reverse_rate;
	float generate_reverse_rate;
	float learning_rate;
	float discount;

	std::map<std::shared_ptr<TRN::Core::Set>, Scheduling > cache;
};

TRN::Scheduler::Snippets::Snippets(const unsigned long &seed, const unsigned int &snippets_size, const unsigned int &time_budget, 
	const float &learn_reverse_rate, const float &generate_reverse_rate,
	const float &learning_rate,
	const float &discount, const std::string &tag) :
	handle(std::make_unique<TRN::Scheduler::Snippets::Handle>())
{
	handle->snippets_size = snippets_size;
	handle->time_budget = time_budget;
	handle->learn_reverse_rate = learn_reverse_rate;
	handle->generate_reverse_rate = generate_reverse_rate;
	handle->learning_rate = learning_rate;
	handle->discount = discount;
	handle->tag = tag;
	handle->random_engine = std::default_random_engine(seed);
}

TRN::Scheduler::Snippets::~Snippets()
{
	handle.reset();
}

void TRN::Scheduler::Snippets::reset()
{
	handle->cache.clear();
}

//#define CUMULATED 1
#define DISCOUNT 
/*

void TRN::Scheduler::Snippets::update(const TRN::Core::Message::Payload<TRN::Core::Message::SET> &payload)
{
#if defined (DISCOUNT)

std::shared_ptr<TRN::Core::Set> set;

if (handle->tag.empty())
{
set = delegate.lock()->retrieve_set(payload.get_label(), payload.get_incoming());
if (handle->cache.find(set) == handle->cache.end())
{
set->get_scheduling()->to(handle->cache[set].batch_offsets, handle->cache[set].batch_durations);
handle->cache[set].reward_elements.resize(handle->cache[set].batch_offsets.size());
handle->cache[set].replay_likelihood.resize(handle->cache[set].batch_offsets.size());
std::fill(handle->cache[set].reward_elements.begin(), handle->cache[set].reward_elements.end(), 1.0f);
std::copy(handle->cache[set].reward_elements.begin(), handle->cache[set].reward_elements.end(), handle->cache[set].replay_likelihood.begin());
}
}
else
{
std::size_t reward_rows;
std::size_t reward_cols;

set = delegate.lock()->retrieve_set(payload.get_label(), handle->tag);
if (handle->cache.find(set) == handle->cache.end())
{
set->get_sequence()->to(handle->cache[set].reward_elements, reward_rows, reward_cols);
set->get_scheduling()->to(handle->cache[set].batch_offsets, handle->cache[set].batch_durations);
handle->cache[set].replay_likelihood.resize(handle->cache[set].batch_offsets.size());
std::copy(handle->cache[set].reward_elements.begin(), handle->cache[set].reward_elements.end(), handle->cache[set].replay_likelihood.begin());
}
}

auto info = handle->cache[set];

std::vector<float> &V = info.replay_likelihood;
std::vector<float> &R = info.reward_elements;
auto alpha = handle->decay;
auto gamma = handle->discount;
auto L = handle->snippets_size;

std::vector<int> sequence_number, sequence_offset;


int s = 0;
for (auto d : info.batch_durations)
{
std::vector<int> D(d, s);
std::vector<int> relative_offset(d);
s++;
sequence_number.insert(sequence_number.end(), D.begin(), D.end());
std::iota(relative_offset.begin(), relative_offset.end(), 0);
sequence_offset.insert(sequence_offset.end(), relative_offset.begin(), relative_offset.end());
}


std::vector<int> set_offset(1, 0);

std::copy(info.batch_durations.begin(), info.batch_durations.end() - 1, std::back_inserter(set_offset));

std::vector<int> offsets;
std::vector<int> durations;

std::uniform_real_distribution<float> reverse(0, 1);


while (offsets.size() < handle->time_budget)
{
auto t = draw_snippet(V, R);
std::discrete_distribution<int> replay_distribution(V.begin(), V.end());
bool drawn = false;
std::vector<int> t;
while (!drawn)
{
auto i = replay_distribution(handle->random_engine);

assert(0 <= i && i < info.reward_elements.size());
auto s = sequence_number[i];

auto d = info.batch_durations[s];
auto o = sequence_offset[i];

if (reverse(handle->random_engine) < handle->reverse_rate)
{
auto begin = std::max(0, (int)(o - L));
auto end = o;
auto size = end - begin;

t.resize(end - begin);
if (size > 1)
{
std::iota(t.begin(), t.end(), begin + set_offset[s]);
std::reverse(t.begin(), t.end());
drawn = true;
}
}
else
{
auto begin = o;
auto end = std::min(d, (int)(o + L));
auto size = end - begin;

if (size > 1)
{
t.resize(end - begin);
std::iota(t.begin(), t.end(), begin + set_offset[s]);
drawn = true;
}
}
}
// tdlambda
assert(1 < t.size() && t.size() <= L);

for (std::size_t n = 0; n < t.size() - 1; n++)
{
V[t[n + 1]] += alpha * (R[t[n + 1]] + gamma * V[t[n]] - V[t[n + 1]]);
}




}

#elif defined (CUMULATED)
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


std::vector<std::vector<float>> reward_sites;
std::vector<std::vector<int>> relative_offsets;

std::size_t b = 0;

for (std::size_t k = 0; k < batch_durations.size(); k++)
{
std::vector<float> batch_reward(batch_durations[k]);
std::vector<int> batch_offset(batch_durations[k]);

std::copy(reward_elements.begin() + b, reward_elements.begin() + b + batch_durations[k], batch_reward.begin());
std::copy(batch_offsets.begin() + b, batch_offsets.begin() + b + batch_durations[k], batch_offset.begin());


reward_sites.push_back(batch_reward);
relative_offsets.push_back(batch_offset);
b += batch_durations[k];
}

auto sequence = std::uniform_int<std::size_t>(0, reward_sites.size() - 1);

std::vector<int> offsets;
std::vector<int> durations;

while (offsets.size() < handle->time_budget)
{
std::size_t s = sequence(handle->random_engine);
std::discrete_distribution<std::size_t> d(reward_sites[s].begin(), reward_sites[s].end());
std::size_t r0 = d(handle->random_engine);

std::size_t o = 0;
std::size_t duration = 0;

do
{
std::size_t r1 = d(handle->random_engine);
if (r1 == r0)
continue;
if (r0 > r1)
std::swap(r0, r1);
std::uniform_int<std::size_t> offset(r0, r1 - 1);

o = offset(handle->random_engine);
duration = std::min((std::size_t)handle->snippets_size, (std::size_t)(reward_sites[s].size() - o));

} while (duration == 0);

if (duration <= 0 || duration > handle->snippets_size || o + duration > reward_sites[s].size())
throw std::runtime_error("invalid snippet duration : " + std::to_string(duration));

std::vector<int> snippet(duration);

std::iota(snippet.begin(), snippet.end(), relative_offsets[s][o]);

durations.push_back(static_cast<int>(snippet.size()));
offsets.insert(offsets.end(), snippet.begin(), snippet.end());

}
#else
std::vector<int> offsets;
std::vector<int> durations;



std::vector<int> score;
std::shared_ptr<TRN::Core::Set> set;

std::vector<int> batch_offsets, batch_durations;

std::vector<float> reward_elements;

if (handle->tag.empty())
{
set = delegate.lock()->retrieve_set(payload.get_label(), payload.get_incoming());
set->get_scheduling()->to(batch_offsets, batch_durations);
reward_elements.resize(batch_offsets.size());
std::fill(reward_elements.begin(), reward_elements.end(), 0.0f);
}
else
{
std::size_t reward_rows;
std::size_t reward_cols;

set = delegate.lock()->retrieve_set(payload.get_label(), handle->tag);
set->get_sequence()->to(reward_elements, reward_rows, reward_cols);
set->get_scheduling()->to(batch_offsets, batch_durations);
}


std::vector<float> cost;
auto b = reward_elements.begin();
for (std::size_t k = 0; k < batch_durations.size(); k++)
{
auto e = b + batch_durations[k];
const float s = std::accumulate(b, e, 0.0f);
if (s > 0.0f)
std::transform(b, e, b, [&](const float &x)
{
return x / s;
});
else
{
std::fill(b, e, 1.0f / batch_durations[k]);
}


while (b + handle->snippets_size < e)
{
cost.push_back(std::accumulate(b, b + handle->snippets_size, 0.0f));
b++;
}
while (b < e)
{
cost.push_back(0.0f);
b++;
}


}
std::discrete_distribution<int> reward_distribution(cost.begin(), cost.end());

for (std::size_t time_budget = 0; time_budget < handle->time_budget; time_budget += handle->snippets_size)
{
auto snippet = reward_distribution(handle->random_engine);
for (int offset = snippet; offset < snippet + handle->snippets_size; offset++)
{
offsets.push_back(offset);
}
durations.push_back(handle->snippets_size);
}
#endif
notify(TRN::Core::Message::Payload<TRN::Core::Message::SCHEDULING>(payload.get_evaluation_id(), TRN::Core::Scheduling::create(offsets, durations)));
}
*/

std::vector<int> TRN::Scheduler::Snippets::draw_snippet(const std::vector<float> &V, const std::vector<float> &R, const std::vector<int> &sequence_offset, const std::vector<int> &set_offset, const std::vector<int> &batch_durations, const std::vector<int> &sequence_number, const float &reverse_rate)
{
	auto L = handle->snippets_size;
	std::uniform_real_distribution<float> reverse(0, 1);
	std::discrete_distribution<int> replay_distribution(V.begin(), V.end());

	std::vector<int> snippet;
	bool drawn = false;
	while (!drawn)
	{
		auto i = replay_distribution(handle->random_engine);

		assert(0 <= i && i < R.size());
		auto s = sequence_number[i];
		auto d = batch_durations[s];
		auto o = sequence_offset[i];

		if (reverse(handle->random_engine) < reverse_rate)
		{
			auto begin = std::max(0, (int)(o - (L - 1)));
			auto end = o;
			auto size = end - begin + 1;

			if (size > 0)
			{
				snippet.resize(size);
				std::iota(snippet.begin(), snippet.end(), begin + set_offset[s]);
				std::reverse(snippet.begin(), snippet.end());
				drawn = true;
			}
		}
		else
		{
			auto begin = o;
			auto end = std::min(d-1, (int)(o + (L - 1)));
			auto size = end - begin + 1;

			if (size > 0)
			{
				snippet.resize(size);
				std::iota(snippet.begin(), snippet.end(), begin + set_offset[s]);
				drawn = true;
			}
		}
	}
	assert(0 < snippet.size() && snippet.size() <= L);

	return snippet;
}

void TRN::Scheduler::Snippets::update(const TRN::Core::Message::Payload<TRN::Core::Message::SET> &payload)
{
	std::shared_ptr<TRN::Core::Set> set;

	if (handle->tag.empty())
	{
		set = delegate.lock()->retrieve_set(payload.get_label(), payload.get_incoming());
		if (handle->cache.find(set) == handle->cache.end())
		{
			set->get_scheduling()->to(handle->cache[set].batch_offsets, handle->cache[set].batch_durations);
			handle->cache[set].reward_elements.resize(handle->cache[set].batch_offsets.size());
			handle->cache[set].replay_likelihood.resize(handle->cache[set].batch_offsets.size());

			auto p = 1.0f / handle->cache[set].replay_likelihood.size();

			std::fill(handle->cache[set].replay_likelihood.begin(), handle->cache[set].replay_likelihood.end(), p);


			/*std::uniform_real<float> distribution(0, handle->learning_rate);
			auto dice = std::bind(distribution, handle->random_engine);

			std::generate(handle->cache[set].replay_likelihood.begin(), handle->cache[set].replay_likelihood.end(), dice);*/
		}
	}
	else
	{
		std::size_t reward_rows;
		std::size_t reward_cols;

		set = delegate.lock()->retrieve_set(payload.get_label(), handle->tag);
		if (handle->cache.find(set) == handle->cache.end())
		{
			set->get_sequence()->to(handle->cache[set].reward_elements, reward_rows, reward_cols);
			set->get_scheduling()->to(handle->cache[set].batch_offsets, handle->cache[set].batch_durations);
			handle->cache[set].replay_likelihood.resize(handle->cache[set].batch_offsets.size());

			auto p = 1.0f / handle->cache[set].replay_likelihood.size();

			std::fill(handle->cache[set].replay_likelihood.begin(), handle->cache[set].replay_likelihood.end(), p);
			/*std::uniform_real<float> distribution(0, handle->learning_rate);
			auto dice = std::bind(distribution, handle->random_engine);

			std::generate(handle->cache[set].replay_likelihood.begin(), handle->cache[set].replay_likelihood.end(), dice);*/
		}
	}

	auto info = handle->cache[set];

	std::vector<float> &V = info.replay_likelihood;
	std::vector<float> &R = info.reward_elements;
	auto alpha = handle->learning_rate;
	auto gamma = handle->discount;


	std::vector<int> sequence_number, sequence_offset;

	
	int s = 0;
	for (auto d : info.batch_durations)
	{
		std::vector<int> D(d, s);
		std::vector<int> relative_offset(d);
		s++;
		sequence_number.insert(sequence_number.end(), D.begin(), D.end());
		std::iota(relative_offset.begin(), relative_offset.end(), 0);
		sequence_offset.insert(sequence_offset.end(), relative_offset.begin(), relative_offset.end());
	}


	std::vector<int> set_offset(1, 0);

	std::partial_sum(info.batch_durations.begin(), info.batch_durations.end() - 1, std::back_inserter(set_offset));

//	std::copy(info.batch_durations.begin(), info.batch_durations.end() - 1, std::back_inserter(set_offset));

	std::vector<int> offsets;
	std::vector<int> durations;

	/*{
		int l = 0;
		for (int s = 0; s < info.batch_durations.size(); s++)
		{
			auto d = info.batch_durations[s];
			std::cout << "s = " << s << " : ";
			for (int k = 0; k < d; k++, l++)
			{
				std::cout << V[l] << " ";
			}
			std::cout << std::endl;
		}
	}*/

	for (std::size_t n = 0; n < handle->time_budget; n++)
	{
		auto t = draw_snippet(V, R, sequence_offset, set_offset, info.batch_durations, sequence_number, handle->learn_reverse_rate);
		
		for (std::size_t k = 0; k < t.size() - 1; k++)
		{
			V[t[k + 1]] += alpha * (R[t[k]] + gamma * V[t[k]] - V[t[k + 1]]);
		}
	}

	/*std::vector<cv::Mat> histograms;
	{
		int l = 0;
		for (int s = 0; s < info.batch_durations.size(); s++)
		{
			auto d = info.batch_durations[s];
			
			cv::Mat hist(1, d, CV_32F);
			for (int k = 0; k < d; k++, l++)
			{
				hist.at<float>(k) = V[l];
			}

			normalize(hist, hist, 0, d, cv::NORM_MINMAX, -1, cv::Mat());

			histograms.push_back(hist);
		}
	}*/

	while (offsets.size() < handle->time_budget)
	{
		auto t = draw_snippet(V, R, sequence_offset, set_offset, info.batch_durations, sequence_number, handle->generate_reverse_rate);

		offsets.insert(offsets.end(), t.begin(), t.end());
		durations.push_back(t.size());
	}

	/*{
	
		std::vector<float> count(R.size(), 0);
	
		
		for (auto o : offsets)
		{
			count[o]++;
		}

		auto m = *std::max_element(count.begin(), count.end());
		
		cv::Mat hist = cv::Mat::zeros(255, R.size(), CV_32F);
		for (int b = 0; b < R.size(); b++)
		{
			auto h = 255 * count[b] / m;
			cv::line(hist, cv::Point(b, 255), cv::Point(b, 255 - h), cv::Scalar(1.0));
		}
		int c = 0;
		
	}*/
	notify(TRN::Core::Message::Payload<TRN::Core::Message::SCHEDULING>(payload.get_evaluation_id(), TRN::Core::Scheduling::create(offsets, durations)));
}

std::shared_ptr<TRN::Scheduler::Snippets> TRN::Scheduler::Snippets::create(const unsigned long &seed, const unsigned int &snippets_size, const unsigned int &time_budget,
	const float &learn_reverse_rate, const float &generate_reverse_rate,
	const float &learning_rate,
	const float &discount, const std::string &tag)
{
	return std::make_shared<TRN::Scheduler::Snippets>(seed, snippets_size, time_budget, learn_reverse_rate, generate_reverse_rate, learning_rate, discount, tag);
}