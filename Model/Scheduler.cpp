#include "stdafx.h"
#include "Scheduler.h"
#include "Scheduler/Tiled.h"
#include "Scheduler/Snippets.h"
#include "Scheduler/Custom.h"

std::shared_ptr<TRN::Core::Scheduler> TRN::Model::Scheduler::Tiled::create(const unsigned int &epochs)
{
	return TRN::Scheduler::Tiled::create(epochs);
}

std::shared_ptr<TRN::Core::Scheduler> TRN::Model::Scheduler::Snippets::create(const unsigned long &seed, const unsigned int &snippets_size, const unsigned int &time_budget,
	const float &learn_reverse_rate, const float &generate_reverse_rate,
	const float &learning_rate,
	const float &discount, const std::string &tag)
{
	return TRN::Scheduler::Snippets::create(seed, snippets_size, time_budget, learn_reverse_rate, generate_reverse_rate, learning_rate, discount, tag);
}

std::shared_ptr<TRN::Core::Scheduler> TRN::Model::Scheduler::Custom::create(const unsigned long &seed,
	const  std::function<void(const unsigned long long &evaluation_id, const unsigned long &seed, const std::vector<float> &elements, const std::size_t &rows, const std::size_t &cols, const std::vector<int> &offsets, const std::vector<int> &durations)> &request,
	std::function<void(const unsigned long long &evaluation_id, const std::vector<int> &offsets, const std::vector<int> &durations)> &reply,
	const std::string &tag)
{
	return TRN::Scheduler::Custom::create(seed, request, reply, tag);
}