#include "stdafx.h"
#include "Scheduler.h"
#include "Scheduler/Tiled.h"
#include "Scheduler/Snippets.h"
#include "Scheduler/Custom.h"

std::shared_ptr<TRN::Core::Scheduler> TRN::Model::Scheduler::Tiled::create(const std::shared_ptr<TRN::Backend::Driver> &driver, const unsigned int &epochs)
{
	return TRN::Scheduler::Tiled::create(driver, epochs);
}

std::shared_ptr<TRN::Core::Scheduler> TRN::Model::Scheduler::Snippets::create(const std::shared_ptr<TRN::Backend::Driver> &driver, const unsigned int &snippets_size, const unsigned int &time_budget, const std::string &tag)
{
	return TRN::Scheduler::Snippets::create(driver, snippets_size, time_budget, tag);
}

std::shared_ptr<TRN::Core::Scheduler> TRN::Model::Scheduler::Custom::create(const std::shared_ptr<TRN::Backend::Driver> &driver, 
	const  std::function<void(const std::vector<float> &elements, const std::size_t &rows, const std::size_t &cols, const std::vector<unsigned int> &offsets, const std::vector<unsigned int> &durations)> &request,
	std::function<void(const std::vector<unsigned int> &offsets, const std::vector<unsigned int> &durations)> &reply,
	const std::string &tag)
{
	return TRN::Scheduler::Custom::create(driver, request, reply, tag);
}