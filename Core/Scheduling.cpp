#include "stdafx.h"
#include "Scheduling_impl.h"



TRN::Core::Scheduling::Scheduling(const std::shared_ptr<TRN::Backend::Driver> &driver, const std::vector<unsigned int> &offsets, const std::vector<unsigned int> &durations) :
	TRN::Helper::Bridge<TRN::Backend::Driver>(driver),
	handle(std::make_unique<Handle>())
{
	if (offsets.size() != durations.size())
		throw std::invalid_argument("offsets and duration must have the same size");
	
	handle->total_duration = std::accumulate(durations.begin(), durations.end(), 0);
	handle->repetitions = std::min(offsets.size(), durations.size());
	
	handle->offsets = offsets;
	handle->durations = durations;
	/*implementor->get_memory()->allocate((void **)&handle->durations, handle->durations_stride, sizeof(int), handle->repetitions, 1);
	implementor->get_memory()->upload(durations.data(), handle->durations, sizeof(int), handle->repetitions, 1, handle->durations_stride);


	implementor->get_memory()->allocate((void **)&handle->offsets, handle->offsets_stride, sizeof(int), handle->repetitions, 1);
	implementor->get_memory()->upload(offsets.data(), handle->offsets, sizeof(int), handle->repetitions, 1, handle->offsets_stride);*/
}

TRN::Core::Scheduling::~Scheduling()
{
	/*implementor->get_memory()->deallocate(handle->durations);
	implementor->get_memory()->deallocate(handle->offsets);*/
	handle->offsets.clear();
	handle->durations.clear();
	handle.reset();
}

void TRN::Core::Scheduling::to(std::vector<unsigned int> &offsets, std::vector<unsigned int> &durations)
{
	/*offsets.resize(handle->repetitions);
	durations.resize(handle->repetitions);
	implementor->get_memory()->download(durations.data(), handle->durations, sizeof(int), handle->repetitions, 1, handle->durations_stride, false);
	implementor->get_memory()->download(offsets.data(), handle->offsets, sizeof(int), handle->repetitions, 1, handle->offsets_stride, false);*/
	offsets = handle->offsets;
	durations = handle->durations;
}

const std::size_t &TRN::Core::Scheduling::get_total_duration()
{
	return handle->total_duration;
}
const std::size_t&TRN::Core::Scheduling::get_repetitions()
{
	return handle->repetitions;
}

unsigned int *TRN::Core::Scheduling::get_offsets()
{
	return handle->offsets.data();
}

unsigned int *TRN::Core::Scheduling::get_durations()
{
	return handle->durations.data();
}

std::shared_ptr<TRN::Core::Scheduling> TRN::Core::Scheduling::create(const std::shared_ptr<TRN::Backend::Driver> &driver, const std::vector<unsigned int> &offsets, const std::vector<unsigned int> &durations) 
{
	return std::make_shared<TRN::Core::Scheduling>(driver, offsets, durations);
}