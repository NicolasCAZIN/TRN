#include "stdafx.h"
#include "Scheduling_impl.h"



TRN::Core::Scheduling::Scheduling(const std::vector<int> &offsets, const std::vector<int> &durations) :
	handle(std::make_unique<Handle>())
{	
	handle->offsets = offsets;
	handle->durations = durations;
}

TRN::Core::Scheduling::Scheduling(const std::vector<std::vector<int>> &indices) :
	handle(std::make_unique<Handle>())
{
	from(indices);
}

TRN::Core::Scheduling::~Scheduling()
{
	handle->offsets.clear();
	handle->durations.clear();
	handle.reset();
}

void TRN::Core::Scheduling::to(std::vector<int> &offsets, std::vector<int> &durations)
{
	offsets = handle->offsets;
	durations = handle->durations;
}


TRN::Core::Scheduling &TRN::Core::Scheduling::operator = (const TRN::Core::Scheduling &scheduling)
{
	this->handle->offsets = scheduling.handle->offsets;
	this->handle->durations = scheduling.handle->durations;
	return *this;
}

void TRN::Core::Scheduling::to(std::vector<std::vector<int>> &indices)
{
	indices.resize(handle->durations.size());
	auto b = std::begin(handle->offsets);
	for (std::size_t k = 0; k < indices.size(); k++)
	{
		auto e = b + handle->durations[k];
		indices[k].resize(handle->durations[k]);
		std::copy(b, e, std::begin(indices[k]));
		b = e;
	}

}
void TRN::Core::Scheduling::from(const std::vector<std::vector<int>> &indices)
{
	auto total_duration = std::accumulate(std::begin(indices), std::end(indices), (int)0, [](const int &accumulator, const std::vector<int> &v) { return accumulator + v.size(); });
	handle->durations.resize(indices.size());
	handle->offsets.resize(total_duration);

	auto b = handle->offsets.begin();
	for (std::size_t k = 0; k < indices.size(); k++)
	{
		auto &offsets_k = indices[k];

		auto duration = offsets_k.size();
		auto e = b + duration;

		std::copy(std::begin(offsets_k), std::end(offsets_k), b);
		handle->durations[k] = duration;
		b = e;
	}
}

std::vector<int> TRN::Core::Scheduling::get_offsets()
{
	return handle->offsets;
}

std::vector<int> TRN::Core::Scheduling::get_durations()
{
	return handle->durations;
}

void TRN::Core::Scheduling::set_offsets(const std::vector<int> &offsets)
{
	handle->offsets = offsets;
}
void TRN::Core::Scheduling::set_durations(const std::vector<int> &durations)
{
	handle->durations = durations;
}

std::size_t TRN::Core::Scheduling::get_total_duration()
{
	return handle->offsets.size();
}


std::shared_ptr<TRN::Core::Scheduling> TRN::Core::Scheduling::create(const std::vector<std::vector<int>> &indices)
{
	return std::make_shared<TRN::Core::Scheduling>(indices);
}

std::shared_ptr<TRN::Core::Scheduling> TRN::Core::Scheduling::create(const std::vector<int> &offsets, const std::vector<int> &durations) 
{
	return std::make_shared<TRN::Core::Scheduling>( offsets, durations);
}