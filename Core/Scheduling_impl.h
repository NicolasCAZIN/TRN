#pragma once

#include "Scheduling.h"

class TRN::Core::Scheduling::Handle
{
public:
	std::size_t durations_stride;
	std::size_t offsets_stride;
	std::size_t total_duration;
	std::size_t repetitions;
	 std::vector<unsigned int> offsets;
	 std::vector<unsigned int> durations;
/*	int *offsets;
	int *durations;*/
};