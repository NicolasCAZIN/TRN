#pragma once

#include "Scheduling.h"

class TRN::Core::Scheduling::Handle
{
public:
	 std::vector<int> offsets;
	 std::vector<int> durations;
};