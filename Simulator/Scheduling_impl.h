#pragma once

#include "Scheduling.h"

class TRN::Simulator::Scheduling::Handle
{
public:


	std::function<void(const std::vector<int> &offsets, const std::vector<int> &durations)> functor;
};
