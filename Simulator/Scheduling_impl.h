#pragma once

#include "Scheduling.h"

class TRN::Simulator::Scheduling::Handle
{
public:


	std::function<void(const std::size_t &trial, const std::vector<int> &offsets, const std::vector<int> &durations)> functor;
};
