#pragma once

#include "Performances.h"

class TRN::Simulator::Performances::Handle
{
public:
	std::size_t preamble;
	std::size_t cycles;
	std::size_t observations;
	std::size_t batch_size;
	std::chrono::high_resolution_clock::time_point start;
	bool train;
	bool prime;
	bool generate;

	std::function<void(const unsigned long long &evaluation_id, const std::string &phase, const float &cycles_per_second, const float &gflops_per_second)> functor;
};
