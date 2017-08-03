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

	std::function<void(const std::string &phase, const size_t &batch_size, const size_t &cycles, const float &gflops, const float &seconds)> functor;
};
