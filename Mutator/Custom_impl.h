#pragma once


#include "Custom.h"
#include "Helper/Queue.h"
class TRN::Mutator::Custom::Handle 
{
public :
	unsigned long seed;
	std::function<void(const unsigned long &seed, const std::size_t &trial, const std::vector<int> &offsets, const std::vector<int> &durations)> functor;

};
