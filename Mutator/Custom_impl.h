#pragma once


#include "Custom.h"
#include "Helper/Queue.h"
class TRN::Mutator::Custom::Handle 
{
public :
	std::function<void(const std::vector<int> &offsets, const std::vector<int> &durations)> functor;
	TRN::Helper::Queue<TRN::Core::Scheduling> queue;
};
