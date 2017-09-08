#pragma once

#include "Reverse.h"
#include "Shuffle_impl.h"

class TRN::Mutator::Reverse::Handle
{
public :
	float rate;
	std::size_t size;
	unsigned long seed;
};
