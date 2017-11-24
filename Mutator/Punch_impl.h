#pragma once

#include "Punch.h"

class TRN::Mutator::Punch::Handle
{
public:
	float rate;
	std::size_t size;
	std::size_t counter;
	unsigned long seed;
};
