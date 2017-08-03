#pragma once

#include "Driver.h"
#include "Memory.h"
#include "Random.h"
#include "Algorithm.h"

class TRN::Backend::Driver::Handle
{
public:
	std::shared_ptr<TRN::Backend::Memory> memory;
	std::shared_ptr<TRN::Backend::Random> random;
	std::shared_ptr<TRN::Backend::Algorithm> algorithm;
}; 
