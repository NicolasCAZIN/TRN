#pragma once

#include "Set.h"

class TRN::Core::Set::Handle
{
public:
	std::shared_ptr<TRN::Core::Matrix> sequence;
	std::shared_ptr<TRN::Core::Scheduling> scheduling;	
};