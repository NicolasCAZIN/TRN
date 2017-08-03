#pragma once

#include "Loop.h"
#include "Batch.h"

class TRN::Core::Loop::Handle
{
	public:
		std::shared_ptr<TRN::Core::Batch> stimulus;
		std::size_t batch_size;
};