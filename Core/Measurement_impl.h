#pragma once

#include "Measurement.h"
#include "Batch.h"

class TRN::Core::Measurement::Abstraction::Handle
{
	public :
		std::shared_ptr<TRN::Core::Matrix> expected;
		std::shared_ptr<TRN::Core::Batch> batched_predicted;
		std::shared_ptr<TRN::Core::Matrix> error;
		std::size_t stored;
		std::size_t preamble;
		std::size_t batch_size;
};
