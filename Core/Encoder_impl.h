#pragma once

#include "Encoder.h"

struct TRN::Core::Encoder::Handle
{
	std::size_t batch_size;
	std::size_t stimulus_size;
};
