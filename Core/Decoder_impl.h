#pragma once

#include "Decoder.h"

struct TRN::Core::Decoder::Handle
{
	std::size_t batch_size;
	std::size_t stimulus_size;
};
