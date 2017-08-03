#pragma once

#include "Random.h"


class TRN::GPU::Random::Handle
{
public:
	std::shared_ptr<TRN::GPU::Context> context;

};


template class GPU_EXPORT std::unique_ptr<TRN::GPU::Random::Handle>;
