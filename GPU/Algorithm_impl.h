#pragma once

#include "Algorithm.h"

class TRN::GPU::Algorithm::Handle
{
public:
	std::shared_ptr<TRN::GPU::Context> context;

	std::vector<void *> temp;
	std::vector<void *> argmax;
	std::vector<std::size_t> size;
};

//template class GPU_EXPORT std::unique_ptr<TRN::GPU::Algorithm::Handle>;
