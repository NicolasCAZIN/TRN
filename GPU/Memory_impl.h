#pragma once

#include "Memory.h"

class TRN::GPU::Memory::Handle
{
public:
	std::shared_ptr<TRN::GPU::Context> context;

public:
	Handle(const std::shared_ptr<TRN::GPU::Context> context) : context(context)
	{
	}
};

template class GPU_EXPORT std::unique_ptr<TRN::GPU::Memory::Handle>;
