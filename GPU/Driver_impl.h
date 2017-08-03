#pragma once

#include "Driver.h"


class TRN::GPU::Driver::Handle
{
public:
	std::shared_ptr<TRN::GPU::Context> context;
};

//template class GPU_EXPORT std::unique_ptr<TRN::GPU::Driver::Handle>;
