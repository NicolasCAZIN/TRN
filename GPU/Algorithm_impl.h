#pragma once

#include "Algorithm.h"

class TRN::GPU::Algorithm::Handle
{
public:
	std::shared_ptr<TRN::GPU::Context> context;


};

//template class GPU_EXPORT std::unique_ptr<TRN::GPU::Algorithm::Handle>;
