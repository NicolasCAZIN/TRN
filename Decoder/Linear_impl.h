#pragma once

#include "Linear.h"

struct TRN::Decoder::Linear::Handle
{
	std::shared_ptr<TRN::Core::Matrix> cx;
	std::shared_ptr<TRN::Core::Matrix> cy;
};
