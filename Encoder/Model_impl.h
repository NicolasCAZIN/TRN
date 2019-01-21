

#pragma once

#include "Model.h"

struct TRN::Encoder::Model::Handle
{
	std::shared_ptr<TRN::Core::Matrix> cx;
	std::shared_ptr<TRN::Core::Matrix> cy;
	std::shared_ptr<TRN::Core::Matrix> width;
};
