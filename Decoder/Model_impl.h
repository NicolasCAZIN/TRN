#pragma once

#include "Model.h"

struct TRN::Decoder::Model::Handle
{
	std::shared_ptr<TRN::Core::Matrix> cx;
	std::shared_ptr<TRN::Core::Matrix> cy;
	std::shared_ptr<TRN::Core::Matrix> width;
	std::shared_ptr<TRN::Core::Batch> gx2w;
	std::shared_ptr<TRN::Core::Batch> gy2w;
};