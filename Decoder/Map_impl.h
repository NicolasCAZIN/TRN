#pragma once

#include "Map.h"

struct TRN::Decoder::Map::Handle
{
	std::shared_ptr<TRN::Core::Matrix> firing_rate_map;
};