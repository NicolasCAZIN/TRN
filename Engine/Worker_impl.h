#pragma once

#include "Worker.h"
#include "Core/Simulator.h"



class TRN::Engine::Worker::Handle
{
public:
	std::map<unsigned int, bool> configured_required;
	std::map<unsigned int, std::size_t> remaining_initializations;
	std::map<unsigned int, std::shared_ptr<TRN::Core::Simulator>> simulators;
};
