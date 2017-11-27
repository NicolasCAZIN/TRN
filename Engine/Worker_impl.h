#pragma once

#include "Worker.h"
#include "Core/Simulator.h"



class TRN::Engine::Worker::Handle
{
public:
	std::set<unsigned short> frontends;
	std::set<unsigned short> quit_not_required;
	std::map<unsigned long long, bool> configured_required;
	std::map<unsigned long long, std::size_t> remaining_initializations;
	std::map<unsigned long long, std::shared_ptr<TRN::Core::Simulator>> simulators;
};
