#pragma once

#include "Worker.h"
#include "Core/Simulator.h"



class TRN::Engine::Worker::Handle
{
public:


	std::map<unsigned int, std::shared_ptr<TRN::Core::Simulator>> simulators;
	


};
