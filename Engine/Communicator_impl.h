#pragma once

#include "Communicator.h"
#include "Helper/Queue.h"

class TRN::Engine::Communicator::Handle
{
public :
	typedef  std::pair<TRN::Engine::Tag, std::string> Blob;
	std::string host;

};
