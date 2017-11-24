#pragma once

#include "Communicator.h"
#include "Helper/Queue.h"

class TRN::Engine::Communicator::Handle
{
public :
	std::shared_ptr<TRN::Engine::Compressor> compressor;
	typedef  std::pair<TRN::Engine::Tag, std::string> Blob;
	std::string host;
	unsigned short offset;

};
