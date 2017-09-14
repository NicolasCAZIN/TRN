#pragma once

#include "Communicator.h"
#include "Helper/Queue.h"

class TRN::Local::Communicator::Handle
{
public :
	typedef  std::pair<TRN::Engine::Tag, std::string> Blob;
	std::vector
		<
			std::shared_ptr
			<
				TRN::Helper::Queue
				<
					std::shared_ptr<Blob>
				>
			>
		> queues;
	std::recursive_mutex mutex;
	std::vector<std::shared_ptr<TRN::Engine::Worker>> workers;
};