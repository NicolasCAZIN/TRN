#include "stdafx.h"
#include "Communicator.h"
#include "Local/Communicator.h"
#include "Distributed/Communicator.h"

#include "Model/Driver.h"
#include "Engine/Worker.h"
/*#include "Remote/Backend.h"
#include "Distributed/Backend.h"
*/
std::shared_ptr<TRN::Engine::Communicator> TRN::ViewModel::Communicator::Local::create(const std::list<unsigned int> &indexes)
{
	std::list<unsigned int> local_indexes;

	if (indexes.empty())
	{
		auto devices = TRN::Model::Driver::enumerate_devices();
		for (auto device : devices)
		{
			local_indexes.push_back(device.first);
		}
	}
	else
	{
		local_indexes = indexes;
	}

	auto communicator = TRN::Local::Communicator::create(local_indexes.size() + 1);
	for (auto index : local_indexes)
	{
		communicator->append(TRN::Engine::Worker::create(TRN::Model::Driver::create(index), communicator));
	}

	return communicator;
}

std::shared_ptr<TRN::Engine::Communicator> TRN::ViewModel::Communicator::Distributed::create(int argc, char *argv[])
{

	return TRN::Distributed::Communicator::create(argc, argv);
}

/*std::shared_ptr<TRN::Engine::Backend> TRN::ViewModel::Backend::Remote::create(const std::string &host, const unsigned short &port)
{
return TRN::Remote::Backend::create(host, port);
}

*/