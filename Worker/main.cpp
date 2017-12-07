#ifdef USE_VLD
#include <vld.h>
#endif 
#include <iostream>
#include <functional>
#include <map>
#include <vector>
#include <list>
#include <mutex>
#include <string>
#include <boost/asio.hpp>
#include <boost/mpi.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/program_options.hpp>
#include <boost/optional.hpp>

#include "ViewModel/Communicator.h"
#include "ViewModel/Node.h"
#include "Helper/Logger.h"
static int DEFAULT_INDEX = 0;

int main(int argc, char *argv[])
{
	try
	{
		TRACE_LOGGER <<   argv[0] << " INITIALIZE" ;
		boost::program_options::options_description desc("Allowed options");
		desc.add_options()
			("help,h", "produce help message")
			("index,i", boost::program_options::value<int>()->default_value(DEFAULT_INDEX), "local device index. 0 for cpu, 1 for first gpu, 2 for second gpu ...")
			("logging,l", boost::program_options::value<TRN::Helper::Logger::Severity>()->default_value(TRN::Helper::Logger::Severity::INFORMATION_LEVEL), "Logging severity level filtering [TRACE|DEBUG|INFORMATION|WARNING|ERROR]")
			;

		boost::program_options::variables_map vm;
		boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);
		boost::program_options::notify(vm);

		if (vm.count("help"))
		{
			std::cerr <<   desc << "\n";
			return 1;
		}
		TRN::Helper::Logger::setup(vm["logging"].as<TRN::Helper::Logger::Severity>());
		auto index = vm.count("index") ? vm["index"].as<int>() : DEFAULT_INDEX;
		auto communicator = TRN::ViewModel::Communicator::Distributed::create(argc, argv);

		
		auto worker = TRN::ViewModel::Node::Backend::create(communicator, communicator->rank(), index);
		worker->start();
		worker->dispose();
		TRACE_LOGGER <<   argv[0] << " EXITED" ;
		return 0;
	}
	catch (std::exception &e)
	{
		ERROR_LOGGER << e.what() ;
		return -1;
	}
}