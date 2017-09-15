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

#include "ViewModel/Communicator.h"
#include "ViewModel/Node.h"

static int DEFAULT_INDEX = 0;
static int DEFAULT_SEED = 0;




int main(int argc, char *argv[])
{
	try
	{
		boost::program_options::options_description desc("Allowed options");
		desc.add_options()
			("help,h", "produce help message")
			("index,i", boost::program_options::value<int>(), "local device index. 0 for cpu, 1 for first gpu, 2 for second gpu ...")
			;

		boost::program_options::variables_map vm;
		boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);
		boost::program_options::notify(vm);

		if (vm.count("help"))
		{
			std::cout << desc << "\n";
			return 1;
		}

		auto index = vm.count("index") ? vm["index"].as<int>() : DEFAULT_INDEX;
		auto seed = vm.count("seed") ? vm["seed"].as<int>() : DEFAULT_SEED;
		auto communicator = TRN::ViewModel::Communicator::Distributed::create(argc, argv);

		
		auto worker = TRN::ViewModel::Node::Worker::create(communicator, communicator->rank(), index);
		worker->start();
		return 0;
	}
	catch (std::exception &e)
	{
		std::cerr << e.what() << std::endl;
		return -1;
	}
}