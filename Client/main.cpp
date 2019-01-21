#include <boost/asio.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/tokenizer.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <boost/program_options/errors.hpp>
#include <memory>
#include <condition_variable>
#include <vector>
#include <random>
#include <mutex>
#include <thread>
#include <algorithm>    // copy
#include <iostream>
#include <fstream>
#include <functional>
#include <string>
#include <list>
#include <queue>

#include "TRN4CPP/TRN4CPP.h"
#include "Helper/Logger.h"


enum Backend
{
	Distributed,
	Remote,
	Local
};

namespace std
{
	std::ostream& operator<<(std::ostream &os, const std::vector<unsigned int> &vec)
	{
		for (auto item : vec)
		{
			os << item << " ";
		}
		return os;
	}
	std::istream &operator >> (std::istream &is, std::vector<unsigned int> &list)
	{
		unsigned int index;

		is >> index;
		list.push_back(index);
		return is;
	}

	std::ostream &operator <<  (std::ostream &os, const Backend &backend)
	{
		switch (backend)
		{
			case  Backend::Local :
				os << "LOCAL";
				break;
			case  Backend::Remote:
				os << "REMOTE";
				break;
			case  Backend::Distributed:
				os << "DISTRIBUTED";
				break;

		}
		return os;
	}
	std::istream &operator >> (std::istream &is, Backend &backend)
	{
		std::string token;

		is >> token;
		boost::to_upper(token);
		if (token == "LOCAL")
		{
			backend = Backend::Local;
		}
		else if (token == "REMOTE")
		{
			backend = Backend::Remote;
		}
		else if (token == "DISTRIBUTED")
		{
			backend = Backend::Distributed;
		}
		else
		{
			throw boost::program_options::invalid_option_value(token);
		}
		return is;
	}



}

int main(int argc, char *argv[])
{
	try
	{
	//	Sleep(10000);
		DEBUG_LOGGER <<   argv[0] << " INITIALIZE" ;
		boost::program_options::options_description desc("Allowed options");
		desc.add_options()
			("help,h", "produce help message")
			("exit_on_error,e", boost::program_options::value<bool>()->default_value(true), "Exit on error flag")
			("index,i", boost::program_options::value<std::vector<unsigned int>>()->multitoken()->composing()->default_value({ (unsigned int)0 }), "Local devices indicex. 0 for cpu (default), 1 for first gpu, 2 for second gpu ... ")
			("port,p", boost::program_options::value<unsigned short>()->default_value(12345), "TCP port")
			("host,h", boost::program_options::value<std::string>()->default_value("127.0.0.1"), "hostname or IPv4 address")
			("backend,b", boost::program_options::value<Backend>()->default_value(Backend::Local), "Backend type [local|remote|distributed]")
			("logging,l", boost::program_options::value<std::string>()->default_value("INFORMATION"), "Logging severity level filtering [TRACE|DEBUG|INFORMATION|WARNING|ERROR]")

			("filename,f", boost::program_options::value<std::string>(), "Scenario filename")
			;

		boost::program_options::variables_map vm;
		boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);
		boost::program_options::notify(vm);

		if (vm.count("help"))
		{
			std::cerr <<   desc << "\n";
			return 1;
		}


		auto severity = vm["logging"].as<std::string>();
		auto exit_on_error = vm["exit_on_error"].as<bool>();
		boost::to_upper(severity);
		if (severity == "ERROR")
		{
			TRN4CPP::Logging::Severity::Error::setup(exit_on_error);
		}
		else if (severity == "WARNING")
		{
			TRN4CPP::Logging::Severity::Warning::setup(exit_on_error);
		}
		else if (severity == "INFORMATION")
		{
			TRN4CPP::Logging::Severity::Information::setup(exit_on_error);
		}
		else if (severity == "DEBUG")
		{
			TRN4CPP::Logging::Severity::Debug::setup(exit_on_error);
		}
		else if (severity == "TRACE")
		{
			TRN4CPP::Logging::Severity::Trace::setup(exit_on_error);
		}
		else
		{
			throw std::invalid_argument("Unexpected token " + severity);
		}

		

		auto backend = vm["backend"].as<Backend>();
		INFORMATION_LOGGER << "Backend " << backend << " selected";


		auto filename = vm["filename"].as<std::string>();


			INFORMATION_LOGGER << "Initializing engine";
			switch (backend)
			{
				case Backend::Local:
				{
					auto index_list = vm["index"].as<std::vector<unsigned int>>();
					TRN4CPP::Engine::Backend::Local::initialize(
					{
						index_list
					});
				}
				break;
				case Backend::Remote:
				{
					auto host = vm["host"].as<std::string>();
					auto port = vm["port"].as<unsigned short>();
					TRN4CPP::Engine::Backend::Remote::initialize(host, port);
				}
				break;
				case Backend::Distributed:
				{
					TRN4CPP::Engine::Backend::Distributed::initialize(argc, argv);
				}
				break;
			}
			INFORMATION_LOGGER << "Computing scenario " << filename << " ...";
			TRN4CPP::Simulation::compute(filename);
			INFORMATION_LOGGER << "Scenario " << filename << " successfully computed";
			TRN4CPP::Engine::uninitialize();
			INFORMATION_LOGGER << "Engine uninitialized";


	
		

		INFORMATION_LOGGER << "Exiting";
		return 0;
	}
	catch (std::exception &e)
	{
		ERROR_LOGGER << e.what() ;
		return -1;
	}

}
