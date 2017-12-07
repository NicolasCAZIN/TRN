

#include <memory>
#include <condition_variable>
#include <vector>
#include <random>
#include <mutex>
#include <thread>
#include <algorithm>    // copy
#include <iostream>
#include <fstream>
#include <iterator>
#include <functional>
#include <string>
#include <list>
#include <mutex>
#include <queue>
#include <iomanip>
#include <boost/asio.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/program_options.hpp>
#include <boost/program_options/errors.hpp>
#include <boost/optional.hpp>

#include "ViewModel/Communicator.h"
#include "ViewModel/Node.h"
#include "Remote/Communicator.h"
#include "Network/Acceptor.h"
#include "Network/Manager.h"
#include "Helper/Visitor.h"
#include "Helper/Queue.h"
#include "Helper/Adapter.h"
#include "Helper/Logger.h"
struct Node
{
	std::vector<int> ranks;
	std::string host;
	unsigned int index;
	std::string name;
};



std::map<std::pair<std::string, unsigned int>, Node> nodes;

std::map<int, std::pair<std::string, unsigned int>> processor_node;
std::map<unsigned long long, int> simulation_processor;
std::map<std::string, std::vector<int>> host_rank;
std::mutex processor;
std::set<int> ranks;

static void on_processor(const int &rank, const std::string &host, const unsigned int &index, const std::string &name)
{

	std::unique_lock<std::mutex> lock(processor);

	ranks.insert(rank);
	auto key = std::make_pair(host, index);
	if (nodes.find(key) == nodes.end())
	{
		nodes[key].host = host;
		nodes[key].name = name;
		nodes[key].index = index;
	}
	nodes[key].ranks.push_back(rank);
	processor_node[rank] = key;
	host_rank[host].push_back(rank);
}

enum Backend
{
	Distributed,
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
	std::istream &operator >> (std::istream &is, Backend &backend)
	{
		std::string token;

		is >> token;
		boost::to_upper(token);
		if (token == "LOCAL")
		{
			backend = Backend::Local;
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

class Mediator : public TRN::Helper::Visitor<TRN::Engine::Proxy>
{
	private :
		std::map<std::shared_ptr<TRN::Engine::Proxy>, std::function<void(void)>> association;
public :
	void visit(std::shared_ptr<TRN::Engine::Proxy> &proxy)
	{
		if (association.find(proxy) == association.end())
			throw std::runtime_error("proxy was not registered");
	
		association[proxy]();
	
		association.erase(proxy);
	}

	void bind(const std::shared_ptr<TRN::Engine::Proxy> &proxy, const std::shared_ptr<TRN::Network::Peer> &peer, const std::function <void(const std::shared_ptr<TRN::Network::Peer> &peer)> &on_terminated)
	{
		if (association.find(proxy) != association.end())
			throw std::runtime_error("proxy is already registered");
		association[proxy] = [=]() 
		{
			on_terminated(peer); 
		};

	}
};

int main(int argc, char *argv[])
{
	try
	{
		boost::program_options::options_description desc("Allowed options");
		desc.add_options()
			("help,h", "produce help message")
			("index,i", boost::program_options::value<std::vector<unsigned int>>()->multitoken()->composing()->default_value({ (unsigned int)0 }), "Local devices indicex. 0 for cpu (default), 1 for first gpu, 2 for second gpu ... ")
			("port,p", boost::program_options::value<unsigned short>()->default_value(12345), "TCP port")
			("host,h", boost::program_options::value<std::string>()->default_value("127.0.0.1"), "hostname or IPv4 address")
			("backend,b", boost::program_options::value<Backend>()->default_value(Backend::Local), "Backend type [local|distributed]")
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

		std::shared_ptr<TRN::Engine::Communicator> worker_communicator;
	
		switch (vm["backend"].as<Backend>())
		{
			case Backend::Local:
			{
				INFORMATION_LOGGER <<   "Local backend selected" ;

				auto index_list = vm["index"].as<std::vector<unsigned int>>();
				worker_communicator = TRN::ViewModel::Communicator::Local::create(index_list);
			}
			break;
			case Backend::Distributed:
			{
				INFORMATION_LOGGER <<   "Distributed backend selected" ;
				worker_communicator = TRN::ViewModel::Communicator::Distributed::create(argc, argv);
			}
			break;
		}

		if (!worker_communicator)
		{
			throw std::runtime_error("No backend were selected");
		}
		
		auto dispatcher = TRN::Engine::Dispatcher::create(worker_communicator);
		dispatcher->start();

		auto host = vm["host"].as<std::string>();
		auto port = vm["port"].as<unsigned short>();
	
		auto mediator = std::make_shared<Mediator>();
		
		auto acceptor = TRN::Network::Acceptor::create(host, port, [&](const std::shared_ptr<TRN::Network::Manager> &manager, const std::shared_ptr<TRN::Network::Connection> &connection, const std::function <void(const std::shared_ptr<TRN::Network::Peer> &peer)> &on_terminated)
		{
		

			class Adapter : public TRN::Helper::Adapter<TRN::Network::Peer, TRN::Engine::Proxy>
			{
		

			public :
				Adapter(const std::shared_ptr<TRN::Engine::Proxy> &proxy) :
					TRN::Helper::Adapter<TRN::Network::Peer, TRN::Engine::Proxy>::Adapter(proxy)
				{

				}

				void start() override
				{
					adaptee->start();
				}

				void stop() override
				{
					adaptee->dispose();
				}
			};

			auto &socket = connection->socket();
			assert(socket.is_open());
			INFORMATION_LOGGER <<   "SERVER local " << socket.local_endpoint().address().to_string() << ":" << socket.local_endpoint().port() <<
				" remote " << socket.remote_endpoint().address().to_string() << ":" << socket.remote_endpoint().port() ;

		

			connection->write(worker_communicator->size());
			
			auto client_communicator = TRN::Remote::Communicator::create(manager, connection, worker_communicator->rank(), worker_communicator->size());

			auto frontend_number = connection->get_id();
			dispatcher->register_frontend(frontend_number, client_communicator);

			auto proxy = TRN::ViewModel::Node::Proxy::create(client_communicator, worker_communicator, dispatcher, mediator, frontend_number);
			auto adapter = std::make_shared<Adapter>(proxy);
			dispatcher->attach(proxy);
			mediator->bind(proxy, adapter, [=](const std::shared_ptr<TRN::Network::Peer> &peer)
			{
			


				on_terminated(peer);
				manager->stop(connection);
				dispatcher->unregister_frontend(frontend_number);
				INFORMATION_LOGGER <<   "Connection #" << connection->get_id() << " stopped" ;
	
			});
			return adapter;
		});
	

		acceptor->run();
	
		dispatcher->dispose();

		return 0;
	}
	catch (std::exception &e)
	{

		ERROR_LOGGER << e.what() ;
		return -1;
	}
}
