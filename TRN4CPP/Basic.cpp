#include "stdafx.h"
#include "TRN4CPP.h"
#include "Basic.h"
#include "ViewModel/Communicator.h"
#include "ViewModel/Frontend.h"
#include "ViewModel/Executor.h"

const bool TRN4CPP::Engine::Execution::DEFAULT_BLOCKING = false;
const std::string TRN4CPP::Simulation::DEFAULT_TAG = "";
const std::string TRN4CPP::Engine::Backend::Remote::DEFAULT_HOST = "127.0.0.1";
const unsigned short TRN4CPP::Engine::Backend::Remote::DEFAULT_PORT = 12345;
std::shared_ptr<TRN::Engine::Frontend> frontend;
std::shared_ptr<TRN::Engine::Executor> executor;

extern std::function<void(const unsigned int &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &primed, const std::vector<float> &predicted, const std::vector<float> &expected, const std::size_t &preamble, const std::size_t &pages, const std::size_t &rows, const  std::size_t &cols)> on_measurement_readout_custom;
extern std::function<void(const unsigned int &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &primed, const std::vector<float> &predicted, const std::vector<float> &expected, const std::size_t &preamble, const std::size_t &pages, const std::size_t &rows, const  std::size_t &cols)> on_measurement_position_custom;
extern std::function<void(const unsigned int &id, const unsigned long &seed, const std::size_t &trial, const std::vector<int> &offsets, const std::vector<int> &durations)> on_mutator;
extern std::function<void(const unsigned int &id, const unsigned long &seed, const std::size_t &trial, const std::vector<float> &elements, const std::size_t &rows, const std::size_t &cols, const std::vector<int> &offsets, const std::vector<int> &durations)> on_scheduler;
extern std::function<void(const unsigned int &id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const  std::size_t &cols)> on_feedforward;
extern std::function<void(const unsigned int &id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const  std::size_t &cols)> on_feedback;
extern std::function<void(const unsigned int &id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const  std::size_t &cols)> on_readout;
extern std::function<void(const unsigned int &id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> on_recurrent;

extern std::function<void(const unsigned int &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols)> on_measurement_readout_mean_square_error;
extern std::function<void(const unsigned int &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols)> on_measurement_readout_frechet_distance;
extern std::function<void(const unsigned int &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols)> on_measurement_position_mean_square_error;
extern std::function<void(const unsigned int &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols)> on_measurement_position_frechet_distance;
extern std::function<void(const unsigned int &id, const std::string &phase, const size_t &batch_size, const size_t &cycles, const float &gflops, const float &seconds)> on_performances;
extern std::function<void(const unsigned int &id, const std::string &phase, const std::string &label, const std::size_t &batch, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &samples, const std::size_t &rows, const std::size_t &cols)> on_states;
extern std::function<void(const unsigned int &id, const std::string &phase, const std::string &label, const std::size_t &batch, const std::size_t &trial, const std::vector<float> &samples, const std::size_t &rows, const std::size_t &cols)> on_weights;
extern std::function<void(const unsigned int &id, const std::size_t &trial, const std::vector<int> &offsets, const std::vector<int> &durations)> on_scheduling;
extern std::function<void(const unsigned int &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols)> on_position;
extern std::function<void(const unsigned int &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &stimulus, const std::size_t &rows, const std::size_t &cols)> on_stimulus;

void TRN4CPP::Engine::Execution::initialize(const bool &blocking)
{
	if (executor)
		throw std::runtime_error("An executor is already setup");
	if (blocking)
	{
		executor = TRN::ViewModel::Executor::Blocking::create();
		std::cout << "TRN initialized with a Blocking executor" << std::endl;
	}
	else
	{
		executor = TRN::ViewModel::Executor::NonBlocking::create();
		std::cout << "TRN initialized with a NonBlocking executor" << std::endl;
	}
}
static void initialize_frontend(const std::shared_ptr<TRN::Engine::Communicator> &communicator = TRN::ViewModel::Communicator::Local::create({ 0 }))
{
	if (frontend)
		throw std::runtime_error("A Frontend is already setup");
	if (!executor)
	{
		std::cerr << "Executor is not ininitialized. Initializing non-blocking executor" << std::endl;
		TRN4CPP::Engine::Execution::initialize(false);
	}

	frontend = TRN::ViewModel::Frontend::create(communicator, executor);
	
	frontend->start();
	std::cout << "TRN successfully initialized" << std::endl;
}
void TRN4CPP::Engine::Backend::Local::initialize(const std::vector<unsigned int> &indices)
{
	std::cout << "Initializing TRN with a local backend" << std::endl;
	initialize_frontend(TRN::ViewModel::Communicator::Local::create(indices));
}
void TRN4CPP::Engine::Backend::Remote::initialize(const std::string &host, const unsigned short &port)
{
	std::cout << "Initializing TRN with a remote backend" << std::endl;
	initialize_frontend(TRN::ViewModel::Communicator::Remote::create(host, port));
}
void TRN4CPP::Engine::Backend::Distributed::initialize(int argc, char *argv[])
{
	std::cout << "Initializing TRN with a distributed backend" << std::endl;
	initialize_frontend(TRN::ViewModel::Communicator::Distributed::create(argc, argv));
}
void TRN4CPP::Engine::initialize()
{
	if (std::getenv("TRN_INITIALIZE"))
	{
		std::cout << "Initializing TRN" << std::endl;
		std::vector<std::string> tokens;

		auto blocking = std::getenv("TRN_INITIALIZE_EXECUTOR");
		if (!blocking)
			blocking = "false";
		boost::to_lower(blocking);
		boost::lexical_cast<bool>(blocking);
		TRN4CPP::Engine::Execution::initialize(blocking);
		auto arguments = std::getenv("TRN_INITIALIZE_DISTRIBUTED");
		if (arguments)
		{
			boost::split(tokens, arguments, boost::is_any_of(" "));
			int argc = tokens.size();
			char **argv = new char*[argc + 1];
			for (int k = 0; k < argc; k++)
				argv[k] = const_cast<char *>(tokens[k].c_str());
			argv[argc] = NULL;
			TRN4CPP::Engine::Backend::Distributed::initialize(argc, argv);

			delete[] argv;
		}
		else
		{
			auto host_port = std::getenv("TRN_INITIALIZE_REMOTE");
			if (host_port)
			{
				boost::split(tokens, host_port, boost::is_any_of("@: "));
				std::string host;
				unsigned short port;
				switch (tokens.size())
				{
					case 2:
						host = tokens[0];
						if (host.empty())
							host = "127.0.0.1";
						port = boost::lexical_cast<unsigned short>(tokens[1]);
						break;
					default :
						throw std::runtime_error("Malformed host:port enironment variable");
				}
				TRN4CPP::Engine::Backend::Remote::initialize(host, port);
			}
			else
			{
				auto devices = std::getenv("TRN_INITIALIZE_LOCAL");
				if (!devices)
					devices = "";

				std::vector<unsigned int> indices;

				boost::split(tokens, devices, boost::is_any_of(",;: "));
				std::transform(tokens.begin(), tokens.end(), indices.begin(), boost::lexical_cast<unsigned int, std::string>);
				TRN4CPP::Engine::Backend::Local::initialize(indices);
			}
		}
	}
	else
	{
		std::cout << "TRN manual initialization is required" << std::endl;
	}
}
void TRN4CPP::Engine::uninitialize()
{
	if (frontend)
	{
		frontend->halt();
		
		frontend.reset();
	}
	if (executor)
	{
		executor->terminate();
		executor.reset();
	}

	on_feedforward = NULL;
	on_feedback = NULL;
	on_recurrent = NULL;
	on_readout = NULL;
	on_mutator = NULL;
	on_scheduler = NULL;
	on_states = NULL;
	on_weights = NULL;
	on_performances = NULL;
	on_scheduling = NULL;
	on_position = NULL;
	on_stimulus = NULL;
	on_measurement_position_custom = NULL;
	on_measurement_position_frechet_distance = NULL;
	on_measurement_position_mean_square_error = NULL;
	on_measurement_readout_custom = NULL;
	on_measurement_readout_frechet_distance = NULL;
	on_measurement_readout_mean_square_error = NULL;
}
