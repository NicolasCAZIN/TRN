#include "stdafx.h"
#include "TRN4CPP.h"
#include "Basic.h"
#include "Custom.h"
#include "Callbacks.h"
#include "Sequences.h"
#include "Search.h"

#include "ViewModel/Communicator.h"
#include "ViewModel/Frontend.h"

#include "Helper/Logger.h"

const std::string TRN4CPP::Simulation::DEFAULT_TAG = "";
const std::string TRN4CPP::Engine::Backend::Remote::DEFAULT_HOST = "127.0.0.1";
const unsigned short TRN4CPP::Engine::Backend::Remote::DEFAULT_PORT = 12345;
std::shared_ptr<TRN::Engine::Frontend> frontend;

extern boost::shared_ptr<TRN4CPP::Plugin::Custom::Interface> custom;
extern std::vector<boost::shared_ptr<TRN4CPP::Plugin::Callbacks::Interface>> callbacks;

extern std::function<void()> on_completed;
extern std::function<void(const unsigned long long &simulation_id, const std::size_t &counter, const bool &success, const std::string &cause)> on_ack;
extern std::function<void(const int &rank, const std::string &host, const unsigned int &index, const std::string &name)> on_processor;
extern std::function<void(const unsigned long long &simulation_id, const int &rank)> on_allocated;
extern std::function<void(const unsigned long long &simulation_id, const int &rank)> on_deallocated;
extern std::function<void(const int &rank)> on_quit;
extern std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id)> on_trained;
extern std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id)> on_primed;
extern std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id)> on_tested;
extern std::function<void(const unsigned long long &simulation_id)> on_configured;
extern std::function<void(const std::string &message) > on_error;
extern std::function<void(const std::string &message) > on_information;
extern std::function<void(const std::string &message) > on_warning;

extern std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &primed, const std::vector<float> &predicted, const std::vector<float> &expected, const std::size_t &preamble, const std::size_t &pages, const std::size_t &rows, const  std::size_t &cols)> on_measurement_readout_raw;
extern std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &primed, const std::vector<float> &predicted, const std::vector<float> &expected, const std::size_t &preamble, const std::size_t &pages, const std::size_t &rows, const  std::size_t &cols)> on_measurement_position_raw;
extern std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const unsigned long &seed, const std::vector<int> &offsets, const std::vector<int> &durations)> on_mutator;
extern std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const unsigned long &seed, const std::vector<float> &elements, const std::size_t &rows, const std::size_t &cols, const std::vector<int> &offsets, const std::vector<int> &durations)> on_scheduler;
extern std::function<void(const unsigned long long &simulation_id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const  std::size_t &cols)> on_feedforward;
extern std::function<void(const unsigned long long &simulation_id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const  std::size_t &cols)> on_readout;
extern std::function<void(const unsigned long long &simulation_id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> on_recurrent;

extern std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols)> on_measurement_readout_mean_square_error;
extern std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols)> on_measurement_readout_frechet_distance;
extern std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols)> on_measurement_position_mean_square_error;
extern std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols)> on_measurement_position_frechet_distance;
extern std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::string &phase, const float &cycles_per_second, const float &gflops_per_second)> on_performances;
extern std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::string &phase, const std::string &label, const std::size_t &batch, const std::vector<float> &samples, const std::size_t &rows, const std::size_t &cols)> on_states;
extern std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::string &phase, const std::string &label, const std::size_t &batch, const std::vector<float> &samples, const std::size_t &rows, const std::size_t &cols)> on_weights;
extern std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<int> &offsets, const std::vector<int> &durations)> on_scheduling;
extern std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols)> on_position;
extern std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &stimulus, const std::size_t &rows, const std::size_t &cols)> on_stimulus;


void TRN4CPP::Logging::Severity::Trace::setup(const bool &exit_on_error)
{
	TRN::Helper::Logger::setup(TRN::Helper::Logger::TRACE_LEVEL, exit_on_error);
}
void TRN4CPP::Logging::Severity::Debug::setup(const bool &exit_on_error)
{
	TRN::Helper::Logger::setup(TRN::Helper::Logger::DEBUG_LEVEL, exit_on_error);
}
void TRN4CPP::Logging::Severity::Information::setup(const bool &exit_on_error)
{
	TRN::Helper::Logger::setup(TRN::Helper::Logger::INFORMATION_LEVEL, exit_on_error);
}
void TRN4CPP::Logging::Severity::Warning::setup(const bool &exit_on_error)
{
	TRN::Helper::Logger::setup(TRN::Helper::Logger::WARNING_LEVEL, exit_on_error);
}
void TRN4CPP::Logging::Severity::Error::setup(const bool &exit_on_error)
{
	TRN::Helper::Logger::setup(TRN::Helper::Logger::ERROR_LEVEL, exit_on_error);
}

void  TRN4CPP::Simulation::encode(const unsigned short &frontend, const unsigned short &condition_number, const unsigned int &batch_number, unsigned long long &simulation_id)
{
	TRACE_LOGGER;
	TRN::Engine::encode(frontend, condition_number, batch_number, simulation_id);
}
void TRN4CPP::Simulation::decode(const unsigned long long &simulation_id, unsigned short &frontend, unsigned short &condition_number, unsigned int &batch_number)
{
	TRACE_LOGGER;
	TRN::Engine::decode(simulation_id,frontend, condition_number, batch_number);
}

void TRN4CPP::Simulation::Evaluation::encode(const unsigned short &trial_number, const unsigned short &train_number, const unsigned short &test_number, const unsigned short &repeat_number, unsigned long long &evaluation_id)
{
	TRACE_LOGGER;
	TRN::Engine::Evaluation::encode(trial_number, train_number, test_number, repeat_number, evaluation_id);
}
void TRN4CPP::Simulation::Evaluation::decode(const unsigned long long &evaluation_id, unsigned short &trial_number, unsigned short &train_number, unsigned short &test_number, unsigned short &repeat_number)
{
	TRACE_LOGGER;
	TRN::Engine::Evaluation::decode(evaluation_id,trial_number, train_number, test_number, repeat_number);
}

static void initialize_frontend(const std::shared_ptr<TRN::Engine::Communicator> &communicator = TRN::ViewModel::Communicator::Local::create({ 0 }))
{
	if (frontend)
		throw std::runtime_error("A Frontend is already setup");

	frontend = TRN::ViewModel::Frontend::create(communicator);
	frontend->install_ack(on_ack);
	frontend->install_completed(on_completed);
	frontend->install_processor(on_processor);
	frontend->install_allocated(on_allocated);
	frontend->install_deallocated(on_deallocated);
	frontend->install_quit(on_quit);
	frontend->install_trained(on_trained);
	frontend->install_tested(on_tested);
	frontend->install_primed(on_primed);
	frontend->install_configured(on_configured);
	frontend->install_error(on_error);
	frontend->install_information(on_information);
	frontend->install_warning(on_warning);
	frontend->start();
	INFORMATION_LOGGER <<   "TRN successfully initialized" ;
}
void TRN4CPP::Engine::Backend::Local::initialize(const std::vector<unsigned int> &indices)
{
	TRACE_LOGGER;
	INFORMATION_LOGGER <<   "Initializing TRN with a local backend" ;
	initialize_frontend(TRN::ViewModel::Communicator::Local::create(indices));
}
void TRN4CPP::Engine::Backend::Remote::initialize(const std::string &host, const unsigned short &port)
{
	TRACE_LOGGER;
	INFORMATION_LOGGER <<   "Initializing TRN with a remote backend" ;
	initialize_frontend(TRN::ViewModel::Communicator::Remote::create(host, port));
}
void TRN4CPP::Engine::Backend::Distributed::initialize(int argc, char *argv[])
{
	TRACE_LOGGER;
	INFORMATION_LOGGER <<   "Initializing TRN with a distributed backend" ;
	initialize_frontend(TRN::ViewModel::Communicator::Distributed::create(argc, argv));
}

std::string getEnvVar(std::string const& key)
{
	char const* val = std::getenv(key.c_str());
	return val == NULL ? std::string() : std::string(val);
}

void TRN4CPP::Engine::initialize()
{
	TRACE_LOGGER;
	std::string initialize = getEnvVar("TRN_INITIALIZE");
	if (!initialize.empty())
	{
		INFORMATION_LOGGER <<   "Initializing TRN" ;
		std::vector<std::string> tokens;

		std::string arguments = getEnvVar("TRN_INITIALIZE_DISTRIBUTED");
		if (!arguments.empty())
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
			std::string  host_port = getEnvVar("TRN_INITIALIZE_REMOTE");
			if (!host_port.empty())
			{
				boost::split(tokens, host_port, boost::is_any_of("@: "));
				std::string host;
				unsigned short port;
				switch (tokens.size())
				{
					case 2:
						
						if (tokens[0].empty())
							host = TRN4CPP::Engine::Backend::Remote::DEFAULT_HOST;
						else
							host = tokens[0];
						if (tokens[1].empty())
							port = TRN4CPP::Engine::Backend::Remote::DEFAULT_PORT;
						else
							port = boost::lexical_cast<unsigned short>(tokens[1]);

						break;
					default :
						throw std::runtime_error("Malformed host:port enironment variable");
				}
				TRN4CPP::Engine::Backend::Remote::initialize(host, port);
			}
			else
			{
				std::string devices = getEnvVar("TRN_INITIALIZE_LOCAL");
			
				std::vector<unsigned int> indices;

				boost::split(tokens, devices, boost::is_any_of(",;: "));
				std::transform(tokens.begin(), tokens.end(), indices.begin(), boost::lexical_cast<unsigned int, std::string>);
				TRN4CPP::Engine::Backend::Local::initialize(indices);
			}
		}
	}
	else
	{
		INFORMATION_LOGGER <<   "TRN manual initialization is required" ;
	}
}
void TRN4CPP::Engine::uninitialize()
{
	TRACE_LOGGER;

	if (frontend)
	{
		frontend->quit();
		frontend.reset();
	}


	TRN4CPP::Plugin::Sequences::uninitialize();
	TRN4CPP::Plugin::Custom::uninitialize();
	TRN4CPP::Plugin::Callbacks::uninitialize();
	TRN4CPP::Plugin::Search::uninitialize();


	on_feedforward = NULL;
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
	on_measurement_position_raw = NULL;
	on_measurement_position_frechet_distance = NULL;
	on_measurement_position_mean_square_error = NULL;
	on_measurement_readout_raw = NULL;
	on_measurement_readout_frechet_distance = NULL;
	on_measurement_readout_mean_square_error = NULL;
}
