#include "stdafx.h"
#include "TRN4CPP.h"
#include "Basic.h"
#include "Custom.h"
#include "Callbacks.h"

#include "ViewModel/Communicator.h"
#include "ViewModel/Frontend.h"

const std::string TRN4CPP::Simulation::DEFAULT_TAG = "";
const std::string TRN4CPP::Engine::Backend::Remote::DEFAULT_HOST = "127.0.0.1";
const unsigned short TRN4CPP::Engine::Backend::Remote::DEFAULT_PORT = 12345;
std::shared_ptr<TRN::Engine::Frontend> frontend;

extern boost::shared_ptr<TRN4CPP::Plugin::Simplified::Interface> simplified;
extern boost::shared_ptr<TRN4CPP::Plugin::Custom::Interface> custom;
extern std::vector<boost::shared_ptr<TRN4CPP::Plugin::Callbacks::Interface>> callbacks;

extern std::function<void()> on_completed;
extern std::function<void(const unsigned long long &id, const std::size_t &number, const bool &success, const std::string &cause)> on_ack;
extern std::function<void(const int &rank, const std::string &host, const unsigned int &index, const std::string &name)> on_processor;
extern std::function<void(const unsigned long long &id, const int &rank)> on_allocated;
extern std::function<void(const unsigned long long &id, const int &rank)> on_deallocated;
extern std::function<void(const int &rank)> on_quit;
extern std::function<void(const unsigned long long &id)> on_trained;
extern std::function<void(const unsigned long long &id)> on_primed;
extern std::function<void(const unsigned long long &id)> on_tested;
extern std::function<void(const unsigned long long &id)> on_configured;
extern std::function<void(const std::string &message) > on_error;
extern std::function<void(const std::string &message) > on_information;
extern std::function<void(const std::string &message) > on_warning;

extern std::function<void(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &primed, const std::vector<float> &predicted, const std::vector<float> &expected, const std::size_t &preamble, const std::size_t &pages, const std::size_t &rows, const  std::size_t &cols)> on_measurement_readout_raw;
extern std::function<void(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &primed, const std::vector<float> &predicted, const std::vector<float> &expected, const std::size_t &preamble, const std::size_t &pages, const std::size_t &rows, const  std::size_t &cols)> on_measurement_position_raw;
extern std::function<void(const unsigned long long &id, const unsigned long &seed, const std::size_t &trial, const std::vector<int> &offsets, const std::vector<int> &durations)> on_mutator;
extern std::function<void(const unsigned long long &id, const unsigned long &seed, const std::size_t &trial, const std::vector<float> &elements, const std::size_t &rows, const std::size_t &cols, const std::vector<int> &offsets, const std::vector<int> &durations)> on_scheduler;
extern std::function<void(const unsigned long long &id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const  std::size_t &cols)> on_feedforward;
extern std::function<void(const unsigned long long &id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const  std::size_t &cols)> on_feedback;
extern std::function<void(const unsigned long long &id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const  std::size_t &cols)> on_readout;
extern std::function<void(const unsigned long long &id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> on_recurrent;

extern std::function<void(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols)> on_measurement_readout_mean_square_error;
extern std::function<void(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols)> on_measurement_readout_frechet_distance;
extern std::function<void(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols)> on_measurement_position_mean_square_error;
extern std::function<void(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols)> on_measurement_position_frechet_distance;
extern std::function<void(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::string &phase, const float &cycles_per_second, const float &gflops_per_second)> on_performances;
extern std::function<void(const unsigned long long &id, const std::string &phase, const std::string &label, const std::size_t &batch, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &samples, const std::size_t &rows, const std::size_t &cols)> on_states;
extern std::function<void(const unsigned long long &id, const std::string &phase, const std::string &label, const std::size_t &batch, const std::size_t &trial, const std::vector<float> &samples, const std::size_t &rows, const std::size_t &cols)> on_weights;
extern std::function<void(const unsigned long long &id, const std::size_t &trial, const std::vector<int> &offsets, const std::vector<int> &durations)> on_scheduling;
extern std::function<void(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols)> on_position;
extern std::function<void(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &stimulus, const std::size_t &rows, const std::size_t &cols)> on_stimulus;

union Identifier
{
	struct
	{
		unsigned long long simulation_number : 32;
		unsigned long long condition_number : 16;
		unsigned long long experiment_number : 16;
	};

	unsigned long long id;
};

void  TRN4CPP::Simulation::encode(const unsigned short &experiment_number, const unsigned short &condition_number, const unsigned int &simulation_number, unsigned long long &id)
{
	Identifier identifier;

	identifier.experiment_number = experiment_number;
	identifier.condition_number = condition_number;
	identifier.simulation_number = simulation_number;
	id = identifier.id;
}
void TRN4CPP::Simulation::decode(const unsigned long long &id, unsigned short &experiment_number, unsigned short &condition_number, unsigned int &simulation_number)
{
	Identifier identifier;

	identifier.id = id;
	experiment_number = identifier.experiment_number;
	condition_number = identifier.condition_number;
	simulation_number = identifier.simulation_number;
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

std::string getEnvVar(std::string const& key)
{
	char const* val = std::getenv(key.c_str());
	return val == NULL ? std::string() : std::string(val);
}

void TRN4CPP::Engine::initialize()
{
	std::string initialize = getEnvVar("TRN_INITIALIZE");
	if (!initialize.empty())
	{
		std::cout << "Initializing TRN" << std::endl;
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
		std::cout << "TRN manual initialization is required" << std::endl;
	}
}
void TRN4CPP::Engine::uninitialize()
{



	if (simplified)
	{
		simplified->uninitialize();
		simplified.reset();
	}
	if (custom)
	{
		custom->uninitialize();
		custom.reset();
	}
	if (!callbacks.empty())
	{
		for (auto callback : callbacks)
		{
			callback->uninitialize();
		}
		callbacks.clear();
	}



	if (frontend)
	{
		frontend->halt();
		frontend.reset();
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
	on_measurement_position_raw = NULL;
	on_measurement_position_frechet_distance = NULL;
	on_measurement_position_mean_square_error = NULL;
	on_measurement_readout_raw = NULL;
	on_measurement_readout_frechet_distance = NULL;
	on_measurement_readout_mean_square_error = NULL;
}
