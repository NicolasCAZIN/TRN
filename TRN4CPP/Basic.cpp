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
