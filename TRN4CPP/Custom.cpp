#include "stdafx.h"
#include "Custom.h"
#include "Engine/Frontend.h"
#include "Helper/Logger.h"

extern std::shared_ptr<TRN::Engine::Frontend> frontend;
static boost::shared_ptr<TRN4CPP::Plugin::Custom::Interface> custom;
static std::recursive_mutex mutex;

std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols)> on_position;
std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &stimulus, const std::size_t &rows, const std::size_t &cols)> on_stimulus;
std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const unsigned long &seed, const std::vector<int> &offsets, const std::vector<int> &durations)> on_mutator;
std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const unsigned long &seed, const std::vector<float> &elements, const std::size_t &rows, const std::size_t &cols, const std::vector<int> &offsets, const std::vector<int> &durations)> on_scheduler;
std::function<void(const unsigned long long &simulation_id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const  std::size_t &cols)> on_feedforward;
std::function<void(const unsigned long long &simulation_id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const  std::size_t &cols)> on_readout;
std::function<void(const unsigned long long &simulation_id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> on_recurrent;

std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols)> reply_position;
std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols)> reply_stimulus;
std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<int> &offsets, const std::vector<int> &durations)> reply_scheduler;
std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<int> &offsets, const std::vector<int> &durations)> reply_mutator;
std::function<void(const unsigned long long &simulation_id, const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> reply_feedforward;
std::function<void(const unsigned long long &simulation_id, const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> reply_recurrent;
std::function<void(const unsigned long long &simulation_id, const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> reply_readout;

void TRN4CPP::Plugin::Custom::initialize(const std::string &library_path, const std::string &name, const std::map<std::string, std::string>  &arguments)
{
	std::unique_lock<std::recursive_mutex> guard(mutex);
	TRACE_LOGGER;
	if (custom)
		throw std::runtime_error("A plugin is already loaded");

	boost::filesystem::path path = library_path;
	path /= name;

	custom = boost::dll::import<TRN4CPP::Plugin::Custom::Interface>(path, "plugin_custom", boost::dll::load_mode::append_decorations);
	custom->initialize(arguments);

	TRN4CPP::Simulation::Encoder::Custom::install(std::bind(&TRN4CPP::Plugin::Custom::Interface::callback_position, custom, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5), reply_position);
	custom->install_position(reply_position);

	TRN4CPP::Simulation::Loop::Custom::install(std::bind(&TRN4CPP::Plugin::Custom::Interface::callback_stimulus, custom, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5), reply_stimulus);
	custom->install_stimulus(reply_stimulus);

	TRN4CPP::Simulation::Scheduler::Custom::install(std::bind(&TRN4CPP::Plugin::Custom::Interface::callback_scheduler, custom, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5, std::placeholders::_6, std::placeholders::_7, std::placeholders::_8), reply_scheduler);
	custom->install_scheduler(reply_scheduler);

	TRN4CPP::Simulation::Scheduler::Mutator::Custom::install(std::bind(&TRN4CPP::Plugin::Custom::Interface::callback_mutator, custom, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5), reply_mutator);
	custom->install_mutator(reply_mutator);

	TRN4CPP::Simulation::Reservoir::Weights::Feedforward::Custom::install(std::bind(&TRN4CPP::Plugin::Custom::Interface::callback_feedforward, custom, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5), reply_feedforward);
	custom->install_feedforward(reply_feedforward);

	TRN4CPP::Simulation::Reservoir::Weights::Recurrent::Custom::install(std::bind(&TRN4CPP::Plugin::Custom::Interface::callback_recurrent, custom, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5), reply_recurrent);
	custom->install_recurrent(reply_recurrent);
	TRN4CPP::Simulation::Reservoir::Weights::Readout::Custom::install(std::bind(&TRN4CPP::Plugin::Custom::Interface::callback_readout, custom, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5), reply_readout);
	custom->install_readout(reply_readout);
	INFORMATION_LOGGER << "Custom plugin " << name << " loaded from path " << library_path;
}
void TRN4CPP::Plugin::Custom::uninitialize()
{
	std::unique_lock<std::recursive_mutex> guard(mutex);
	if (custom)
	{
		custom->uninitialize();
		custom.reset();
	}
}
void TRN4CPP::Simulation::Scheduler::Custom::install(const std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const unsigned long &seed, const std::vector<float> &elements, const std::size_t &rows, const std::size_t &cols, const std::vector<int> &offsets, const std::vector<int> &durations)> &request,
	std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<int> &offsets, const std::vector<int> &durations)> &reply)
{
	std::unique_lock<std::recursive_mutex> guard(mutex);
	TRACE_LOGGER;
	if (on_scheduler)
		throw std::runtime_error("Scheduler functor is already installed");
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	on_scheduler = request;
	reply = std::bind(&TRN::Engine::Broker::notify_scheduler, frontend, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4);
}
void TRN4CPP::Simulation::Scheduler::Mutator::Custom::install(const std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const unsigned long &seed, const std::vector<int> &offsets, const std::vector<int> &durations)> &request,
	std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<int> &offsets, const std::vector<int> &durations)> &reply)
{
	std::unique_lock<std::recursive_mutex> guard(mutex);
	TRACE_LOGGER;
	if (on_mutator)
		throw std::runtime_error("Mutator functor is already installed");
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	on_mutator = request;
	reply = std::bind(&TRN::Engine::Broker::notify_mutator, frontend, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4);
}
void TRN4CPP::Simulation::Reservoir::Weights::Feedforward::Custom::install(const std::function<void(const unsigned long long &simulation_id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &request,
	std::function<void(const unsigned long long &simulation_id, const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &reply)
{
	std::unique_lock<std::recursive_mutex> guard(mutex);
	TRACE_LOGGER;
	if (on_feedforward)
		throw std::runtime_error("Feedforward functor is already installed");
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	on_feedforward = request;
	reply = std::bind(&TRN::Engine::Broker::notify_feedforward, frontend, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5);
}

void TRN4CPP::Simulation::Reservoir::Weights::Recurrent::Custom::install(const std::function<void(const unsigned long long &simulation_id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &request,
	std::function<void(const unsigned long long &simulation_id, const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &reply)
{
	std::unique_lock<std::recursive_mutex> guard(mutex);
	TRACE_LOGGER;
	if (on_recurrent)
		throw std::runtime_error("Recurrent functor is already installed");
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	on_recurrent = request;
	reply = std::bind(&TRN::Engine::Broker::notify_recurrent, frontend, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5);
}
void TRN4CPP::Simulation::Reservoir::Weights::Readout::Custom::install(const std::function<void(const unsigned long long &simulation_id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &request,
	std::function<void(const unsigned long long &simulation_id, const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &reply)
{
	std::unique_lock<std::recursive_mutex> guard(mutex);
	TRACE_LOGGER;
	if (on_readout)
		throw std::runtime_error("Readout functor is already installed");
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	on_readout = request;
	reply = std::bind(&TRN::Engine::Broker::notify_readout, frontend, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5);
}

void TRN4CPP::Simulation::Loop::Custom::install(const std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &prediction, const std::size_t &rows, const std::size_t &cols)> &request,
	std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &stimulus, const std::size_t &rows, const std::size_t &cols)> &reply)
{
	std::unique_lock<std::recursive_mutex> guard(mutex);
	TRACE_LOGGER;
	if (on_stimulus)
		throw std::runtime_error("Predicted readout functor is already installed");
	on_stimulus = request;
	reply = std::bind(&TRN::Engine::Broker::notify_stimulus, frontend, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5);
}
void TRN4CPP::Simulation::Encoder::Custom::install(const std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &prediction, const std::size_t &rows, const std::size_t &cols)> &request,
	std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols)> &reply)
{
	std::unique_lock<std::recursive_mutex> guard(mutex);
	TRACE_LOGGER;
	if (on_position)
		throw std::runtime_error("Position functor is already installed");
	on_position = request;
	reply = std::bind(&TRN::Engine::Broker::notify_position, frontend, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5);
}