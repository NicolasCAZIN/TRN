#include "stdafx.h"
#include "Callbacks.h"
#include "Engine/Frontend.h"
#include "Helper/Logger.h"
extern std::shared_ptr<TRN::Engine::Frontend> frontend;

std::function<void(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &primed, const std::vector<float> &predicted, const std::vector<float> &expected, const std::size_t &preamble, const std::size_t &pages, const std::size_t &rows, const  std::size_t &cols)> on_measurement_readout_raw;
std::function<void(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &primed, const std::vector<float> &predicted, const std::vector<float> &expected, const std::size_t &preamble, const std::size_t &pages, const std::size_t &rows, const  std::size_t &cols)> on_measurement_position_raw;
std::function<void(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols)> on_measurement_readout_mean_square_error;
std::function<void(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols)> on_measurement_readout_frechet_distance;
std::function<void(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols)> on_measurement_position_mean_square_error;
std::function<void(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols)> on_measurement_position_frechet_distance;
std::function<void(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::string &phase, const float &cycles_per_second, const float &gflops_per_second)> on_performances;
std::function<void(const unsigned long long &id, const std::string &phase, const std::string &label, const std::size_t &batch, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &samples, const std::size_t &rows, const std::size_t &cols)> on_states;
std::function<void(const unsigned long long &id, const std::string &phase, const std::string &label, const std::size_t &batch, const std::size_t &trial, const std::vector<float> &samples, const std::size_t &rows, const std::size_t &cols)> on_weights;
std::function<void(const unsigned long long &id, const std::size_t &trial, const std::vector<int> &offsets, const std::vector<int> &durations)> on_scheduling;

static std::recursive_mutex mutex;
static std::vector<boost::shared_ptr<TRN4CPP::Plugin::Callbacks::Interface>> callbacks;

static void callback_measurement_readout_raw(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &primed, const std::vector<float> &predicted, const std::vector<float> &expected, const std::size_t &preamble, const std::size_t &pages, const std::size_t &rows, const  std::size_t &cols)
{
	std::unique_lock<std::recursive_mutex> guard(mutex);
#pragma omp parallel for
	for (int k = 0; k < callbacks.size(); k++)
	{
		auto plugin = callbacks[k];
		plugin->callback_measurement_readout_raw(id, trial, evaluation, primed, predicted, expected, preamble, pages, rows, cols);
	}
}
static void  callback_measurement_position_raw(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &primed, const std::vector<float> &predicted, const std::vector<float> &expected, const std::size_t &preamble, const std::size_t &pages, const std::size_t &rows, const  std::size_t &cols)
{
	std::unique_lock<std::recursive_mutex> guard(mutex);
#pragma omp parallel for
	for (int k = 0; k < callbacks.size(); k++)
	{
		auto plugin = callbacks[k];
		plugin->callback_measurement_position_raw(id, trial, evaluation, primed, predicted, expected, preamble, pages, rows, cols);
	}
}
static void callback_measurement_readout_mean_square_error(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols)
{
	std::unique_lock<std::recursive_mutex> guard(mutex);
#pragma omp parallel for
	for (int k = 0; k < callbacks.size(); k++)
	{
		auto plugin = callbacks[k];
		plugin->callback_measurement_readout_mean_square_error(id, trial, evaluation, values, rows, cols);
	}
}
static void callback_measurement_readout_frechet_distance(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols)
{
	std::unique_lock<std::recursive_mutex> guard(mutex);
#pragma omp parallel for
	for (int k = 0; k < callbacks.size(); k++)
	{
		auto plugin = callbacks[k];
		plugin->callback_measurement_readout_frechet_distance(id, trial, evaluation, values, rows, cols);
	}
}
static void callback_measurement_position_mean_square_error(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols)
{
	std::unique_lock<std::recursive_mutex> guard(mutex);
#pragma omp parallel for
	for (int k = 0; k < callbacks.size(); k++)
	{
		auto plugin = callbacks[k];
		plugin->callback_measurement_position_mean_square_error(id, trial, evaluation, values, rows, cols);
	}
}
static void callback_measurement_position_frechet_distance(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols)
{
	std::unique_lock<std::recursive_mutex> guard(mutex);
#pragma omp parallel for
	for (int k = 0; k < callbacks.size(); k++)
	{
		auto plugin = callbacks[k];
		plugin->callback_measurement_position_frechet_distance(id, trial, evaluation, values, rows, cols);
	}
}
static void callback_performances(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::string &phase, const float &cycles_per_second, const float &gflops_per_second)
{
	std::unique_lock<std::recursive_mutex> guard(mutex);
#pragma omp parallel for
	for (int k = 0; k < callbacks.size(); k++)
	{
		auto plugin = callbacks[k];
		plugin->callback_performances(id, trial, evaluation, phase, cycles_per_second, gflops_per_second);
	}
}
static void callback_states(const unsigned long long &id, const std::string &phase, const std::string &label, const std::size_t &batch, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &samples, const std::size_t &rows, const std::size_t &cols)
{
	std::unique_lock<std::recursive_mutex> guard(mutex);
#pragma omp parallel for
	for (int k = 0; k < callbacks.size(); k++)
	{
		auto plugin = callbacks[k];
		plugin->callback_states(id, phase, label, batch, trial, evaluation, samples, rows, cols);
	}
}
static void callback_weights(const unsigned long long &id, const std::string &phase, const std::string &label, const std::size_t &batch, const std::size_t &trial, const std::vector<float> &samples, const std::size_t &rows, const std::size_t &cols)
{
	std::unique_lock<std::recursive_mutex> guard(mutex);
#pragma omp parallel for
	for (int k = 0; k < callbacks.size(); k++)
	{
		auto plugin = callbacks[k];
		plugin->callback_weights(id, phase, label, batch, trial, samples, rows, cols);
	}
}
static void callback_scheduling(const unsigned long long &id, const std::size_t &trial, const std::vector<int> &offsets, const std::vector<int> &durations)
{
	std::unique_lock<std::recursive_mutex> guard(mutex);
#pragma omp parallel for
	for (int k = 0; k < callbacks.size(); k++)
	{
		auto plugin = callbacks[k];
		plugin->callback_scheduling(id, trial, offsets, durations);
	}
}
void TRN4CPP::Plugin::Callbacks::append(const boost::shared_ptr<TRN4CPP::Plugin::Callbacks::Interface> &plugin)
{
	std::unique_lock<std::recursive_mutex> guard(mutex);
	TRACE_LOGGER;
	callbacks.push_back(plugin);
}

void TRN4CPP::Plugin::Callbacks::initialize(const std::string &library_path, const std::string &name, const std::map<std::string, std::string>  &arguments)
{
	std::unique_lock<std::recursive_mutex> guard(mutex);
	TRACE_LOGGER;
	if (callbacks.empty())
	{ 
		TRN4CPP::Simulation::Measurement::Readout::Raw::install(callback_measurement_readout_raw);
		TRN4CPP::Simulation::Measurement::Position::Raw::install(callback_measurement_position_raw);

		TRN4CPP::Simulation::Measurement::Position::MeanSquareError::install(callback_measurement_position_mean_square_error);
		TRN4CPP::Simulation::Measurement::Readout::MeanSquareError::install(callback_measurement_readout_mean_square_error);
		TRN4CPP::Simulation::Measurement::Position::FrechetDistance::install(callback_measurement_position_frechet_distance);
		TRN4CPP::Simulation::Measurement::Readout::FrechetDistance::install(callback_measurement_readout_frechet_distance);

		TRN4CPP::Simulation::Recording::Performances::install(callback_performances);
		TRN4CPP::Simulation::Recording::Scheduling::install(callback_scheduling);
		TRN4CPP::Simulation::Recording::States::install(callback_states);
		TRN4CPP::Simulation::Recording::Weights::install(callback_weights);
	}

	boost::filesystem::path path = library_path;
	path /= name;

	auto plugin = boost::dll::import<TRN4CPP::Plugin::Callbacks::Interface>(path, "plugin_callbacks", boost::dll::load_mode::append_decorations);
	plugin->initialize(arguments);
	callbacks.push_back(plugin);
	INFORMATION_LOGGER << "Callbacks plugin " << name << " loaded from path " << library_path;
}

void TRN4CPP::Plugin::Callbacks::uninitialize()
{
	std::unique_lock<std::recursive_mutex> guard(mutex);
	TRACE_LOGGER;

	if (!callbacks.empty())
	{
		for (auto callback : callbacks)
		{
			callback->uninitialize();
		}
		callbacks.clear();
	}

}

void TRN4CPP::Simulation::Measurement::Position::Raw::install(const std::function<void(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &predicted, const std::vector<float> &primed, const std::vector<float> &expected, const std::size_t &preamble, const std::size_t &pages, const std::size_t &rows, const std::size_t &cols)> &functor)
{
	std::unique_lock<std::recursive_mutex> guard(mutex);
	TRACE_LOGGER;
	if (on_measurement_position_raw)
		throw std::runtime_error("Position raw functor is already installed");
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	on_measurement_position_raw = functor;
}
void TRN4CPP::Simulation::Measurement::Readout::Raw::install(const std::function<void(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &predicted, const std::vector<float> &primed, const std::vector<float> &expected, const std::size_t &preamble, const std::size_t &pages, const std::size_t &rows, const std::size_t &cols)> &functor)
{
	std::unique_lock<std::recursive_mutex> guard(mutex);
	TRACE_LOGGER;
	if (on_measurement_readout_raw)
		throw std::runtime_error("Readout raw functor is already installed");
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	on_measurement_readout_raw = functor;
}


void TRN4CPP::Simulation::Recording::States::install(const std::function<void(const unsigned long long &id, const std::string &phase, const std::string &label, const std::size_t &batch, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &samples, const std::size_t &rows, const std::size_t &cols)> &functor)
{
	std::unique_lock<std::recursive_mutex> guard(mutex);
	TRACE_LOGGER;
	if (on_states)
		throw std::runtime_error("States functor is already installed");
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	on_states = functor;
}
void TRN4CPP::Simulation::Recording::Weights::install(const std::function<void(const unsigned long long &id, const std::string &phase, const std::string &label, const std::size_t &batch, const std::size_t &trial, const std::vector<float> &weights, const std::size_t &rows, const std::size_t &cols)> &functor)
{
	std::unique_lock<std::recursive_mutex> guard(mutex);
	TRACE_LOGGER;
	if (on_weights)
		throw std::runtime_error("Weights functor is already installed");
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	on_weights = functor;
}
void TRN4CPP::Simulation::Recording::Performances::install(const std::function<void(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::string &phase, const float &cycles_per_second, const float &gflops_per_second)> &functor)
{
	std::unique_lock<std::recursive_mutex> guard(mutex);
	TRACE_LOGGER;
	if (on_performances)
		throw std::runtime_error("Performances functor is already installed");
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	on_performances = functor;
}
void TRN4CPP::Simulation::Recording::Scheduling::install(const std::function<void(const unsigned long long &id, const std::size_t &trial, const std::vector<int> &offsets, const std::vector<int> &durations)> &functor)
{
	std::unique_lock<std::recursive_mutex> guard(mutex);
	TRACE_LOGGER;
	if (on_scheduling)
		throw std::runtime_error("Scheduling functor is already installed");
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	on_scheduling = functor;
}
void TRN4CPP::Simulation::Measurement::Readout::MeanSquareError::install(const std::function<void(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const std::size_t &cols)> &functor)
{
	std::unique_lock<std::recursive_mutex> guard(mutex);
	TRACE_LOGGER;
	if (on_measurement_readout_mean_square_error)
		throw std::runtime_error("Readout mean square error functor is already installed");
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	on_measurement_readout_mean_square_error = functor;
}
void TRN4CPP::Simulation::Measurement::Readout::FrechetDistance::install(const std::function<void(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const std::size_t &cols)> &functor)
{
	std::unique_lock<std::recursive_mutex> guard(mutex);
	TRACE_LOGGER;
	if (on_measurement_readout_frechet_distance)
		throw std::runtime_error("Readout frechet distance functor is already installed");
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	on_measurement_readout_frechet_distance = functor;
}
void TRN4CPP::Simulation::Measurement::Position::MeanSquareError::install(const std::function<void(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const std::size_t &cols)> &functor)
{
	std::unique_lock<std::recursive_mutex> guard(mutex);
	TRACE_LOGGER;
	if (on_measurement_position_mean_square_error)
		throw std::runtime_error("Position mean square error functor is already installed");
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	on_measurement_position_mean_square_error = functor;
}
void TRN4CPP::Simulation::Measurement::Position::FrechetDistance::install(const std::function<void(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const std::size_t &cols)> &functor)
{
	std::unique_lock<std::recursive_mutex> guard(mutex);
	TRACE_LOGGER;
	if (on_measurement_position_frechet_distance)
		throw std::runtime_error("Position frechet distance functor is already installed");
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	on_measurement_position_frechet_distance = functor;
}

