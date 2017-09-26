#include "stdafx.h"
#include "Callbacks.h"
#include "Engine/Frontend.h"

extern std::shared_ptr<TRN::Engine::Frontend> frontend;

std::function<void(const unsigned int &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols)> on_measurement_readout_mean_square_error;
std::function<void(const unsigned int &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols)> on_measurement_readout_frechet_distance;
std::function<void(const unsigned int &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols)> on_measurement_position_mean_square_error;
std::function<void(const unsigned int &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols)> on_measurement_position_frechet_distance;
std::function<void(const unsigned int &id, const std::string &phase, const size_t &batch_size, const size_t &cycles, const float &gflops, const float &seconds)> on_performances;
std::function<void(const unsigned int &id, const std::string &phase, const std::string &label, const std::size_t &batch, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &samples, const std::size_t &rows, const std::size_t &cols)> on_states;
std::function<void(const unsigned int &id, const std::string &phase, const std::string &label, const std::size_t &batch, const std::size_t &trial, const std::vector<float> &samples, const std::size_t &rows, const std::size_t &cols)> on_weights;
std::function<void(const unsigned int &id, const std::size_t &trial, const std::vector<int> &offsets, const std::vector<int> &durations)> on_scheduling;
std::function<void(const unsigned int &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols)> on_position;
std::function<void(const unsigned int &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &stimulus, const std::size_t &rows, const std::size_t &cols)> on_stimulus;

void TRN4CPP::Simulation::Recording::States::install(const std::function<void(const unsigned int &id, const std::string &phase, const std::string &label, const std::size_t &batch, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &samples, const std::size_t &rows, const std::size_t &cols)> &functor)
{
	if (on_states)
		throw std::runtime_error("States functor is already installed");
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	on_states = functor;
}
void TRN4CPP::Simulation::Recording::Weights::install(const std::function<void(const unsigned int &id, const std::string &phase, const std::string &label, const std::size_t &batch, const std::size_t &trial, const std::vector<float> &weights, const std::size_t &rows, const std::size_t &cols)> &functor)
{
	if (on_weights)
		throw std::runtime_error("Weights functor is already installed");
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	on_weights = functor;
}
void TRN4CPP::Simulation::Recording::Performances::install(const std::function<void(const unsigned int &id, const std::string &phase, const std::size_t &batch_size, const size_t &cycles, const float &gflops, const float &seconds)> &functor)
{
	if (on_performances)
		throw std::runtime_error("Performances functor is already installed");
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	on_performances = functor;
}
void TRN4CPP::Simulation::Recording::Scheduling::install(const std::function<void(const unsigned int &id, const std::size_t &trial, const std::vector<int> &offsets, const std::vector<int> &durations)> &functor)
{
	if (on_scheduling)
		throw std::runtime_error("Scheduling functor is already installed");
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	on_scheduling = functor;
}
void TRN4CPP::Simulation::Measurement::Readout::MeanSquareError::install(const std::function<void(const unsigned int &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const std::size_t &cols)> &functor)
{
	if (on_measurement_readout_mean_square_error)
		throw std::runtime_error("Readout mean square error functor is already installed");
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	on_measurement_readout_mean_square_error = functor;
}
void TRN4CPP::Simulation::Measurement::Readout::FrechetDistance::install(const std::function<void(const unsigned int &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const std::size_t &cols)> &functor)
{
	if (on_measurement_readout_frechet_distance)
		throw std::runtime_error("Readout frechet distance functor is already installed");
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	on_measurement_readout_frechet_distance = functor;
}
void TRN4CPP::Simulation::Measurement::Position::MeanSquareError::install(const std::function<void(const unsigned int &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const std::size_t &cols)> &functor)
{
	if (on_measurement_position_mean_square_error)
		throw std::runtime_error("Position mean square error functor is already installed");
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	on_measurement_position_mean_square_error = functor;
}
void TRN4CPP::Simulation::Measurement::Position::FrechetDistance::install(const std::function<void(const unsigned int &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const std::size_t &cols)> &functor)
{
	if (on_measurement_position_frechet_distance)
		throw std::runtime_error("Position frechet distance functor is already installed");
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	on_measurement_position_frechet_distance = functor;
}

void TRN4CPP::Simulation::Loop::Readout::install(const std::function<void(const unsigned int &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &prediction, const std::size_t &rows, const std::size_t &cols)> &request,
	std::function<void(const unsigned int &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &stimulus, const std::size_t &rows, const std::size_t &cols)> &reply)
{
	if (on_stimulus)
		throw std::runtime_error("Stimulus functor is already installed");
	on_stimulus = request;
	reply = std::bind(&TRN::Engine::Broker::notify_stimulus, frontend, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5, std::placeholders::_6);
}
void TRN4CPP::Simulation::Loop::Position::install(const std::function<void(const unsigned int &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &prediction, const std::size_t &rows, const std::size_t &cols)> &request,
	std::function<void(const unsigned int &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &stimulus, const std::size_t &rows, const std::size_t &cols)> &reply)
{
	if (on_position)
		throw std::runtime_error("Position functor is already installed");
	on_position = request;
	reply = std::bind(&TRN::Engine::Broker::notify_position, frontend, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5, std::placeholders::_6);
}