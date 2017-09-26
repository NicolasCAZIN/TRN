#include "stdafx.h"
#include "Custom.h"
#include "Engine/Frontend.h"

extern std::shared_ptr<TRN::Engine::Frontend> frontend;

std::function<void(const unsigned int &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &primed, const std::vector<float> &predicted, const std::vector<float> &expected, const std::size_t &preamble, const std::size_t &pages, const std::size_t &rows, const  std::size_t &cols)> on_measurement_readout_custom;
std::function<void(const unsigned int &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &primed, const std::vector<float> &predicted, const std::vector<float> &expected, const std::size_t &preamble, const std::size_t &pages, const std::size_t &rows, const  std::size_t &cols)> on_measurement_position_custom;
std::function<void(const unsigned int &id, const unsigned long &seed, const std::size_t &trial, const std::vector<int> &offsets, const std::vector<int> &durations)> on_mutator;
std::function<void(const unsigned int &id, const unsigned long &seed, const std::size_t &trial, const std::vector<float> &elements, const std::size_t &rows, const std::size_t &cols, const std::vector<int> &offsets, const std::vector<int> &durations)> on_scheduler;
std::function<void(const unsigned int &id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const  std::size_t &cols)> on_feedforward;
std::function<void(const unsigned int &id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const  std::size_t &cols)> on_feedback;
std::function<void(const unsigned int &id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const  std::size_t &cols)> on_readout;
std::function<void(const unsigned int &id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> on_recurrent;

void TRN4CPP::Simulation::Scheduler::Custom::install(const std::function<void(const unsigned int &id, const unsigned long &seed, const std::size_t &trial, const std::vector<float> &elements, const std::size_t &rows, const std::size_t &cols, const std::vector<int> &offsets, const std::vector<int> &durations)> &request,
	std::function<void(const unsigned int &id, const std::size_t &trial, const std::vector<int> &offsets, const std::vector<int> &durations)> &reply)
{
	if (on_scheduler)
		throw std::runtime_error("Scheduler functor is already installed");
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	on_scheduler = request;
	reply = std::bind(&TRN::Engine::Broker::notify_scheduler, frontend, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4);
}

void TRN4CPP::Simulation::Reservoir::Weights::Feedforward::Custom::install(const std::function<void(const unsigned int &id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &request,
	std::function<void(const unsigned int &id, const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &reply)
{
	if (on_feedforward)
		throw std::runtime_error("Feedforward functor is already installed");
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	on_feedforward = request;
	reply = std::bind(&TRN::Engine::Broker::notify_feedforward, frontend, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5);
}
void TRN4CPP::Simulation::Reservoir::Weights::Feedback::Custom::install(const std::function<void(const unsigned int &id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &request,
	std::function<void(const unsigned int &id, const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &reply)
{
	if (on_feedback)
		throw std::runtime_error("Feedbackfunctor is already installed");
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	on_feedback = request;
	reply = std::bind(&TRN::Engine::Broker::notify_feedback, frontend, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5);
}
void TRN4CPP::Simulation::Reservoir::Weights::Recurrent::Custom::install(const std::function<void(const unsigned int &id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &request,
	std::function<void(const unsigned int &id, const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &reply)
{
	if (on_recurrent)
		throw std::runtime_error("Recurrent functor is already installed");
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	on_recurrent = request;
	reply = std::bind(&TRN::Engine::Broker::notify_recurrent, frontend, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5);
}
void TRN4CPP::Simulation::Reservoir::Weights::Readout::Custom::install(const std::function<void(const unsigned int &id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &request,
	std::function<void(const unsigned int &id, const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &reply)
{
	if (on_readout)
		throw std::runtime_error("Readout functor is already installed");
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	on_readout = request;
	reply = std::bind(&TRN::Engine::Broker::notify_readout, frontend, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5);
}
void TRN4CPP::Simulation::Measurement::Position::Custom::install(const std::function<void(const unsigned int &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &predicted, const std::vector<float> &primed, const std::vector<float> &expected, const std::size_t &preamble, const std::size_t &pages, const std::size_t &rows, const std::size_t &cols)> &functor)
{
	if (on_measurement_position_custom)
		throw std::runtime_error("Position custom functor is already installed");
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	on_measurement_position_custom = functor;
}
void TRN4CPP::Simulation::Measurement::Readout::Custom::install(const std::function<void(const unsigned int &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &predicted, const std::vector<float> &primed, const std::vector<float> &expected, const std::size_t &preamble, const std::size_t &pages, const std::size_t &rows, const std::size_t &cols)> &functor)
{
	if (on_measurement_readout_custom)
		throw std::runtime_error("Readout custom functor is already installed");
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	on_measurement_readout_custom = functor;
}
