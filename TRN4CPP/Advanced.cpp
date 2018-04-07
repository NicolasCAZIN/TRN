#include "stdafx.h"
#include "Advanced.h"


#include "ViewModel/Communicator.h"
#include "ViewModel/Frontend.h"
#include "Helper/Logger.h"

extern std::shared_ptr<TRN::Engine::Frontend> frontend;

std::function<void()> on_completed;
std::function<void(const unsigned long long &simulation_id, const std::size_t &counter, const bool &success, const std::string &cause)> on_ack;
std::function<void(const int &rank, const std::string &host, const unsigned int &index, const std::string &name)> on_processor;
std::function<void(const unsigned long long &simulation_id, const int &rank)> on_allocated;
std::function<void(const unsigned long long &simulation_id, const int &rank)> on_deallocated;
std::function<void(const int &rank)> on_quit;
std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id)> on_trained;
std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id)> on_primed;
std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id)> on_tested;
std::function<void(const unsigned long long &simulation_id)> on_configured;
std::function<void(const std::string &message) > on_error;
std::function<void(const std::string &message) > on_information;
std::function<void(const std::string &message) > on_warning;

void TRN4CPP::Engine::Events::Completed::install(const std::function<void()> &functor)
{
	TRACE_LOGGER;
	if (frontend)
		throw std::runtime_error("Frontend is already initialized");
	on_completed = functor;
}
void TRN4CPP::Engine::Events::Ack::install(const std::function<void(const unsigned long long &simulation_id, const std::size_t &counter, const bool &success, const std::string &cause)> &functor)
{
	TRACE_LOGGER;
	if (frontend)
		throw std::runtime_error("Frontend is already initialized");
	on_ack = functor;
}
void TRN4CPP::Engine::Events::Allocated::install(const std::function<void(const unsigned long long &simulation_id, const int &rank)> &functor)
{
	TRACE_LOGGER;
	if (frontend)
		throw std::runtime_error("Frontend is already initialized");
	on_allocated = functor;
}
void TRN4CPP::Engine::Events::Configured::install(const std::function<void(const unsigned long long &simulation_id)> &functor)
{
	TRACE_LOGGER;
	if (frontend)
		throw std::runtime_error("Frontend is already initialized");
	on_configured = functor;
}
void TRN4CPP::Engine::Events::Trained::install(const std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id)> &functor)
{
	TRACE_LOGGER;
	if (frontend)
		throw std::runtime_error("Frontend is already initialized");
	on_trained = functor;
}
void TRN4CPP::Engine::Events::Primed::install(const std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id)> &functor)
{
	TRACE_LOGGER;
	if (frontend)
		throw std::runtime_error("Frontend is already initialized");
	on_primed = functor;
}
void TRN4CPP::Engine::Events::Tested::install(const std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id)> &functor)
{
	TRACE_LOGGER;
	if (frontend)
		throw std::runtime_error("Frontend is already initialized");
	on_tested = functor;
}

void TRN4CPP::Engine::Events::Deallocated::install(const std::function<void(const unsigned long long &simulation_id, const int &rank)> &functor)
{
	TRACE_LOGGER;
	if (frontend)
		throw std::runtime_error("Frontend is already initialized");
	on_deallocated = functor;
}
void TRN4CPP::Engine::Events::Processor::install(const std::function<void(const int &rank, const std::string &host, const unsigned int &index, const std::string &name)> &functor)
{
	TRACE_LOGGER;
	if (frontend)
		throw std::runtime_error("Frontend is already initialized");
	on_processor = functor;
}

void   	TRN4CPP::Simulation::Encoder::Custom::configure(const unsigned long long &simulation_id, const std::size_t &batch_size, const std::size_t &stimulus_size,
	const std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &prediction, const std::size_t &rows, const std::size_t &cols)> &predicted_position,
	std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &stimulus, const std::size_t &rows, const std::size_t &cols)> &estimated_position,
	std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &stimulus, const std::size_t &rows, const std::size_t &cols)> &perceived_stimulus)
{
	TRACE_LOGGER;
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	estimated_position = std::bind(&TRN::Engine::Broker::notify_position, frontend, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5);
	perceived_stimulus = std::bind(&TRN::Engine::Broker::notify_stimulus, frontend, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5);
	frontend->install_position(simulation_id, predicted_position);
	frontend->configure_encoder_custom(simulation_id, batch_size, stimulus_size);
}

void TRN4CPP::Simulation::Recording::States::configure(const unsigned long long &simulation_id, const std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::string &phase, const std::string &label, const std::size_t &batch, const std::vector<float> &samples, const std::size_t &rows, const std::size_t &cols)> &functor, const bool &train, const bool &prime, const bool &generate)
{
	TRACE_LOGGER;
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");

	if (!train && !prime && !generate)
	{
		ERROR_LOGGER << "States decorator won't be installed because no experiment stage (train, prime, generate) is selected" ;
	}
	else
	{
		frontend->install_states(simulation_id, functor);
		frontend->setup_states(simulation_id, train, prime, generate);
	}
}
void TRN4CPP::Simulation::Recording::Weights::configure(const unsigned long long &simulation_id, const std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::string &phase, const std::string &label, const std::size_t &batch,  const std::vector<float> &weights, const std::size_t &rows, const std::size_t &cols)> &functor, const bool &initialize, const bool &train)
{
	TRACE_LOGGER;
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	frontend->install_weights(simulation_id, functor);
	frontend->setup_weights(simulation_id, initialize, train);
}
void TRN4CPP::Simulation::Recording::Performances::configure(const unsigned long long &simulation_id, const std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::string &phase, const float &cycles_per_second, const float &gflops_per_second)> &functor, const bool &train, const bool &prime, const bool &generate)
{
	TRACE_LOGGER;
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	frontend->install_performances(simulation_id, functor);
	frontend->setup_performances(simulation_id, train, prime, generate);
}
void TRN4CPP::Simulation::Recording::Scheduling::configure(const unsigned long long &simulation_id, const std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<int> &offsets, const std::vector<int> &durations)> &functor)
{
	TRACE_LOGGER;
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	frontend->install_scheduling(simulation_id, functor);
	frontend->setup_scheduling(simulation_id);
}
void TRN4CPP::Simulation::Measurement::Readout::MeanSquareError::configure(const unsigned long long &simulation_id, const std::size_t &batch_size, const std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &values, const std::size_t &rows, const std::size_t &cols)> &functor)
{
	TRACE_LOGGER;
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	frontend->install_measurement_readout_mean_square_error(simulation_id, functor);
	frontend->configure_measurement_readout_mean_square_error(simulation_id, batch_size);
}
void TRN4CPP::Simulation::Measurement::Readout::FrechetDistance::configure(const unsigned long long &simulation_id,  const std::size_t &batch_size, const std::string &norm, const std::string &aggregator, const std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &values, const std::size_t &rows, const std::size_t &cols)> &functor)
{
	TRACE_LOGGER;
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	frontend->install_measurement_readout_frechet_distance(simulation_id, functor);
	frontend->configure_measurement_readout_frechet_distance(simulation_id, batch_size, norm, aggregator);
}
void TRN4CPP::Simulation::Measurement::Readout::Custom::configure(const unsigned long long &simulation_id, const std::size_t &batch_size, const std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &primed, const std::vector<float> &predicted, const std::vector<float> &expected, const std::size_t &preamble, const std::size_t &pages, const std::size_t &rows, const std::size_t &cols)> &functor)
{
	TRACE_LOGGER;
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	frontend->install_measurement_readout_custom(simulation_id, functor);
	frontend->configure_measurement_readout_custom(simulation_id, batch_size);
}
void TRN4CPP::Simulation::Measurement::Position::MeanSquareError::configure(const unsigned long long &simulation_id, const std::size_t &batch_size, const std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &values, const std::size_t &rows, const std::size_t &cols)> &functor)
{
	TRACE_LOGGER;
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	frontend->install_measurement_position_mean_square_error(simulation_id, functor);
	frontend->configure_measurement_position_mean_square_error(simulation_id, batch_size);
}
void TRN4CPP::Simulation::Measurement::Position::FrechetDistance::configure(const unsigned long long &simulation_id, const std::size_t &batch_size, const std::string &norm, const std::string &aggregator, const std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &values, const std::size_t &rows, const std::size_t &cols)> &functor)
{
	TRACE_LOGGER;
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	frontend->install_measurement_position_frechet_distance(simulation_id, functor);
	frontend->configure_measurement_position_frechet_distance(simulation_id, batch_size, norm, aggregator);
}
void TRN4CPP::Simulation::Measurement::Position::Custom::configure(const unsigned long long &simulation_id, const std::size_t &batch_size, const std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &primed, const std::vector<float> &predicted, const std::vector<float> &expected, const std::size_t &preamble, const std::size_t &pages, const std::size_t &rows, const std::size_t &cols)> &functor)
{
	TRACE_LOGGER;
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	frontend->install_measurement_position_custom(simulation_id, functor);
	frontend->configure_measurement_position_custom(simulation_id, batch_size);
}


void TRN4CPP::Simulation::Loop::Custom::configure(const unsigned long long &simulation_id, const std::size_t &batch_size, const std::size_t &stimulus_size,
	const std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &prediction, const std::size_t &rows, const std::size_t &cols)> &request,
	std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &stimulus, const std::size_t &rows, const std::size_t &cols)> &reply
)
{
	TRACE_LOGGER;
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	reply = std::bind(&TRN::Engine::Broker::notify_stimulus, frontend, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5);
	frontend->install_stimulus(simulation_id, request);
	frontend->configure_loop_custom(simulation_id, batch_size, stimulus_size);
}
void TRN4CPP::Simulation::Scheduler::Custom::configure(const unsigned long long &simulation_id, const unsigned long &seed,
	const std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const unsigned long &seed, const std::vector<float> &elements, const std::size_t &rows, const std::size_t &cols, const std::vector<int> &offsets, const std::vector<int> &durations)> &request,
	std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<int> &offsets, const std::vector<int> &durations)> &reply, const std::string &tag)
{
	TRACE_LOGGER;
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	reply = std::bind(&TRN::Engine::Broker::notify_scheduler, frontend, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4);
	frontend->install_scheduler(simulation_id, request);
	frontend->configure_scheduler_custom(simulation_id, seed, tag);
}
void TRN4CPP::Simulation::Scheduler::Mutator::Custom::configure(const unsigned long long &simulation_id, const unsigned long &seed,
	const std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const unsigned long &seed, const std::vector<int> &offsets, const std::vector<int> &durations)> &request,
	std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<int> &offsets, const std::vector<int> &durations)> &reply
)
{
	TRACE_LOGGER;
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	reply = std::bind(&TRN::Engine::Broker::notify_mutator, frontend, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4);
	frontend->install_mutator(simulation_id, request);
	frontend->configure_mutator_custom(simulation_id, seed);
}
void TRN4CPP::Simulation::Reservoir::Weights::Readout::Custom::configure(const unsigned long long &simulation_id, const std::function<void(const unsigned long long &simulation_id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &request,
	std::function<void(const unsigned long long &simulation_id, const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &reply)
{
	TRACE_LOGGER;
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	reply = std::bind(&TRN::Engine::Broker::notify_readout, frontend, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5);
	frontend->install_readout(simulation_id, request);
	frontend->configure_readout_custom(simulation_id);
}

void TRN4CPP::Simulation::Reservoir::Weights::Recurrent::Custom::configure(const unsigned long long &simulation_id, const std::function<void(const unsigned long long &simulation_id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &request,
	std::function<void(const unsigned long long &simulation_id, const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &reply)
{
	TRACE_LOGGER;
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	reply = std::bind(&TRN::Engine::Broker::notify_recurrent, frontend, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5);
	frontend->install_recurrent(simulation_id, request);
	frontend->configure_recurrent_custom(simulation_id);
}
void TRN4CPP::Simulation::Reservoir::Weights::Feedforward::Custom::configure(const unsigned long long &simulation_id, const std::function<void(const unsigned long long &simulation_id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &request,
	std::function<void(const unsigned long long &simulation_id, const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &reply)
{
	TRACE_LOGGER;
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	reply = std::bind(&TRN::Engine::Broker::notify_feedforward, frontend, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5);
	frontend->install_feedforward(simulation_id, request);
	frontend->configure_feedforward_custom(simulation_id);
}