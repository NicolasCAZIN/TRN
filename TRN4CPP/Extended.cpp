#include "stdafx.h"
#include "Extended.h"

#include "ViewModel/Communicator.h"
#include "ViewModel/Frontend.h"
#include "ViewModel/Executor.h"





extern std::shared_ptr<TRN::Engine::Frontend> frontend;
extern std::shared_ptr<TRN::Engine::Executor> executor;

void TRN4CPP::Engine::Execution::run()
{
	if (!executor)
	{
		throw std::runtime_error("Executor is not setup");
	}
	executor->run();
}
void TRN4CPP::Engine::Execution::run(const std::size_t &count)
{
	if (!executor)
	{
		throw std::runtime_error("Executor is not setup");
	}
	for (std::size_t k = 0; k < count; k++)
		executor->run_one();
}
void TRN4CPP::Engine::Events::Completed::install(const std::function<void()> &functor)
{
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	frontend->install_completed(functor);
}
void TRN4CPP::Engine::Events::Ack::install(const std::function<void(const unsigned int &id, const std::size_t &number, const bool &success, const std::string &cause)> &functor)
{
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	frontend->install_ack(functor);
}
void TRN4CPP::Engine::Events::Allocated::install(const std::function<void(const unsigned int &id, const int &rank)> &functor)
{
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	frontend->install_allocated(functor);
}
void TRN4CPP::Engine::Events::Configured::install(const std::function<void(const unsigned int &id)> &functor)
{
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	frontend->install_configured(functor);
}
void TRN4CPP::Engine::Events::Trained::install(const std::function<void(const unsigned int &id)> &functor)
{
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	frontend->install_trained(functor);
}
void TRN4CPP::Engine::Events::Primed::install(const std::function<void(const unsigned int &id)> &functor)
{
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	frontend->install_primed(functor);
}
void TRN4CPP::Engine::Events::Tested::install(const std::function<void(const unsigned int &id)> &functor)
{
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	frontend->install_tested(functor);
}

void TRN4CPP::Engine::Events::Deallocated::install(const std::function<void(const unsigned int &id, const int &rank)> &functor)
{
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	frontend->install_deallocated(functor);
}
void TRN4CPP::Engine::Events::Processor::install(const std::function<void(const int &rank, const std::string &host, const unsigned int &index, const std::string &name)> &functor)
{
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	frontend->install_processor(functor);
}
void TRN4CPP::Simulation::allocate(const unsigned int &id)
{
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");

	frontend->allocate(id);
}
void TRN4CPP::Simulation::deallocate(const unsigned int &id)
{
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	frontend->deallocate(id);
}
void TRN4CPP::Simulation::train(const unsigned int &id, const std::string &label, const std::string &incoming, const std::string &expected)
{
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	frontend->train(id, label, incoming, expected);
}
void TRN4CPP::Simulation::test(const unsigned int &id, const std::string &sequence, const std::string &incoming, const std::string &expected, const unsigned int &preamble, const unsigned int &supplementary_generations)
{
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	frontend->test(id, sequence, incoming, expected, preamble, supplementary_generations);
}
void TRN4CPP::Simulation::declare_sequence(const unsigned int &id, const std::string &label, const std::string &tag, const std::vector<float> &sequence, const std::size_t &observations)
{
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	frontend->declare_sequence(id, label, tag, sequence, observations);
}
void TRN4CPP::Simulation::declare_set(const unsigned int &id, const std::string &label, const std::string &tag, const std::vector<std::string> &labels)
{
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	frontend->declare_set(id, label, tag, labels);
}
void TRN4CPP::Simulation::configure_begin(const unsigned int &id)
{
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	frontend->configure_begin(id);
}
void TRN4CPP::Simulation::configure_end(const unsigned int &id)
{
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	frontend->configure_end(id);
}
void TRN4CPP::Simulation::Recording::States::configure(const unsigned int &id, const std::function<void(const unsigned int &id, const std::string &phase, const std::string &label, const std::size_t &batch, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &samples, const std::size_t &rows, const std::size_t &cols)> &functor, const bool &train, const bool &prime, const bool &generate)
{
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	frontend->install_states(id, functor);
	frontend->setup_states(id, train, prime, generate);
}
void TRN4CPP::Simulation::Recording::Weights::configure(const unsigned int &id, const std::function<void(const unsigned int &id, const std::string &phase, const std::string &label, const std::size_t &batch, const std::size_t &trial, const std::vector<float> &weights, const std::size_t &rows, const std::size_t &cols)> &functor, const bool &initialize, const bool &train)
{
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	frontend->install_weights(id, functor);
	frontend->setup_weights(id, initialize, train);
}
void TRN4CPP::Simulation::Recording::Performances::configure(const unsigned int &id, const std::function<void(const unsigned int &id, const std::string &phase, const std::size_t &batch_size, const size_t &cycles, const float &gflops, const float &seconds)> &functor, const bool &train, const bool &prime, const bool &generate)
{
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	frontend->install_performances(id, functor);
	frontend->setup_performances(id, train, prime, generate);
}
void TRN4CPP::Simulation::Recording::Scheduling::configure(const unsigned int &id, const std::function<void(const unsigned int &id, const std::size_t &trial, const std::vector<int> &offsets, const std::vector<int> &durations)> &functor)
{
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	frontend->install_scheduling(id, functor);
	frontend->setup_scheduling(id);
}
void TRN4CPP::Simulation::Measurement::Readout::MeanSquareError::configure(const unsigned int &id, const std::size_t &batch_size, const std::function<void(const unsigned int &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const std::size_t &cols)> &functor)
{
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	frontend->install_measurement_readout_mean_square_error(id, functor);
	frontend->configure_measurement_readout_mean_square_error(id, batch_size);
}
void TRN4CPP::Simulation::Measurement::Readout::FrechetDistance::configure(const unsigned int &id, const std::size_t &batch_size, const std::function<void(const unsigned int &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const std::size_t &cols)> &functor)
{
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	frontend->install_measurement_readout_frechet_distance(id, functor);
	frontend->configure_measurement_readout_frechet_distance(id, batch_size);
}
void TRN4CPP::Simulation::Measurement::Readout::Custom::configure(const unsigned int &id, const std::size_t &batch_size, const std::function<void(const unsigned int &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &primed, const std::vector<float> &predicted, const std::vector<float> &expected, const std::size_t &preamble, const std::size_t &pages, const std::size_t &rows, const std::size_t &cols)> &functor)
{
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	frontend->install_measurement_readout_custom(id, functor);
	frontend->configure_measurement_readout_custom(id, batch_size);
}
void TRN4CPP::Simulation::Measurement::Position::MeanSquareError::configure(const unsigned int &id, const std::size_t &batch_size, const std::function<void(const unsigned int &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const std::size_t &cols)> &functor)
{
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	frontend->install_measurement_position_mean_square_error(id, functor);
	frontend->configure_measurement_position_mean_square_error(id, batch_size);
}
void TRN4CPP::Simulation::Measurement::Position::FrechetDistance::configure(const unsigned int &id, const std::size_t &batch_size, const std::function<void(const unsigned int &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const std::size_t &cols)> &functor)
{
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	frontend->install_measurement_position_frechet_distance(id, functor);
	frontend->configure_measurement_position_frechet_distance(id, batch_size);
}
void TRN4CPP::Simulation::Measurement::Position::Custom::configure(const unsigned int &id, const std::size_t &batch_size, const std::function<void(const unsigned int &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &primed, const std::vector<float> &predicted, const std::vector<float> &expected, const std::size_t &preamble, const std::size_t &pages, const std::size_t &rows, const std::size_t &cols)> &functor)
{
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	frontend->install_measurement_position_custom(id, functor);
	frontend->configure_measurement_position_custom(id, batch_size);
}
void TRN4CPP::Simulation::Reservoir::WidrowHoff::configure(const unsigned int &id, const std::size_t &stimulus_size, const std::size_t &prediction_size, const std::size_t &reservoir_size, const float &leak_rate, const float &initial_state_scale, const float &learning_rate, const unsigned long &seed, const std::size_t &batch_size)
{
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	frontend->configure_reservoir_widrow_hoff(id, stimulus_size, prediction_size, reservoir_size, leak_rate, initial_state_scale, learning_rate, seed, batch_size);
}
void TRN4CPP::Simulation::Loop::Copy::configure(const unsigned int &id, const std::size_t &batch_size, const std::size_t &stimulus_size)
{
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	frontend->configure_loop_copy(id, batch_size, stimulus_size);
}
void TRN4CPP::Simulation::Loop::SpatialFilter::configure(const unsigned int &id, const std::size_t &batch_size, const std::size_t &stimulus_size, const unsigned long &seed,
	const std::function<void(const unsigned int &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols)> &predicted_position,
	std::function<void(const unsigned int &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols)> &estimated_position,
	const std::function<void(const unsigned int &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols)> &predicted_stimulus,
	std::function<void(const unsigned int &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &stimulus, const std::size_t &rows, const std::size_t &cols)> &perceived_stimulus,
	const std::size_t &rows, const std::size_t &cols, const std::pair<float, float> &x, const std::pair<float, float> &y,
	const std::vector<float> response, const float &sigma, const float &radius, const float &scale, const std::string &tag
)
{
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	estimated_position = std::bind(&TRN::Engine::Broker::notify_position, frontend, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5, std::placeholders::_6);
	perceived_stimulus = std::bind(&TRN::Engine::Broker::notify_stimulus, frontend, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5, std::placeholders::_6);
	frontend->install_position(id, predicted_position);
	frontend->install_stimulus(id, predicted_stimulus);
	frontend->configure_loop_spatial_filter(id, batch_size, stimulus_size, seed,
		rows, cols, x, y, response, sigma, radius, scale, tag);
}
void TRN4CPP::Simulation::Loop::Custom::configure(const unsigned int &id, const std::size_t &batch_size, const std::size_t &stimulus_size,
	const std::function<void(const unsigned int &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &prediction, const std::size_t &rows, const std::size_t &cols)> &request,
	std::function<void(const unsigned int &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &stimulus, const std::size_t &rows, const std::size_t &cols)> &reply
)
{
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	reply = std::bind(&TRN::Engine::Broker::notify_stimulus, frontend, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5, std::placeholders::_6);
	frontend->install_stimulus(id, request);
	frontend->configure_loop_custom(id, batch_size, stimulus_size);
}
void TRN4CPP::Simulation::Scheduler::Tiled::configure(const unsigned int &id, const unsigned int &epochs)
{
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	frontend->configure_scheduler_tiled(id, epochs);
}
void TRN4CPP::Simulation::Scheduler::Snippets::configure(const unsigned int &id, const unsigned long &seed, const unsigned int &snippets_size, const unsigned int &time_budget, const std::string &tag)
{
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	frontend->configure_scheduler_snippets(id, seed, snippets_size, time_budget, tag);
}
void TRN4CPP::Simulation::Scheduler::Custom::configure(const unsigned int &id, const unsigned long &seed,
	const std::function<void(const unsigned int &id, const unsigned long &seed, const std::size_t &trial, const std::vector<float> &elements, const std::size_t &rows, const std::size_t &cols, const std::vector<int> &offsets, const std::vector<int> &durations)> &request,
	std::function<void(const unsigned int &id, const std::size_t &trial, const std::vector<int> &offsets, const std::vector<int> &durations)> &reply, const std::string &tag)
{
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	reply = std::bind(&TRN::Engine::Broker::notify_scheduler, frontend, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4);
	frontend->install_scheduler(id, request);
	frontend->configure_scheduler_custom(id, seed, tag);
}
void TRN4CPP::Simulation::Scheduler::Mutator::Shuffle::configure(const unsigned int &id, const unsigned long &seed)
{
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	frontend->configure_mutator_shuffle(id, seed);
}
void TRN4CPP::Simulation::Scheduler::Mutator::Reverse::configure(const unsigned int &id, const unsigned long &seed, const float &rate, const std::size_t &size)
{
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	frontend->configure_mutator_reverse(id, seed, rate, size);
}
void TRN4CPP::Simulation::Scheduler::Mutator::Punch::configure(const unsigned int &id, const unsigned long &seed, const float &rate, const std::size_t &size, const std::size_t &number)
{
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	frontend->configure_mutator_punch(id, seed, rate, size, number);
}
void TRN4CPP::Simulation::Scheduler::Mutator::Custom::configure(const unsigned int &id, const unsigned long &seed,
	const std::function<void(const unsigned int &id, const unsigned long &seed, const std::size_t &trial, const std::vector<int> &offsets, const std::vector<int> &durations)> &request,
	std::function<void(const unsigned int &id, const std::size_t &trial, const std::vector<int> &offsets, const std::vector<int> &durations)> &reply
)
{
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	reply = std::bind(&TRN::Engine::Broker::notify_mutator, frontend, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4);
	frontend->install_mutator(id, request);
	frontend->configure_mutator_custom(id, seed);
}
void TRN4CPP::Simulation::Reservoir::Weights::Readout::Uniform::configure(const unsigned int &id, const float &a, const float &b, const float &sparsity)
{
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	frontend->configure_readout_uniform(id, a, b, sparsity);
}
void TRN4CPP::Simulation::Reservoir::Weights::Readout::Gaussian::configure(const unsigned int &id, const float &mu, const float &sigma)
{
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	frontend->configure_readout_gaussian(id, mu, sigma);
}
void TRN4CPP::Simulation::Reservoir::Weights::Readout::Custom::configure(const unsigned int &id, const std::function<void(const unsigned int &id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &request,
	std::function<void(const unsigned int &id, const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &reply)
{
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	reply = std::bind(&TRN::Engine::Broker::notify_readout, frontend, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5);
	frontend->install_readout(id, request);
	frontend->configure_readout_custom(id);
}
void TRN4CPP::Simulation::Reservoir::Weights::Feedback::Uniform::configure(const unsigned int &id, const float &a, const float &b, const float &sparsity)
{
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	frontend->configure_feedback_uniform(id, a, b, sparsity);
}
void TRN4CPP::Simulation::Reservoir::Weights::Feedback::Gaussian::configure(const unsigned int &id, const float &mu, const float &sigma)
{
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	frontend->configure_feedback_gaussian(id, mu, sigma);
}
void TRN4CPP::Simulation::Reservoir::Weights::Feedback::Custom::configure(const unsigned int &id, const std::function<void(const unsigned int &id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &request,
	std::function<void(const unsigned int &id, const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &reply)
{
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	reply = std::bind(&TRN::Engine::Broker::notify_feedback, frontend, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5);
	frontend->install_feedback(id, request);
	frontend->configure_feedback_custom(id);
}
void TRN4CPP::Simulation::Reservoir::Weights::Recurrent::Uniform::configure(const unsigned int &id, const float &a, const float &b, const float &sparsity)
{
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	frontend->configure_recurrent_uniform(id, a, b, sparsity);
}
void TRN4CPP::Simulation::Reservoir::Weights::Recurrent::Gaussian::configure(const unsigned int &id, const float &mu, const float &sigma)
{
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	frontend->configure_recurrent_gaussian(id, mu, sigma);
}
void TRN4CPP::Simulation::Reservoir::Weights::Recurrent::Custom::configure(const unsigned int &id, const std::function<void(const unsigned int &id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &request,
	std::function<void(const unsigned int &id, const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &reply)
{
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	reply = std::bind(&TRN::Engine::Broker::notify_recurrent, frontend, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5);
	frontend->install_recurrent(id, request);
	frontend->configure_recurrent_custom(id);
}
void TRN4CPP::Simulation::Reservoir::Weights::Feedforward::Uniform::configure(const unsigned int &id, const float &a, const float &b, const float &sparsity)
{
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	frontend->configure_feedforward_uniform(id, a, b, sparsity);
}
void TRN4CPP::Simulation::Reservoir::Weights::Feedforward::Gaussian::configure(const unsigned int &id, const float &mu, const float &sigma)
{
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	frontend->configure_feedforward_gaussian(id, mu, sigma);
}
void TRN4CPP::Simulation::Reservoir::Weights::Feedforward::Custom::configure(const unsigned int &id, const std::function<void(const unsigned int &id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &request,
	std::function<void(const unsigned int &id, const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &reply)
{
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	reply = std::bind(&TRN::Engine::Broker::notify_feedforward, frontend, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5);
	frontend->install_feedforward(id, request);
	frontend->configure_feedforward_custom(id);
}