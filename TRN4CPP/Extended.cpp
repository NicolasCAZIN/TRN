#include "stdafx.h"
#include "Extended.h"
#include "Advanced.h"
#include "Custom.h"

#include "ViewModel/Communicator.h"
#include "ViewModel/Frontend.h"
#include "Helper/Logger.h"


const bool TRN4CPP::Simulation::DEFAULT_RESET_READOUT = false;

extern std::shared_ptr<TRN::Engine::Frontend> frontend;

extern std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &primed, const std::vector<float> &predicted, const std::vector<float> &expected, const std::size_t &preamble, const std::size_t &pages, const std::size_t &rows, const  std::size_t &cols)> on_measurement_readout_raw;
extern std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &primed, const std::vector<float> &predicted, const std::vector<float> &expected, const std::size_t &preamble, const std::size_t &pages, const std::size_t &rows, const  std::size_t &cols)> on_measurement_position_raw;
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
extern std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const unsigned long &seed,  const std::vector<int> &offsets, const std::vector<int> &durations)> on_mutator;
extern std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const unsigned long &seed,  const std::vector<float> &elements, const std::size_t &rows, const std::size_t &cols, const std::vector<int> &offsets, const std::vector<int> &durations)> on_scheduler;
extern std::function<void(const unsigned long long &simulation_id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const  std::size_t &cols)> on_feedforward;
extern std::function<void(const unsigned long long &simulation_id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const  std::size_t &cols)> on_feedback;
extern std::function<void(const unsigned long long &simulation_id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const  std::size_t &cols)> on_readout;
extern std::function<void(const unsigned long long &simulation_id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> on_recurrent;
extern std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols)> reply_position;
extern std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols)> reply_stimulus;
extern std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<int> &offsets, const std::vector<int> &durations)> reply_scheduler;
extern std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<int> &offsets, const std::vector<int> &durations)> reply_mutator;
extern std::function<void(const unsigned long long &simulation_id, const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> reply_feedforward;
extern std::function<void(const unsigned long long &simulation_id, const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> reply_feedback;
extern std::function<void(const unsigned long long &simulation_id, const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> reply_recurrent;
extern std::function<void(const unsigned long long &simulation_id, const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> reply_readout;

template<typename T, typename... U>
size_t get_address(const std::function<T(U...)> &f) {
	typedef T(function_type)(U...);
	function_type  *const*function_pointer = f.template target<function_type*>();
	if (function_pointer == 0)
		return 0;
	else
		return reinterpret_cast<size_t>(const_cast<function_type  *>(*function_pointer));
}

template<typename T, typename... U>
bool operator == (const std::function<T(U...)> &f1, const std::function<T(U...)> &f2)
{
	return get_address(f1) == get_address(f2);
}

template<typename T, typename... U>
bool operator != (const std::function<T(U...)> &f1, const std::function<T(U...)> &f2)
{
	return !(f1 == f2);
}


void TRN4CPP::Engine::Execution::run()
{
	TRACE_LOGGER;
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	frontend->dispose();
}


void TRN4CPP::Simulation::allocate(const unsigned long long &simulation_id)
{
	TRACE_LOGGER;
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");

	frontend->allocate(simulation_id);
}
void TRN4CPP::Simulation::deallocate(const unsigned long long &simulation_id)
{
	TRACE_LOGGER;
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	frontend->deallocate(simulation_id);
}
void TRN4CPP::Simulation::train(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::string &label, const std::string &incoming, const std::string &expected, const bool &reset_readout)
{
	TRACE_LOGGER;
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	frontend->train(simulation_id, evaluation_id,label, incoming, expected, reset_readout);
}
void TRN4CPP::Simulation::test(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::string &sequence, const std::string &incoming, const std::string &expected, const unsigned int &preamble, const bool &autonomous, const unsigned int &supplementary_generations)
{
	TRACE_LOGGER;
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	frontend->test(simulation_id, evaluation_id,sequence, incoming, expected, preamble, autonomous, supplementary_generations);
}
void TRN4CPP::Simulation::declare_sequence(const unsigned long long &simulation_id, const std::string &label, const std::string &tag, const std::vector<float> &sequence, const std::size_t &observations)
{
	TRACE_LOGGER;
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	frontend->declare_sequence(simulation_id,label, tag, sequence, observations);
}
void TRN4CPP::Simulation::declare_set(const unsigned long long &simulation_id, const std::string &label, const std::string &tag, const std::vector<std::string> &labels)
{
	TRACE_LOGGER;
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	frontend->declare_set(simulation_id,label, tag, labels);
}
void TRN4CPP::Simulation::configure_begin(const unsigned long long &simulation_id)
{
	TRACE_LOGGER;
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	frontend->configure_begin(simulation_id);
}
void TRN4CPP::Simulation::configure_end(const unsigned long long &simulation_id)
{
	TRACE_LOGGER;
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	frontend->configure_end(simulation_id);
}
void TRN4CPP::Simulation::Recording::States::configure(const unsigned long long &simulation_id, const bool &train, const bool &prime, const bool &generate)
{
	TRACE_LOGGER;
	if (!on_states)
		throw std::runtime_error("States callback is not installed");
	TRN4CPP::Simulation::Recording::States::configure(simulation_id,on_states, train, prime, generate);
}
void TRN4CPP::Simulation::Recording::Weights::configure(const unsigned long long &simulation_id, const bool &initialize, const bool &train)
{
	TRACE_LOGGER;
	if (!on_weights)
		throw std::runtime_error("States callback is not installed");
	TRN4CPP::Simulation::Recording::Weights::configure(simulation_id,on_weights, initialize, train);
}
void TRN4CPP::Simulation::Recording::Performances::configure(const unsigned long long &simulation_id, const bool &train, const bool &prime, const bool &generate)
{
	TRACE_LOGGER;
	if (!on_states)
		throw std::runtime_error("States callback is not installed");
	TRN4CPP::Simulation::Recording::Performances::configure(simulation_id,on_performances, train, prime, generate);
}
void TRN4CPP::Simulation::Recording::Scheduling::configure(const unsigned long long &simulation_id)
{
	TRACE_LOGGER;
	if (!on_states)
		throw std::runtime_error("States callback is not installed");
	TRN4CPP::Simulation::Recording::Scheduling::configure(simulation_id,on_scheduling);
}
void TRN4CPP::Simulation::Measurement::Readout::MeanSquareError::configure(const unsigned long long &simulation_id, const std::size_t &batch_size)
{
	TRACE_LOGGER;
	if (!on_measurement_readout_mean_square_error)
		throw std::runtime_error("Measurement readout mean square error callback is not installed");
	TRN4CPP::Simulation::Measurement::Readout::MeanSquareError::configure(simulation_id,batch_size, on_measurement_readout_mean_square_error);
}
void TRN4CPP::Simulation::Measurement::Readout::FrechetDistance::configure(const unsigned long long &simulation_id, const std::size_t &batch_size)
{
	TRACE_LOGGER;
	if (!on_measurement_readout_frechet_distance)
		throw std::runtime_error("Measurement readout Frechet distance callback is not installed");
	TRN4CPP::Simulation::Measurement::Readout::FrechetDistance::configure(simulation_id,batch_size, on_measurement_readout_frechet_distance);
}
void TRN4CPP::Simulation::Measurement::Readout::Custom::configure(const unsigned long long &simulation_id, const std::size_t &batch_size)
{
	TRACE_LOGGER;
	if (!on_measurement_readout_raw)
		throw std::runtime_error("Measurement readout raw callback is not installed");
	TRN4CPP::Simulation::Measurement::Readout::Custom::configure(simulation_id,batch_size, on_measurement_readout_raw);
}
void TRN4CPP::Simulation::Measurement::Position::MeanSquareError::configure(const unsigned long long &simulation_id, const std::size_t &batch_size)
{
	TRACE_LOGGER;
	if (!on_measurement_position_mean_square_error)
		throw std::runtime_error("Measurement position mean square error callback is not installed");
	TRN4CPP::Simulation::Measurement::Position::MeanSquareError::configure(simulation_id,batch_size, on_measurement_position_mean_square_error);
}
void TRN4CPP::Simulation::Measurement::Position::FrechetDistance::configure(const unsigned long long &simulation_id, const std::size_t &batch_size)
{
	TRACE_LOGGER;
	if (!on_measurement_position_frechet_distance)
		throw std::runtime_error("Measurement position Frechet distance callback is not installed");
	TRN4CPP::Simulation::Measurement::Position::FrechetDistance::configure(simulation_id,batch_size, on_measurement_position_frechet_distance);
}
void TRN4CPP::Simulation::Measurement::Position::Custom::configure(const unsigned long long &simulation_id, const std::size_t &batch_size)
{
	TRACE_LOGGER;
	if (!on_measurement_position_raw)
		throw std::runtime_error("Measurement position raw callback is not installed");
	TRN4CPP::Simulation::Measurement::Position::Custom::configure(simulation_id,batch_size, on_measurement_position_raw);
}
void TRN4CPP::Simulation::Reservoir::WidrowHoff::configure(const unsigned long long &simulation_id, const std::size_t &stimulus_size, const std::size_t &prediction_size, const std::size_t &reservoir_size, const float &leak_rate, const float &initial_state_scale, const float &learning_rate, const unsigned long &seed, const std::size_t &batch_size)
{
	TRACE_LOGGER;
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	frontend->configure_reservoir_widrow_hoff(simulation_id,stimulus_size, prediction_size, reservoir_size, leak_rate, initial_state_scale, learning_rate, seed, batch_size);
}
void TRN4CPP::Simulation::Loop::Copy::configure(const unsigned long long &simulation_id, const std::size_t &batch_size, const std::size_t &stimulus_size)
{
	TRACE_LOGGER;
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	frontend->configure_loop_copy(simulation_id,batch_size, stimulus_size);
}

void TRN4CPP::Simulation::Loop::SpatialFilter::configure(const unsigned long long &simulation_id, const std::size_t &batch_size, const std::size_t &stimulus_size, const unsigned long &seed,
	const std::size_t &rows, const std::size_t &cols, const std::pair<float, float> &x, const std::pair<float, float> &y,
	const std::vector<float> response, const float &sigma, const float &radius, const float &angle, const float &scale, const std::string &tag
)
{
	TRACE_LOGGER;
	if (!on_position)
		throw std::runtime_error("Position callback is not installed");
	if (!reply_position)
		throw std::runtime_error("Position reply functor is not installed");
	if (!on_stimulus)
		throw std::runtime_error("Stimulus callback is not installed");
	if (!reply_stimulus)
		throw std::runtime_error("Stimulus reply functor is not installed");
	auto old_reply_position = reply_position;
	auto old_reply_stimulus = reply_stimulus;
	TRN4CPP::Simulation::Loop::SpatialFilter::configure(simulation_id,batch_size, stimulus_size, seed, on_position, reply_position, on_stimulus, reply_stimulus, rows, cols, x, y, response, sigma, radius, angle, scale, tag);
	if (old_reply_position != reply_position)
		throw std::runtime_error("Position reply functor changed");
	if (old_reply_stimulus != reply_stimulus)
		throw std::runtime_error("Stimulus reply functor changed");
}
void TRN4CPP::Simulation::Loop::Custom::configure(const unsigned long long &simulation_id, const std::size_t &batch_size, const std::size_t &stimulus_size)
{
	TRACE_LOGGER;
	if (!on_stimulus)
		throw std::runtime_error("Stimulus callback is not installed");
	if (!reply_stimulus)
		throw std::runtime_error("Stimulus reply functor is not installed");
	auto old_reply_stimulus = reply_stimulus;
	TRN4CPP::Simulation::Loop::Custom::configure(simulation_id,batch_size, stimulus_size, on_stimulus, reply_stimulus);
	if (old_reply_stimulus != reply_stimulus)
		throw std::runtime_error("Stimulus reply functor changed");
}
void TRN4CPP::Simulation::Scheduler::Tiled::configure(const unsigned long long &simulation_id, const unsigned int &epochs)
{
	TRACE_LOGGER;
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	frontend->configure_scheduler_tiled(simulation_id,epochs);
}
void TRN4CPP::Simulation::Scheduler::Snippets::configure(const unsigned long long &simulation_id, const unsigned long &seed, const unsigned int &snippets_size, const unsigned int &time_budget, const std::string &tag)
{
	TRACE_LOGGER;
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	frontend->configure_scheduler_snippets(simulation_id,seed, snippets_size, time_budget, tag);
}
void TRN4CPP::Simulation::Scheduler::Custom::configure(const unsigned long long &simulation_id, const unsigned long &seed,const std::string &tag)
{
	TRACE_LOGGER;
	if (!on_scheduler)
		throw std::runtime_error("Scheduler callback is not installed");
	if (!reply_scheduler)
		throw std::runtime_error("Scheduler reply functor is not installed");
	auto old_scheduler = reply_scheduler;
	TRN4CPP::Simulation::Scheduler::Custom::configure(simulation_id,seed, on_scheduler, reply_scheduler, tag);
	if (old_scheduler != reply_scheduler)
		throw std::runtime_error("Scheduler reply functor changed");
}
void TRN4CPP::Simulation::Scheduler::Mutator::Shuffle::configure(const unsigned long long &simulation_id, const unsigned long &seed)
{
	TRACE_LOGGER;
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	frontend->configure_mutator_shuffle(simulation_id,seed);
}
void TRN4CPP::Simulation::Scheduler::Mutator::Reverse::configure(const unsigned long long &simulation_id, const unsigned long &seed, const float &rate, const std::size_t &size)
{
	TRACE_LOGGER;
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	frontend->configure_mutator_reverse(simulation_id,seed, rate, size);
}
void TRN4CPP::Simulation::Scheduler::Mutator::Punch::configure(const unsigned long long &simulation_id, const unsigned long &seed, const float &rate, const std::size_t &size, const std::size_t &counter)
{
	TRACE_LOGGER;
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	frontend->configure_mutator_punch(simulation_id,seed, rate, size, counter);
}
void TRN4CPP::Simulation::Scheduler::Mutator::Custom::configure(const unsigned long long &simulation_id, const unsigned long &seed)
{
	TRACE_LOGGER;
	if (!on_mutator)
		throw std::runtime_error("Mutator callback is not installed");
	if (!reply_mutator)
		throw std::runtime_error("Mutator reply functor is not installed");
	auto old_mutator = reply_mutator;
	TRN4CPP::Simulation::Scheduler::Mutator::Custom::configure(simulation_id,seed, on_mutator, reply_mutator);
	if (old_mutator != reply_mutator)
		throw std::runtime_error("Mutator reply functor changed");
}
void TRN4CPP::Simulation::Reservoir::Weights::Readout::Uniform::configure(const unsigned long long &simulation_id, const float &a, const float &b, const float &sparsity)
{
	TRACE_LOGGER;
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	frontend->configure_readout_uniform(simulation_id,a, b, sparsity);
}
void TRN4CPP::Simulation::Reservoir::Weights::Readout::Gaussian::configure(const unsigned long long &simulation_id, const float &mu, const float &sigma, const float &sparsity)
{
	TRACE_LOGGER;
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	frontend->configure_readout_gaussian(simulation_id,mu, sigma, sparsity);
}
void TRN4CPP::Simulation::Reservoir::Weights::Readout::Custom::configure(const unsigned long long &simulation_id)
{
	TRACE_LOGGER;
	if (!on_readout)
		throw std::runtime_error("Readout callback is not installed");
	if (!reply_readout)
		throw std::runtime_error("Readout reply functor is not installed");
	auto old_readout = reply_readout;
	TRN4CPP::Simulation::Reservoir::Weights::Readout::Custom::configure(simulation_id,on_readout, reply_readout);
	if (old_readout != reply_readout)
		throw std::runtime_error("Readout reply functor changed");
}
void TRN4CPP::Simulation::Reservoir::Weights::Feedback::Uniform::configure(const unsigned long long &simulation_id, const float &a, const float &b, const float &sparsity)
{
	TRACE_LOGGER;
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	frontend->configure_feedback_uniform(simulation_id,a, b, sparsity);
}
void TRN4CPP::Simulation::Reservoir::Weights::Feedback::Gaussian::configure(const unsigned long long &simulation_id, const float &mu, const float &sigma, const float &sparsity)
{
	TRACE_LOGGER;
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	frontend->configure_feedback_gaussian(simulation_id,mu, sigma, sparsity);
}
void TRN4CPP::Simulation::Reservoir::Weights::Feedback::Custom::configure(const unsigned long long &simulation_id)
{
	TRACE_LOGGER;
	if (!on_feedback)
		throw std::runtime_error("Feedback callback is not installed");
	if (!reply_feedback)
		throw std::runtime_error("Feedback reply functor is not installed");
	auto old_feedback = reply_feedback;
	TRN4CPP::Simulation::Reservoir::Weights::Feedback::Custom::configure(simulation_id,on_feedback, reply_feedback);
	if (old_feedback != reply_feedback)
		throw std::runtime_error("Feedback reply functor changed");
}
void TRN4CPP::Simulation::Reservoir::Weights::Recurrent::Uniform::configure(const unsigned long long &simulation_id, const float &a, const float &b, const float &sparsity)
{
	TRACE_LOGGER;
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	frontend->configure_recurrent_uniform(simulation_id,a, b, sparsity);
}
void TRN4CPP::Simulation::Reservoir::Weights::Recurrent::Gaussian::configure(const unsigned long long &simulation_id, const float &mu, const float &sigma, const float &sparsity)
{
	TRACE_LOGGER;
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	frontend->configure_recurrent_gaussian(simulation_id,mu, sigma, sparsity);
}
void TRN4CPP::Simulation::Reservoir::Weights::Recurrent::Custom::configure(const unsigned long long &simulation_id)
{
	TRACE_LOGGER;
	if (!on_recurrent)
		throw std::runtime_error("Recurrent callback is not installed");
	if (!reply_recurrent)
		throw std::runtime_error("Recurrent reply functor is not installed");
	auto old_recurrent = reply_recurrent;
	TRN4CPP::Simulation::Reservoir::Weights::Recurrent::Custom::configure(simulation_id,on_recurrent, reply_recurrent);
	if (old_recurrent != reply_recurrent)
		throw std::runtime_error("Recurrent reply functor changed");
}
void TRN4CPP::Simulation::Reservoir::Weights::Feedforward::Uniform::configure(const unsigned long long &simulation_id, const float &a, const float &b, const float &sparsity)
{
	TRACE_LOGGER;
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	frontend->configure_feedforward_uniform(simulation_id,a, b, sparsity);
}
void TRN4CPP::Simulation::Reservoir::Weights::Feedforward::Gaussian::configure(const unsigned long long &simulation_id, const float &mu, const float &sigma, const float &sparsity)
{
	TRACE_LOGGER;
	if (!frontend)
		throw std::runtime_error("Frontend is not initialized");
	frontend->configure_feedforward_gaussian(simulation_id,mu, sigma, sparsity);
}
void TRN4CPP::Simulation::Reservoir::Weights::Feedforward::Custom::configure(const unsigned long long &simulation_id)
{
	TRACE_LOGGER;
	if (!on_feedforward)
		throw std::runtime_error("Feedforward callback is not installed");
	if (!reply_feedforward)
		throw std::runtime_error("Feedforward reply functor is not installed");
	auto old_feedforward = reply_feedforward;
	TRN4CPP::Simulation::Reservoir::Weights::Feedforward::Custom::configure(simulation_id,on_feedforward, reply_feedforward);
	if (old_feedforward != reply_feedforward)
		throw std::runtime_error("Feedforward reply functor changed");
}
