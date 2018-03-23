#include "stdafx.h"
#include "Frontend_impl.h"
#include "Broker_impl.h"
#include "Helper/Logger.h"

TRN::Engine::Frontend::Frontend(const std::shared_ptr<TRN::Engine::Communicator> &communicator) :
	TRN::Engine::Broker(communicator),
	handle(std::make_unique<Handle>())
{
	TRACE_LOGGER;
}
TRN::Engine::Frontend::~Frontend()
{
	TRACE_LOGGER;
	handle.reset();
}

void TRN::Engine::Frontend::initialize()
{
	TRACE_LOGGER;
	TRN::Engine::Broker::initialize();
	TRN::Engine::Message<TRN::Engine::START> start;
	start.number = 0;

	TRN::Engine::Broker::handle->communicator->broadcast(start);
}
void TRN::Engine::Frontend::uninitialize()
{
	TRACE_LOGGER;
	TRN::Engine::Broker::uninitialize();
}
void TRN::Engine::Frontend::install_completed(const std::function<void()> &functor)
{
	TRACE_LOGGER;
	handle->on_completed = functor;
}
void TRN::Engine::Frontend::install_ack(const std::function<void(const unsigned long long &simulation_id, const std::size_t &counter, const bool &success, const std::string &cause)> &functor)
{
	TRACE_LOGGER;
	handle->on_ack = functor;
}
void TRN::Engine::Frontend::install_processor(const std::function<void(const int &rank, const std::string &host, const unsigned int &index, const std::string &name)> &functor)
{
	TRACE_LOGGER;
	handle->on_processor = functor;
}
void TRN::Engine::Frontend::install_allocated(const std::function<void(const unsigned long long &simulation_id, const int &rank)> &functor)
{
	TRACE_LOGGER;
	handle->on_allocated = functor;
}
void TRN::Engine::Frontend::install_deallocated(const std::function<void(const unsigned long long &simulation_id, const int &rank)> &functor)
{
	TRACE_LOGGER;
	handle->on_deallocated = functor;
}
void TRN::Engine::Frontend::install_quit(const std::function<void(const int &rank)> &functor)
{
	TRACE_LOGGER;
	handle->on_quit = functor;
}
void TRN::Engine::Frontend::install_configured(const std::function<void(const unsigned long long &simulation_id)> &functor)
{
	TRACE_LOGGER;
	handle->on_configured = functor;
}
void TRN::Engine::Frontend::install_trained(const std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id)> &functor)
{
	TRACE_LOGGER;
	handle->on_trained = functor;
}
void TRN::Engine::Frontend::install_primed(const std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id)> &functor)
{
	TRACE_LOGGER;
	handle->on_primed = functor;
}
void TRN::Engine::Frontend::install_tested(const std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id)> &functor)
{
	TRACE_LOGGER;
	handle->on_tested = functor;
}
void TRN::Engine::Frontend::install_error(const std::function<void(const std::string &message)> &functor)
{
	TRACE_LOGGER;
	handle->on_error = functor;
}
void TRN::Engine::Frontend::install_information(const std::function<void(const std::string &message)> &functor)
{
	TRACE_LOGGER;
	handle->on_information = functor;
}
void TRN::Engine::Frontend::install_warning(const std::function<void(const std::string &message)> &functor)
{
	TRACE_LOGGER;
	handle->on_warning = functor;
}

template<typename Key, typename Value>
static void set_map(std::map<Key, Value> &map, const Key &key, const Value &value)
{
	if (map.find(key) != map.end())
	{
		throw std::invalid_argument("Functor is already installed");
	}
	map[key] = value;
}

template<typename Key, typename Value>
static  Value &get_map(std::map<Key, Value> &map, const Key &key)
{
	if (map.find(key) == map.end())
	{
		throw std::invalid_argument("Functor is not installed");
	}
	return map[key];
}


void TRN::Engine::Frontend::install_measurement_readout_mean_square_error(const unsigned long long &simulation_id, const std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols)> &functor)
{
	TRACE_LOGGER;
	set_map(handle->on_measurement_readout_mean_square_error,simulation_id, functor);
}
void TRN::Engine::Frontend::install_measurement_readout_frechet_distance(const unsigned long long &simulation_id, const std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols)> &functor)
{
	TRACE_LOGGER;
	set_map(handle->on_measurement_readout_frechet_distance,simulation_id, functor);
}
void TRN::Engine::Frontend::install_measurement_readout_custom(const unsigned long long &simulation_id, const std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &primed, const std::vector<float> &predicted, const std::vector<float> &expected, const std::size_t &preamble, const std::size_t &pages, const std::size_t &rows, const  std::size_t &cols)> &functor)
{
	TRACE_LOGGER;
	set_map(handle->on_measurement_readout_custom,simulation_id, functor);
}
void TRN::Engine::Frontend::install_measurement_position_mean_square_error(const unsigned long long &simulation_id, const std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols)> &functor)
{
	TRACE_LOGGER;
	set_map(handle->on_measurement_position_mean_square_error,simulation_id, functor);
}
void TRN::Engine::Frontend::install_measurement_position_frechet_distance(const unsigned long long &simulation_id, const std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols)> &functor)
{
	TRACE_LOGGER;
	set_map(handle->on_measurement_position_frechet_distance,simulation_id, functor);
}
void TRN::Engine::Frontend::install_measurement_position_custom(const unsigned long long &simulation_id, const std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &primed, const std::vector<float> &predicted, const std::vector<float> &expected, const std::size_t &preamble, const std::size_t &pages, const std::size_t &rows, const  std::size_t &cols)> &functor)
{
	TRACE_LOGGER;
	set_map(handle->on_measurement_position_custom,simulation_id, functor);
}
void TRN::Engine::Frontend::install_performances(const unsigned long long &simulation_id, const std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::string &phase, const float &cycles_per_second, const float &gflops_per_second)> &functor)
{
	TRACE_LOGGER;
	set_map(handle->on_performances,simulation_id, functor);
}
void TRN::Engine::Frontend::install_states(const unsigned long long &simulation_id, const std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::string &phase, const std::string &label, const std::size_t &batch, const std::vector<float> &samples, const std::size_t &rows, const std::size_t &cols)> &functor)
{
	TRACE_LOGGER;
	set_map(handle->on_states,simulation_id, functor);
}
void TRN::Engine::Frontend::install_weights(const unsigned long long &simulation_id, const std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::string &phase, const std::string &label, const std::size_t &batch, const std::vector<float> &samples, const std::size_t &rows, const std::size_t &cols)> &functor)
{
	TRACE_LOGGER;
	set_map(handle->on_weights,simulation_id, functor);
}
void TRN::Engine::Frontend::install_position(const unsigned long long &simulation_id, const std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols)> &functor)
{
	TRACE_LOGGER;
	set_map(handle->on_position,simulation_id, functor);
}
void TRN::Engine::Frontend::install_stimulus(const unsigned long long &simulation_id, const std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &stimulus, const std::size_t &rows, const std::size_t &cols)> &functor)
{
	TRACE_LOGGER;
	set_map(handle->on_stimulus,simulation_id, functor);
}
void TRN::Engine::Frontend::install_mutator(const unsigned long long &simulation_id, const std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const unsigned long &seed,const std::vector<int> &offsets, const std::vector<int> &durations)> &functor)
{
	TRACE_LOGGER;
	set_map(handle->on_mutator,simulation_id, functor);
}
void TRN::Engine::Frontend::install_scheduler(const unsigned long long &simulation_id, const std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const unsigned long &seed, const std::vector<float> &elements, const std::size_t &rows, const std::size_t &cols, const std::vector<int> &offsets, const std::vector<int> &durations)> &functor)
{
	TRACE_LOGGER;
	set_map(handle->on_scheduler,simulation_id, functor);
}
void TRN::Engine::Frontend::install_scheduling(const unsigned long long &simulation_id, const std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<int> &offsets, const std::vector<int> &durations)> &functor)
{
	TRACE_LOGGER;
	set_map(handle->on_scheduling,simulation_id, functor);
}
void TRN::Engine::Frontend::install_feedforward(const unsigned long long &simulation_id, const std::function<void(const unsigned long long &simulation_id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const  std::size_t &cols)> &functor)
{
	TRACE_LOGGER;
	set_map(handle->on_feedforward,simulation_id, functor);
}
void TRN::Engine::Frontend::install_feedback(const unsigned long long &simulation_id, const std::function<void(const unsigned long long &simulation_id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const  std::size_t &cols)> &functor)
{
	TRACE_LOGGER;
	set_map(handle->on_feedback,simulation_id, functor);
}
void TRN::Engine::Frontend::install_readout(const unsigned long long &simulation_id, const std::function<void(const unsigned long long &simulation_id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const  std::size_t &cols)> &functor)
{
	TRACE_LOGGER;
	set_map(handle->on_readout,simulation_id, functor);
}
void TRN::Engine::Frontend::install_recurrent(const unsigned long long &simulation_id, const std::function<void(const unsigned long long &simulation_id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &functor)
{
	TRACE_LOGGER;
	set_map(handle->on_recurrent,simulation_id, functor);
}
void TRN::Engine::Frontend::callback_completed()
{
	TRACE_LOGGER;
	if (handle->on_completed)
		handle->on_completed();
}

void TRN::Engine::Frontend::callback_ack(const unsigned long long &simulation_id, const std::size_t &counter, const bool &success, const std::string &cause)
{
	TRACE_LOGGER;
	if (handle->on_ack)
		handle->on_ack(simulation_id, counter, success, cause);
}
void TRN::Engine::Frontend::callback_processor(const int &rank, const std::string &host, const unsigned int &index, const std::string &name)
{
	TRACE_LOGGER;
	if (handle->on_processor)
		handle->on_processor(rank, host, index, name);
}
void TRN::Engine::Frontend::callback_allocated(const unsigned long long &simulation_id, const int &rank)
{
	TRACE_LOGGER;
	if (handle->on_allocated)
		handle->on_allocated(simulation_id, rank);
}
void TRN::Engine::Frontend::callback_deallocated(const unsigned long long &simulation_id, const int &rank)
{
	TRACE_LOGGER;
	if (handle->on_deallocated)
		handle->on_deallocated(simulation_id, rank);
}
void TRN::Engine::Frontend::callback_terminated(const int &rank)
{
	TRACE_LOGGER;
	DEBUG_LOGGER <<   "Worker #" << rank << " terminated";

	if (handle->on_quit)
		handle->on_quit(rank);

}
void TRN::Engine::Frontend::callback_exit(const unsigned short &number, const int &rank)
{
	TRACE_LOGGER;
	DEBUG_LOGGER <<   "Worker #" << rank << " exiting for client #" << number;

	TRN::Engine::Message<TRN::Engine::STOP> stop;
	stop.number = 0;

	TRN::Engine::Broker::handle->communicator->send(stop, rank);
}


void TRN::Engine::Frontend::callback_configured(const unsigned long long &simulation_id)
{
	TRACE_LOGGER;
	if (handle->on_configured)
		handle->on_configured(simulation_id);
}
void TRN::Engine::Frontend::callback_trained(const unsigned long long &simulation_id, const unsigned long long &evaluation_id)
{
	TRACE_LOGGER;
	if (handle->on_trained)
		handle->on_trained(simulation_id, evaluation_id);
}
void TRN::Engine::Frontend::callback_primed(const unsigned long long &simulation_id, const unsigned long long &evaluation_id)
{
	TRACE_LOGGER;
	if (handle->on_primed)
		handle->on_primed(simulation_id, evaluation_id);
}
void TRN::Engine::Frontend::callback_tested(const unsigned long long &simulation_id, const unsigned long long &evaluation_id)
{
	TRACE_LOGGER;
	if (handle->on_tested)
		handle->on_tested(simulation_id, evaluation_id);
}
void TRN::Engine::Frontend::callback_error(const std::string &message)
{
	TRACE_LOGGER;
	ERROR_LOGGER <<  message ;
	if (handle->on_error)
		handle->on_error(message);
}
void TRN::Engine::Frontend::callback_information(const std::string &message)
{
	TRACE_LOGGER;
	INFORMATION_LOGGER <<    message ;
	if (handle->on_information)
		handle->on_information(message);
}
void TRN::Engine::Frontend::callback_warning(const std::string &message)
{
	TRACE_LOGGER;
	INFORMATION_LOGGER <<    message ;
	if (handle->on_warning)
		handle->on_warning(message);
}
void TRN::Engine::Frontend::callback_measurement_readout_mean_square_error(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols)
{
	TRACE_LOGGER;
	get_map(handle->on_measurement_readout_mean_square_error, simulation_id)(simulation_id, evaluation_id, values, rows, cols);
}
void TRN::Engine::Frontend::callback_measurement_readout_frechet_distance(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols)
{
	TRACE_LOGGER;
	get_map(handle->on_measurement_readout_frechet_distance, simulation_id)(simulation_id, evaluation_id, values, rows, cols);
}
void TRN::Engine::Frontend::callback_measurement_readout_custom(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &primed, const std::vector<float> &predicted, const std::vector<float> &expected, const std::size_t &preamble, const std::size_t &pages, const std::size_t &rows, const  std::size_t &cols)
{
	TRACE_LOGGER;
	get_map(handle->on_measurement_readout_custom, simulation_id)(simulation_id, evaluation_id, primed, predicted, expected, preamble, pages, rows, cols);
}
void TRN::Engine::Frontend::callback_measurement_position_mean_square_error(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols)
{
	TRACE_LOGGER;
	get_map(handle->on_measurement_position_mean_square_error, simulation_id)(simulation_id, evaluation_id, values, rows, cols);
}
void TRN::Engine::Frontend::callback_measurement_position_frechet_distance(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols)
{
	TRACE_LOGGER;
	get_map(handle->on_measurement_position_frechet_distance, simulation_id)(simulation_id, evaluation_id, values, rows, cols);
}
void TRN::Engine::Frontend::callback_measurement_position_custom(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &primed, const std::vector<float> &predicted, const std::vector<float> &expected, const std::size_t &preamble, const std::size_t &pages, const std::size_t &rows, const  std::size_t &cols)
{
	TRACE_LOGGER;
	get_map(handle->on_measurement_position_custom, simulation_id)(simulation_id, evaluation_id, primed, predicted, expected, preamble, pages, rows, cols);
}
void TRN::Engine::Frontend::callback_performances(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::string &phase, const float &cycles_per_second, const float &gflops_per_second)
{
	TRACE_LOGGER;
	get_map(handle->on_performances, simulation_id)(simulation_id, evaluation_id, phase, cycles_per_second, gflops_per_second);
}
void TRN::Engine::Frontend::callback_states(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::string &phase, const std::string &label, const std::size_t &batch, const std::vector<float> &samples, const std::size_t &rows, const std::size_t &cols)
{
	TRACE_LOGGER;
	get_map(handle->on_states, simulation_id)(simulation_id, evaluation_id, phase, label, batch, samples, rows, cols);
}
void TRN::Engine::Frontend::callback_weights(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::string &phase, const std::string &label, const std::size_t &batch, const std::vector<float> &samples, const std::size_t &rows, const std::size_t &cols)
{
	TRACE_LOGGER;
	get_map(handle->on_weights, simulation_id)(simulation_id, evaluation_id, phase, label, batch,  samples, rows, cols);
}

void TRN::Engine::Frontend::callback_scheduling(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<int> &offsets, const std::vector<int> &durations)
{
	TRACE_LOGGER;
	get_map(handle->on_scheduling, simulation_id)(simulation_id, evaluation_id, offsets, durations);
}
void TRN::Engine::Frontend::callback_position(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols)
{
	TRACE_LOGGER;
	get_map(handle->on_position, simulation_id)(simulation_id, evaluation_id, position, rows, cols);
}
void TRN::Engine::Frontend::callback_stimulus(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &stimulus, const std::size_t &rows, const std::size_t &cols)
{
	TRACE_LOGGER;
	get_map(handle->on_stimulus,simulation_id)(simulation_id, evaluation_id, stimulus, rows, cols);
}

void TRN::Engine::Frontend::callback_mutator(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const unsigned long &seed, const std::vector<int> &offsets, const std::vector<int> &durations)
{
	TRACE_LOGGER;
	get_map(handle->on_mutator, simulation_id)(simulation_id, evaluation_id, seed, offsets, durations);
}
void TRN::Engine::Frontend::callback_scheduler(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const unsigned long &seed, const std::vector<float> &elements, const std::size_t &rows, const std::size_t &cols, const std::vector<int> &offsets, const std::vector<int> &durations)
{
	TRACE_LOGGER;
	get_map(handle->on_scheduler, simulation_id)(simulation_id, evaluation_id, seed, elements, rows, cols, offsets, durations);
}

void TRN::Engine::Frontend::callback_feedforward(const unsigned long long &simulation_id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const  std::size_t &cols)
{
	TRACE_LOGGER;
	get_map(handle->on_feedforward, simulation_id)(simulation_id, seed, matrices, rows, cols);
}

void TRN::Engine::Frontend::callback_readout(const unsigned long long &simulation_id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const  std::size_t &cols)
{
	TRACE_LOGGER;
	get_map(handle->on_readout, simulation_id)(simulation_id, seed, matrices, rows, cols);
}
void TRN::Engine::Frontend::callback_recurrent(const unsigned long long &simulation_id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)
{
	TRACE_LOGGER;
	get_map(handle->on_recurrent, simulation_id)(simulation_id, seed, matrices, rows, cols);
}


std::shared_ptr<TRN::Engine::Frontend> TRN::Engine::Frontend::create(const std::shared_ptr<TRN::Engine::Communicator> &communicator)
{
	TRACE_LOGGER;
	return std::make_shared<TRN::Engine::Frontend>(communicator);
}