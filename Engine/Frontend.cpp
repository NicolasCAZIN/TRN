#include "stdafx.h"
#include "Frontend_impl.h"
#include "Broker_impl.h"

TRN::Engine::Frontend::Frontend(const std::shared_ptr<TRN::Engine::Communicator> &communicator) :
	TRN::Engine::Broker(communicator),
	handle(std::make_unique<Handle>())
{

}
TRN::Engine::Frontend::~Frontend()
{
	handle.reset();
}

void TRN::Engine::Frontend::install_completed(const std::function<void()> &functor)
{
	handle->on_completed = functor;
}
void TRN::Engine::Frontend::install_ack(const std::function<void(const unsigned long long &id, const std::size_t &number, const bool &success, const std::string &cause)> &functor)
{
	handle->on_ack = functor;
}
void TRN::Engine::Frontend::install_processor(const std::function<void(const int &rank, const std::string &host, const unsigned int &index, const std::string &name)> &functor)
{
	handle->on_processor = functor;
}
void TRN::Engine::Frontend::install_allocated(const std::function<void(const unsigned long long &id, const int &rank)> &functor)
{
	handle->on_allocated = functor;
}
void TRN::Engine::Frontend::install_deallocated(const std::function<void(const unsigned long long &id, const int &rank)> &functor)
{
	handle->on_deallocated = functor;
}
void TRN::Engine::Frontend::install_quit(const std::function<void(const int &rank)> &functor)
{
	handle->on_quit = functor;
}
void TRN::Engine::Frontend::install_configured(const std::function<void(const unsigned long long &id)> &functor)
{
	handle->on_configured = functor;
}
void TRN::Engine::Frontend::install_trained(const std::function<void(const unsigned long long &id)> &functor)
{
	handle->on_trained = functor;
}
void TRN::Engine::Frontend::install_primed(const std::function<void(const unsigned long long &id)> &functor)
{
	handle->on_primed = functor;
}
void TRN::Engine::Frontend::install_tested(const std::function<void(const unsigned long long &id)> &functor)
{
	handle->on_tested = functor;
}
void TRN::Engine::Frontend::install_error(const std::function<void(const std::string &message)> &functor)
{
	handle->on_error = functor;
}
void TRN::Engine::Frontend::install_information(const std::function<void(const std::string &message)> &functor)
{
	handle->on_information = functor;
}
void TRN::Engine::Frontend::install_warning(const std::function<void(const std::string &message)> &functor)
{
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


void TRN::Engine::Frontend::install_measurement_readout_mean_square_error(const unsigned long long &id, const std::function<void(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols)> &functor)
{
	set_map(handle->on_measurement_readout_mean_square_error, id, functor);
}
void TRN::Engine::Frontend::install_measurement_readout_frechet_distance(const unsigned long long &id, const std::function<void(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols)> &functor)
{
	set_map(handle->on_measurement_readout_frechet_distance, id, functor);
}
void TRN::Engine::Frontend::install_measurement_readout_custom(const unsigned long long &id, const std::function<void(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &primed, const std::vector<float> &predicted, const std::vector<float> &expected, const std::size_t &preamble, const std::size_t &pages, const std::size_t &rows, const  std::size_t &cols)> &functor)
{
	set_map(handle->on_measurement_readout_custom, id, functor);
}
void TRN::Engine::Frontend::install_measurement_position_mean_square_error(const unsigned long long &id, const std::function<void(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols)> &functor)
{
	set_map(handle->on_measurement_position_mean_square_error, id, functor);
}
void TRN::Engine::Frontend::install_measurement_position_frechet_distance(const unsigned long long &id, const std::function<void(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols)> &functor)
{
	set_map(handle->on_measurement_position_frechet_distance, id, functor);
}
void TRN::Engine::Frontend::install_measurement_position_custom(const unsigned long long &id, const std::function<void(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &primed, const std::vector<float> &predicted, const std::vector<float> &expected, const std::size_t &preamble, const std::size_t &pages, const std::size_t &rows, const  std::size_t &cols)> &functor)
{
	set_map(handle->on_measurement_position_custom, id, functor);
}
void TRN::Engine::Frontend::install_performances(const unsigned long long &id, const std::function<void(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::string &phase, const float &cycles_per_second, const float &gflops_per_second)> &functor)
{
	set_map(handle->on_performances, id, functor);
}
void TRN::Engine::Frontend::install_states(const unsigned long long &id, const std::function<void(const unsigned long long &id, const std::string &phase, const std::string &label, const std::size_t &batch, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &samples, const std::size_t &rows, const std::size_t &cols)> &functor)
{
	set_map(handle->on_states, id, functor);
}
void TRN::Engine::Frontend::install_weights(const unsigned long long &id, const std::function<void(const unsigned long long &id, const std::string &phase, const std::string &label, const std::size_t &batch, const std::size_t &trial, const std::vector<float> &samples, const std::size_t &rows, const std::size_t &cols)> &functor)
{
	set_map(handle->on_weights, id, functor);
}
void TRN::Engine::Frontend::install_position(const unsigned long long &id, const std::function<void(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols)> &functor)
{
	set_map(handle->on_position, id, functor);
}
void TRN::Engine::Frontend::install_stimulus(const unsigned long long &id, const std::function<void(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &stimulus, const std::size_t &rows, const std::size_t &cols)> &functor)
{
	set_map(handle->on_stimulus, id, functor);
}
void TRN::Engine::Frontend::install_mutator(const unsigned long long &id, const std::function<void(const unsigned long long &id, const unsigned long &seed, const std::size_t &trial, const std::vector<int> &offsets, const std::vector<int> &durations)> &functor)
{
	set_map(handle->on_mutator, id, functor);
}
void TRN::Engine::Frontend::install_scheduler(const unsigned long long &id, const std::function<void(const unsigned long long &id, const unsigned long &seed, const std::size_t &trial, const std::vector<float> &elements, const std::size_t &rows, const std::size_t &cols, const std::vector<int> &offsets, const std::vector<int> &durations)> &functor)
{
	set_map(handle->on_scheduler, id, functor);
}
void TRN::Engine::Frontend::install_scheduling(const unsigned long long &id, const std::function<void(const unsigned long long &id, const std::size_t &trial, const std::vector<int> &offsets, const std::vector<int> &durations)> &functor)
{
	set_map(handle->on_scheduling, id, functor);
}
void TRN::Engine::Frontend::install_feedforward(const unsigned long long &id, const std::function<void(const unsigned long long &id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const  std::size_t &cols)> &functor)
{
	set_map(handle->on_feedforward, id, functor);
}
void TRN::Engine::Frontend::install_feedback(const unsigned long long &id, const std::function<void(const unsigned long long &id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const  std::size_t &cols)> &functor)
{
	set_map(handle->on_feedback, id, functor);
}
void TRN::Engine::Frontend::install_readout(const unsigned long long &id, const std::function<void(const unsigned long long &id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const  std::size_t &cols)> &functor)
{
	set_map(handle->on_readout, id, functor);
}
void TRN::Engine::Frontend::install_recurrent(const unsigned long long &id, const std::function<void(const unsigned long long &id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &functor)
{
	set_map(handle->on_recurrent, id, functor);
}
void TRN::Engine::Frontend::callback_completed()
{
	if (handle->on_completed)
		handle->on_completed();
}

void TRN::Engine::Frontend::callback_ack(const unsigned long long &id, const std::size_t &number, const bool &success, const std::string &cause)
{
	if (handle->on_ack)
		handle->on_ack(id, number, success, cause);
}
void TRN::Engine::Frontend::callback_processor(const int &rank, const std::string &host, const unsigned int &index, const std::string &name)
{
	if (handle->on_processor)
		handle->on_processor(rank, host, index, name);
}
void TRN::Engine::Frontend::callback_allocated(const unsigned long long &id, const int &rank)
{
	if (handle->on_allocated)
		handle->on_allocated(id, rank);
}
void TRN::Engine::Frontend::callback_deallocated(const unsigned long long &id, const int &rank)
{
	if (handle->on_deallocated)
		handle->on_deallocated(id, rank);


}
void TRN::Engine::Frontend::callback_quit(const int &rank)
{
	if (handle->on_quit)
		handle->on_quit(rank);
}
void TRN::Engine::Frontend::callback_configured(const unsigned long long &id)
{
	if (handle->on_configured)
		handle->on_configured(id);
}
void TRN::Engine::Frontend::callback_trained(const unsigned long long &id)
{
	if (handle->on_trained)
		handle->on_trained(id);
}
void TRN::Engine::Frontend::callback_primed(const unsigned long long &id)
{
	if (handle->on_primed)
		handle->on_primed(id);
}
void TRN::Engine::Frontend::callback_tested(const unsigned long long &id)
{
	if (handle->on_tested)
		handle->on_tested(id);
}
void TRN::Engine::Frontend::callback_error(const std::string &message)
{
	std::cerr << "ERROR : " << message << std::endl;
	if (handle->on_error)
		handle->on_error(message);
}
void TRN::Engine::Frontend::callback_information(const std::string &message)
{
	std::cout << "INFORMATION : " << message << std::endl;
	if (handle->on_information)
		handle->on_information(message);
}
void TRN::Engine::Frontend::callback_warning(const std::string &message)
{
	std::cout << "WARNING : " << message << std::endl;
	if (handle->on_warning)
		handle->on_warning(message);
}
void TRN::Engine::Frontend::callback_measurement_readout_mean_square_error(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols)
{
	get_map(handle->on_measurement_readout_mean_square_error, id)(id, trial, evaluation, values, rows, cols);
}
void TRN::Engine::Frontend::callback_measurement_readout_frechet_distance(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols)
{
	get_map(handle->on_measurement_readout_frechet_distance, id)(id, trial, evaluation, values, rows, cols);
}
void TRN::Engine::Frontend::callback_measurement_readout_custom(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &primed, const std::vector<float> &predicted, const std::vector<float> &expected, const std::size_t &preamble, const std::size_t &pages, const std::size_t &rows, const  std::size_t &cols)
{
	get_map(handle->on_measurement_readout_custom, id)(id, trial, evaluation, primed, predicted, expected, preamble, pages, rows, cols);
}
void TRN::Engine::Frontend::callback_measurement_position_mean_square_error(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols)
{
	get_map(handle->on_measurement_position_mean_square_error, id)(id, trial, evaluation, values, rows, cols);
}
void TRN::Engine::Frontend::callback_measurement_position_frechet_distance(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols)
{
	get_map(handle->on_measurement_position_frechet_distance, id)(id, trial, evaluation, values, rows, cols);
}
void TRN::Engine::Frontend::callback_measurement_position_custom(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &primed, const std::vector<float> &predicted, const std::vector<float> &expected, const std::size_t &preamble, const std::size_t &pages, const std::size_t &rows, const  std::size_t &cols)
{
	get_map(handle->on_measurement_position_custom, id)(id, trial, evaluation, primed, predicted, expected, preamble, pages, rows, cols);
}
void TRN::Engine::Frontend::callback_performances(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::string &phase, const float &cycles_per_second, const float &gflops_per_second)
{
	get_map(handle->on_performances, id)(id, trial, evaluation, phase, cycles_per_second, gflops_per_second);
}
void TRN::Engine::Frontend::callback_states(const unsigned long long &id, const std::string &phase, const std::string &label, const std::size_t &batch, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &samples, const std::size_t &rows, const std::size_t &cols)
{
	get_map(handle->on_states, id)(id, phase, label, batch, trial, evaluation, samples, rows, cols);
}
void TRN::Engine::Frontend::callback_weights(const unsigned long long &id, const std::string &phase, const std::string &label, const std::size_t &batch, const std::size_t &trial, const std::vector<float> &samples, const std::size_t &rows, const std::size_t &cols)
{
	get_map(handle->on_weights, id)(id, phase, label, batch, trial, samples, rows, cols);
}

void TRN::Engine::Frontend::callback_scheduling(const unsigned long long &id, const std::size_t &trial, const std::vector<int> &offsets, const std::vector<int> &durations)
{
	get_map(handle->on_scheduling, id)(id, trial, offsets, durations);
}
void TRN::Engine::Frontend::callback_position(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols)
{
	get_map(handle->on_position, id)(id, trial, evaluation, position, rows, cols);
}
void TRN::Engine::Frontend::callback_stimulus(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &stimulus, const std::size_t &rows, const std::size_t &cols)
{
	get_map(handle->on_stimulus, id)(id, trial, evaluation, stimulus, rows, cols);
}

void TRN::Engine::Frontend::callback_mutator(const unsigned long long &id, const unsigned long &seed, const std::size_t &trial, const std::vector<int> &offsets, const std::vector<int> &durations)
{
	get_map(handle->on_mutator, id)(id, seed, trial, offsets, durations);
}
void TRN::Engine::Frontend::callback_scheduler(const unsigned long long &id, const unsigned long &seed, const std::size_t &trial, const std::vector<float> &elements, const std::size_t &rows, const std::size_t &cols, const std::vector<int> &offsets, const std::vector<int> &durations)
{
	get_map(handle->on_scheduler, id)(id, seed, trial, elements, rows, cols, offsets, durations);
}

void TRN::Engine::Frontend::callback_feedforward(const unsigned long long &id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const  std::size_t &cols)
{
	get_map(handle->on_feedforward, id)(id, seed, matrices, rows, cols);
}
void TRN::Engine::Frontend::callback_feedback(const unsigned long long &id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const  std::size_t &cols)
{
	get_map(handle->on_feedback, id)(id, seed, matrices, rows, cols);
}
void TRN::Engine::Frontend::callback_readout(const unsigned long long &id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const  std::size_t &cols)
{
	get_map(handle->on_readout, id)(id, seed, matrices, rows, cols);
}
void TRN::Engine::Frontend::callback_recurrent(const unsigned long long &id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)
{
	get_map(handle->on_recurrent, id)(id, seed, matrices, rows, cols);
}


std::shared_ptr<TRN::Engine::Frontend> TRN::Engine::Frontend::create(const std::shared_ptr<TRN::Engine::Communicator> &communicator)
{
	return std::make_shared<TRN::Engine::Frontend>(communicator);
}