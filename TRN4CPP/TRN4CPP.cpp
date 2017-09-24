#include "stdafx.h"
#include "TRN4CPP.h"

#include "ViewModel/Communicator.h"
#include "Frontend.h"
#include "Engine/Blocking.h"
#include "Engine/NonBlocking.h"

const bool TRN4CPP::DEFAULT_BLOCKING = true;
const bool TRN4CPP::DEFAULT_INITIALIZE = true;
const bool TRN4CPP::DEFAULT_TRAIN = false;
const bool TRN4CPP::DEFAULT_PRIME = false;
const bool TRN4CPP::DEFAULT_GENERATE = true;

const unsigned long TRN4CPP::DEFAULT_SEED = 1;
const std::size_t TRN4CPP::DEFAULT_BATCH_SIZE = 1;
const unsigned int TRN4CPP::DEFAULT_INDEX = 0;
const unsigned int TRN4CPP::DEFAULT_SUPPLEMENTARY_GENERATIONS = 0;
const std::string TRN4CPP::DEFAULT_HOST = "127.0.0.1";
const unsigned short TRN4CPP::DEFAULT_PORT = 12345;
const std::string TRN4CPP::DEFAULT_TAG = "";

static std::shared_ptr<TRN4CPP::Frontend> frontend;

static std::shared_ptr<TRN::Engine::Executor> executor;

static std::function<void(const unsigned int &id, const std::size_t &number, const bool &success, const std::string &cause)> ack;
static std::function<void(const int &rank, const std::string &host, const unsigned int &index, const std::string &name)> processor;
static std::function<void()> completed;
static std::function<void(const unsigned int &id, const int &rank)> allocation;
static std::function<void(const unsigned int &id, const int &rank)> deallocation;

void TRN4CPP::install_completed(const std::function<void()> &functor)
{
	if (frontend)
		throw std::runtime_error("A Frontend is already setup");
	completed = functor;
}

void TRN4CPP::install_ack(const std::function<void(const unsigned int &id, const std::size_t &number, const bool &success, const std::string &cause)> &functor)
{
	if (frontend)
		throw std::runtime_error("A Frontend is already setup");
	ack = functor;
}

void TRN4CPP::install_allocation(const std::function<void(const unsigned int &id, const int &rank)> &functor)
{
	if (frontend)
		throw std::runtime_error("A Frontend is already setup");
	allocation = functor;
}

void TRN4CPP::install_deallocation(const std::function<void(const unsigned int &id, const int &rank)> &functor)
{
	if (frontend)
		throw std::runtime_error("A Frontend is already setup");
	deallocation = functor;
}

void TRN4CPP::install_processor(const std::function<void(const int &rank, const std::string &host, const unsigned int &index, const std::string &name)> &functor)
{
	if (frontend)
		throw std::runtime_error("A Frontend is already setup");
	processor = functor;
}
void TRN4CPP::initialize_executor(const bool &blocking)
{
	if (executor)
		throw std::runtime_error("An executor is already setup");
	if (blocking)
		executor = TRN::Engine::Blocking::create();
	else
	{
		executor = TRN::Engine::NonBlocking::create();
	}
}

static void initialize_frontend(const std::shared_ptr<TRN::Engine::Communicator> &communicator = TRN::ViewModel::Communicator::Local::create({ 0 }))
{
	if (frontend)
		throw std::runtime_error("A Frontend is already setup");
	if (!executor)
	{
		std::cerr << "Executor is not ininitialized. Initializing non-blocking executor" << std::endl;
		TRN4CPP::initialize_executor(false);
	}
		
	frontend = TRN4CPP::Frontend::create(communicator, executor);
	if (completed)
		frontend->install_completed(completed);
	if (ack)
		frontend->install_ack(ack);
	if (processor)
		frontend->install_processor(processor);
	if (allocation)
		frontend->install_allocation(allocation);
	frontend->install_deallocation(deallocation);
	frontend->start();
}


void TRN4CPP::initialize_local(const std::vector<unsigned int> &indexes)
{
	initialize_frontend(TRN::ViewModel::Communicator::Local::create(indexes));
}
void TRN4CPP::initialize_remote(const std::string &host, const unsigned short &port)
{
	initialize_frontend(TRN::ViewModel::Communicator::Remote::create(host, port));
}
void TRN4CPP::initialize_distributed(int argc, char *argv[])
{
	initialize_frontend(TRN::ViewModel::Communicator::Distributed::create(argc, argv));
}

void TRN4CPP::run()
{
	if (!executor)
	{
		throw std::runtime_error("Executor is not setup");
	}
	executor->run();
}

void TRN4CPP::run_one()
{
	if (!executor)
	{
		throw std::runtime_error("Executor is not setup");
	}
	executor->run_one();
}

void TRN4CPP::uninitialize()
{
	if (!frontend)
		throw std::runtime_error("A frontend have never been setup");

	frontend.reset();
}

void TRN4CPP::allocate(const unsigned int &id)
{
	if (!frontend)
		initialize_frontend();

	frontend->allocate(id);
}
void TRN4CPP::deallocate(const unsigned int &id)
{
	if (!frontend)
		initialize_frontend();
	frontend->deallocate(id);
}
void TRN4CPP::train(const unsigned int &id, const std::string &label, const std::string &incoming, const std::string &expected)
{
	if (!frontend)
		initialize_frontend();
	frontend->train(id, label, incoming, expected);
}
void TRN4CPP::test(const unsigned int &id, const std::string &sequence, const std::string &incoming, const std::string &expected, const unsigned int &preamble, const unsigned int &supplementary_generations)
{
	if (!frontend)
		initialize_frontend();
	frontend->test(id, sequence, incoming, expected, preamble, supplementary_generations);
}
void TRN4CPP::declare_sequence(const unsigned int &id, const std::string &label, const std::string &tag, const std::vector<float> &sequence, const std::size_t &observations)
{
	if (!frontend)
		initialize_frontend();
	frontend->declare_sequence(id, label, tag, sequence, observations);
}

void TRN4CPP::declare_set(const unsigned int &id, const std::string &label, const std::string &tag, const std::vector<std::string> &labels)
{
	if (!frontend)
		initialize_frontend();
	frontend->declare_set(id, label, tag, labels);
}
void TRN4CPP::setup_states(const unsigned int &id, const std::function<void(const unsigned int &id, const std::string &phase, const std::string &label, const std::size_t &batch, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &samples, const std::size_t &rows, const std::size_t &cols)> &functor, const bool &train, const bool &prime, const bool &generate)
{
	if (!frontend)
		initialize_frontend();
	frontend->install_states(id, functor);
	frontend->setup_states(id, train, prime, generate);
}
void TRN4CPP::setup_weights(const unsigned int &id, const std::function<void(const unsigned int &id, const std::string &phase, const std::string &label, const std::size_t &batch, const std::size_t &trial, const std::vector<float> &weights, const std::size_t &rows, const std::size_t &cols)> &functor, const bool &initialize, const bool &train)
{
	if (!frontend)
		initialize_frontend();
	frontend->install_weights(id, functor);
	frontend->setup_weights(id, initialize, train);
}
void TRN4CPP::setup_performances(const unsigned int &id, const std::function<void(const unsigned int &id, const std::string &phase, const std::size_t &batch_size, const size_t &cycles, const float &gflops, const float &seconds)> &functor, const bool &train, const bool &prime, const bool &generate)
{
	if (!frontend)
		initialize_frontend();
	frontend->install_performances(id, functor);
	frontend->setup_performances(id, train, prime, generate);
}

void TRN4CPP::setup_scheduling(const unsigned int &id, const std::function<void(const unsigned int &id, const std::size_t &trial, const std::vector<int> &offsets, const std::vector<int> &durations)> &functor)
{
	if (!frontend)
		initialize_frontend();
	frontend->install_scheduling(id, functor);
	frontend->setup_scheduling(id);
}

void TRN4CPP::configure_begin(const unsigned int &id)
{
	if (!frontend)
		initialize_frontend();
	frontend->configure_begin(id);
}
void TRN4CPP::configure_end(const unsigned int &id)
{
	if (!frontend)
		initialize_frontend();
	frontend->configure_end(id);
}

void 	TRN4CPP::configure_measurement_readout_mean_square_error(const unsigned int &id, const std::size_t &batch_size, const std::function<void(const unsigned int &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const std::size_t &cols)> &functor)
{
	if (!frontend)
		initialize_frontend();
	frontend->install_measurement_readout_mean_square_error(id, functor);
	frontend->configure_measurement_readout_mean_square_error(id, batch_size);
}

void 	TRN4CPP::configure_measurement_readout_frechet_distance(const unsigned int &id, const std::size_t &batch_size, const std::function<void(const unsigned int &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const std::size_t &cols)> &functor)
{
	if (!frontend)
		initialize_frontend();
	frontend->install_measurement_readout_frechet_distance(id, functor);
	frontend->configure_measurement_readout_frechet_distance(id, batch_size);
}
void 	TRN4CPP::configure_measurement_readout_custom(const unsigned int &id, const std::size_t &batch_size, const std::function<void(const unsigned int &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &primed, const std::vector<float> &predicted, const std::vector<float> &expected, const std::size_t &preamble, const std::size_t &pages, const std::size_t &rows, const std::size_t &cols)> &functor)
{
	if (!frontend)
		initialize_frontend();
	frontend->install_measurement_readout_custom(id, functor);
	frontend->configure_measurement_readout_custom(id, batch_size);
}

void 	TRN4CPP::configure_measurement_position_mean_square_error(const unsigned int &id, const std::size_t &batch_size, const std::function<void(const unsigned int &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const std::size_t &cols)> &functor)
{
	if (!frontend)
		initialize_frontend();
	frontend->install_measurement_position_mean_square_error(id, functor);
	frontend->configure_measurement_position_mean_square_error(id, batch_size);
}

void 	TRN4CPP::configure_measurement_position_frechet_distance(const unsigned int &id, const std::size_t &batch_size, const std::function<void(const unsigned int &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const std::size_t &cols)> &functor)
{
	if (!frontend)
		initialize_frontend();
	frontend->install_measurement_position_frechet_distance(id, functor);
	frontend->configure_measurement_position_frechet_distance(id, batch_size);
}
void 	TRN4CPP::configure_measurement_position_custom(const unsigned int &id, const std::size_t &batch_size, const std::function<void(const unsigned int &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &primed, const std::vector<float> &predicted, const std::vector<float> &expected, const std::size_t &preamble, const std::size_t &pages, const std::size_t &rows, const std::size_t &cols)> &functor)
{
	if (!frontend)
		initialize_frontend();
	frontend->install_measurement_position_custom(id, functor);
	frontend->configure_measurement_position_custom(id, batch_size);
}

void TRN4CPP::configure_reservoir_widrow_hoff(const unsigned int &id, const std::size_t &stimulus_size, const std::size_t &prediction_size, const std::size_t &reservoir_size, const float &leak_rate, const float &initial_state_scale, const float &learning_rate, const unsigned long &seed, const std::size_t &batch_size)
{
	if (!frontend)
		initialize_frontend();
	frontend->configure_reservoir_widrow_hoff(id, stimulus_size, prediction_size, reservoir_size, leak_rate, initial_state_scale, learning_rate, seed, batch_size);
}

void TRN4CPP::configure_loop_copy(const unsigned int &id, const std::size_t &batch_size, const std::size_t &stimulus_size)
{
	if (!frontend)
		initialize_frontend();
	frontend->configure_loop_copy(id, batch_size, stimulus_size);
}
void TRN4CPP::configure_loop_spatial_filter(const unsigned int &id, const std::size_t &batch_size, const std::size_t &stimulus_size, const unsigned long &seed,
	const std::function<void(const unsigned int &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols)> &predicted_position,
	std::function<void(const unsigned int &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols)> &estimated_position,
	const std::function<void(const unsigned int &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols)> &predicted_stimulus,
	std::function<void(const unsigned int &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &stimulus, const std::size_t &rows, const std::size_t &cols)> &perceived_stimulus,
	const std::size_t &rows, const std::size_t &cols, const std::pair<float, float> &x, const std::pair<float, float> &y,
	const std::vector<float> response, const float &sigma, const float &radius, const float &scale, const std::string &tag
	
	)
{
	if (!frontend)
		initialize_frontend();
	estimated_position = std::bind(&TRN::Engine::Broker::notify_position, frontend, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5, std::placeholders::_6);
	perceived_stimulus = std::bind(&TRN::Engine::Broker::notify_stimulus, frontend, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5, std::placeholders::_6);
	frontend->install_position(id, predicted_position);
	frontend->install_stimulus(id, predicted_stimulus);
	frontend->configure_loop_spatial_filter(id, batch_size, stimulus_size, seed,
		rows, cols, x, y, response, sigma, radius, scale, tag);
}
void TRN4CPP::configure_loop_custom(const unsigned int &id, const std::size_t &batch_size, const std::size_t &stimulus_size,
	const std::function<void(const unsigned int &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &prediction, const std::size_t &rows, const std::size_t &cols)> &request,
	std::function<void(const unsigned int &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &stimulus, const std::size_t &rows, const std::size_t &cols)> &reply
	)
{
	if (!frontend)
		initialize_frontend();
	reply = std::bind(&TRN::Engine::Broker::notify_stimulus, frontend, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5, std::placeholders::_6);
	frontend->install_stimulus(id, request);
	frontend->configure_loop_custom(id, batch_size, stimulus_size);
}
void TRN4CPP::configure_scheduler_tiled(const unsigned int &id, const unsigned int &epochs)
{
	if (!frontend)
		initialize_frontend();
	frontend->configure_scheduler_tiled(id, epochs);
}
void TRN4CPP::configure_scheduler_snippets(const unsigned int &id, const unsigned long &seed, const unsigned int &snippets_size, const unsigned int &time_budget, const std::string &tag)
{
	if (!frontend)
		initialize_frontend();
	frontend->configure_scheduler_snippets(id, seed, snippets_size, time_budget, tag);
}
void TRN4CPP::configure_scheduler_custom(const unsigned int &id, const unsigned long &seed,
	const std::function<void(const unsigned int &id, const unsigned long &seed, const std::size_t &trial, const std::vector<float> &elements, const std::size_t &rows, const std::size_t &cols, const std::vector<int> &offsets, const std::vector<int> &durations)> &request,
	std::function<void(const unsigned int &id, const std::size_t &trial, const std::vector<int> &offsets, const std::vector<int> &durations)> &reply, const std::string &tag)
{
	if (!frontend)
		initialize_frontend();
	reply = std::bind(&TRN::Engine::Broker::notify_scheduler, frontend, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4);
	frontend->install_scheduler(id, request);
	frontend->configure_scheduler_custom(id, seed, tag);
}

void TRN4CPP::configure_mutator_shuffle(const unsigned int &id, const unsigned long &seed)
{
	if (!frontend)
		initialize_frontend();
	frontend->configure_mutator_shuffle(id, seed);
}
void TRN4CPP::configure_mutator_reverse(const unsigned int &id, const unsigned long &seed, const float &rate, const std::size_t &size)
{
	if (!frontend)
		initialize_frontend();
	frontend->configure_mutator_reverse(id, seed, rate, size);
}
void TRN4CPP::configure_mutator_punch(const unsigned int &id, const unsigned long &seed, const float &rate, const std::size_t &size, const std::size_t &number)
{
	if (!frontend)
		initialize_frontend();
	frontend->configure_mutator_punch(id, seed, rate, size, number);
}
void TRN4CPP::configure_mutator_custom(const unsigned int &id, const unsigned long &seed,
	const std::function<void(const unsigned int &id, const unsigned long &seed, const std::size_t &trial, const std::vector<int> &offsets, const std::vector<int> &durations)> &request,
	std::function<void(const unsigned int &id, const std::size_t &trial, const std::vector<int> &offsets, const std::vector<int> &durations)> &reply
)
{
	if (!frontend)
		initialize_frontend();
	reply = std::bind(&TRN::Engine::Broker::notify_mutator, frontend, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4);
	frontend->install_mutator(id, request);
	frontend->configure_mutator_custom(id, seed);
}


void TRN4CPP::configure_readout_uniform(const unsigned int &id, const float &a, const float &b, const float &sparsity)
{
	if (!frontend)
		initialize_frontend();
	frontend->configure_readout_uniform(id, a, b, sparsity);
}
void TRN4CPP::configure_readout_gaussian(const unsigned int &id, const float &mu, const float &sigma)
{
	if (!frontend)
		initialize_frontend();
	frontend->configure_readout_gaussian(id, mu, sigma);
}
void TRN4CPP::configure_readout_custom(const unsigned int &id, const std::function<void(const unsigned int &id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &request,
	std::function<void(const unsigned int &id, const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &reply)
{
	if (!frontend)
		initialize_frontend();
	reply = std::bind(&TRN::Engine::Broker::notify_readout, frontend, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5);
	frontend->install_readout(id, request);
	frontend->configure_readout_custom(id);
}

void TRN4CPP::configure_feedback_uniform(const unsigned int &id, const float &a, const float &b, const float &sparsity)
{
	if (!frontend)
		initialize_frontend();
	frontend->configure_feedback_uniform(id, a, b, sparsity);
}
void TRN4CPP::configure_feedback_gaussian(const unsigned int &id, const float &mu, const float &sigma)
{
	if (!frontend)
		initialize_frontend();
	frontend->configure_feedback_gaussian(id, mu, sigma);
}
void TRN4CPP::configure_feedback_custom(const unsigned int &id, const std::function<void(const unsigned int &id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &request,
	std::function<void(const unsigned int &id, const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &reply)
{
	if (!frontend)
		initialize_frontend();
	reply = std::bind(&TRN::Engine::Broker::notify_feedback, frontend, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5);
	frontend->install_feedback(id, request);
	frontend->configure_feedback_custom(id);
}

void TRN4CPP::configure_recurrent_uniform(const unsigned int &id, const float &a, const float &b, const float &sparsity)
{
	if (!frontend)
		initialize_frontend();
	frontend->configure_recurrent_uniform(id, a, b, sparsity);
}
void TRN4CPP::configure_recurrent_gaussian(const unsigned int &id, const float &mu, const float &sigma)
{
	if (!frontend)
		initialize_frontend();
	frontend->configure_recurrent_gaussian(id, mu, sigma);
}
void TRN4CPP::configure_recurrent_custom(const unsigned int &id, const std::function<void(const unsigned int &id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &request,
	std::function<void(const unsigned int &id, const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &reply)
{
	if (!frontend)
		initialize_frontend();
	reply = std::bind(&TRN::Engine::Broker::notify_recurrent, frontend, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5);
	frontend->install_recurrent(id, request);
	frontend->configure_recurrent_custom(id);
}

void TRN4CPP::configure_feedforward_uniform(const unsigned int &id, const float &a, const float &b, const float &sparsity)
{
	if (!frontend)
		initialize_frontend();
	frontend->configure_feedforward_uniform(id, a, b, sparsity);
}
void TRN4CPP::configure_feedforward_gaussian(const unsigned int &id, const float &mu, const float &sigma)
{
	if (!frontend)
		initialize_frontend();
	frontend->configure_feedforward_gaussian(id, mu, sigma);
}
void TRN4CPP::configure_feedforward_custom(const unsigned int &id, const std::function<void(const unsigned int &id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &request,
	std::function<void(const unsigned int &id, const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &reply)
{
	if (!frontend)
		initialize_frontend();
	reply = std::bind(&TRN::Engine::Broker::notify_feedforward, frontend, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5);
	frontend->install_feedforward(id, request);
	frontend->configure_feedforward_custom(id);
}