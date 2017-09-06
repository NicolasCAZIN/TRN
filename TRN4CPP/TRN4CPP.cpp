#include "stdafx.h"
#include "TRN4CPP.h"

#include "ViewModel/Communicator.h"
#include "ViewModel/Broker.h"

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
static std::mutex mutex;

static std::map<unsigned int, std::function<void(const std::vector<float> &stimulus, const std::size_t &rows, const std::size_t &cols)>> id_perceived_stimulus;
static std::map<unsigned int, std::function<void(const std::vector<float> &estimated_position, const std::size_t &rows, const std::size_t &cols)>> id_estimated_position;
static std::map<unsigned int, std::function<void(const std::vector<int> &offsets, const std::vector<int> &durations)>> id_scheduler;
static std::map<unsigned int, std::function<void(const std::vector<int> &offsets, const std::vector<int> &durations)>> id_mutator;
static std::map<unsigned int, std::function<void(const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)>> id_feedforward_weights;
static std::map<unsigned int, std::function<void(const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)>> id_feedback_weights;
static std::map<unsigned int, std::function<void(const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)>> id_readout_weights;
static std::map<unsigned int, std::function<void(const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)>> id_recurrent_weights;


static void api_deallocation(const unsigned int &id, const int &rank)
{
	 std::unique_lock<std::mutex> lock(mutex);

	if (id_perceived_stimulus.find(id) != id_perceived_stimulus.end())
		id_perceived_stimulus.erase(id);
	if (id_estimated_position.find(id) != id_estimated_position.end())
		id_estimated_position.erase(id);
	if (id_scheduler.find(id) != id_scheduler.end())
		id_scheduler.erase(id);
	if (id_mutator.find(id) != id_mutator.end())
		id_mutator.erase(id);
	if (id_feedforward_weights.find(id) != id_feedforward_weights.end())
		id_feedforward_weights.erase(id);
	if (id_feedback_weights.find(id) != id_feedback_weights.end())
		id_feedback_weights.erase(id);
	if (id_readout_weights.find(id) != id_readout_weights.end())
		id_readout_weights.erase(id);
	if (id_recurrent_weights.find(id) != id_recurrent_weights.end())
		id_recurrent_weights.erase(id);
}

static std::shared_ptr<TRN::Engine::Broker> broker;
static std::function<void(const int &rank, const std::string &host, const unsigned int &index, const std::string &name)> processor;
static std::function<void(const unsigned int &id, const int &rank)> allocation;
static std::function<void(const unsigned int &id, const int &rank)> deallocation = api_deallocation;



void TRN4CPP::install_allocation(const std::function<void(const unsigned int &id, const int &rank)> &functor)
{
	std::unique_lock<std::mutex> lock(mutex);
	if (broker)
		throw std::runtime_error("A broker is already setup");
	allocation = functor;
}

void TRN4CPP::install_deallocation(const std::function<void(const unsigned int &id, const int &rank)> &functor)
{
	std::unique_lock<std::mutex> lock(mutex);
	if (broker)
		throw std::runtime_error("A broker is already setup");
	deallocation = [=](const unsigned int &id, const int &rank)
	{
		functor(id, rank);
		api_deallocation(id, rank);
	};
}

void TRN4CPP::install_processor(const std::function<void(const int &rank, const std::string &host, const unsigned int &index, const std::string &name)> &functor)
{
	std::unique_lock<std::mutex> lock(mutex);
	if (broker)
		throw std::runtime_error("A broker is already setup");
	processor = functor;
}

static void initialize_broker(const std::shared_ptr<TRN::Engine::Communicator> &communicator)
{

	std::unique_lock<std::mutex> lock(mutex);
	if (broker)
		throw std::runtime_error("A broker is already setup");
	// // std::unique_lock<std::mutex> lock(mutex);

	broker = TRN::ViewModel::Broker::create(communicator);

	if (processor)
		broker->setup_processor(processor);
	if (allocation)
		broker->setup_allocation(allocation);
	broker->setup_deallocation(deallocation);
	broker->start();
}

void TRN4CPP::initialize_local(const std::list<unsigned int> &indexes)
{
	initialize_broker(TRN::ViewModel::Communicator::Local::create(indexes));
	// std::cout << "TRN4CPP : sucessful call to " << __FUNCTION__ << std::endl;
}
void TRN4CPP::initialize_remote(const std::string &host, const unsigned short &port)
{
	// std::unique_lock<std::mutex> lock(mutex);
	if (broker)
		throw std::runtime_error("A broker is already setup");
	//
	//broker = TRN::ViewModel::Broker::Remote::create(host, port);
	//// std::cout << "TRN4CPP : sucessful call to " << __FUNCTION__ << std::endl;
}
void TRN4CPP::initialize_distributed(int argc, char *argv[])
{
	initialize_broker(TRN::ViewModel::Communicator::Distributed::create(argc, argv));
}

void TRN4CPP::uninitialize()
{
	if (!broker)
		throw std::runtime_error("A broker have never been setup");
	// // std::unique_lock<std::mutex> lock(mutex);
	broker->stop();
	broker.reset();
	std::unique_lock<std::mutex> lock(mutex);
	id_perceived_stimulus.clear();
	id_estimated_position.clear();
	id_scheduler.clear();
	id_mutator.clear();
	id_feedforward_weights.clear();
	id_feedback_weights.clear();
	id_readout_weights.clear();
	id_recurrent_weights.clear();
}

void TRN4CPP::allocate(const unsigned int &id)
{
	// std::unique_lock<std::mutex> lock(mutex);
	// // std::unique_lock<std::mutex> lock(mutex);
	if (!broker)
		throw std::logic_error("broker had not been initialized");

	broker->allocate(id);
	// std::cout << "TRN4CPP : sucessful call to " << __FUNCTION__ << std::endl;
}
void TRN4CPP::deallocate(const unsigned int &id)
{
	// std::unique_lock<std::mutex> lock(mutex);
	// // std::unique_lock<std::mutex> lock(mutex);
	if (!broker)
		throw std::logic_error("broker had not been initialized");
	broker->deallocate(id);
	

	// std::cout << "TRN4CPP : sucessful call to " << __FUNCTION__ << std::endl;
}
void TRN4CPP::train(const unsigned int &id, const std::string &label, const std::string &incoming, const std::string &expected)
{
	// std::unique_lock<std::mutex> lock(mutex);
	// // std::unique_lock<std::mutex> lock(mutex);
	if (!broker)
		throw std::logic_error("broker had not been initialized");
	broker->train(id, label, incoming, expected);
	// std::cout << "TRN4CPP : sucessful call to " << __FUNCTION__ << std::endl;
}
void TRN4CPP::test(const unsigned int &id, const std::string &sequence, const std::string &incoming, const std::string &expected, const unsigned int &preamble, const unsigned int &supplementary_generations)
{
	// std::unique_lock<std::mutex> lock(mutex);
	// // std::unique_lock<std::mutex> lock(mutex);
	if (!broker)
		throw std::logic_error("broker had not been initialized");
	broker->test(id, sequence, incoming, expected, preamble, supplementary_generations);
	// std::cout << "TRN4CPP : sucessful call to " << __FUNCTION__ << std::endl;
}
void TRN4CPP::declare_sequence(const unsigned int &id, const std::string &label, const std::string &tag, const std::vector<float> &sequence, const std::size_t &observations)
{
	// std::unique_lock<std::mutex> lock(mutex);
	// // std::unique_lock<std::mutex> lock(mutex);
	if (!broker)
		throw std::logic_error("broker had not been initialized");
	broker->declare_sequence(id, label, tag, sequence, observations);
	// std::cout << "TRN4CPP : sucessful call to " << __FUNCTION__ << std::endl;
}

void TRN4CPP::declare_set(const unsigned int &id, const std::string &label, const std::string &tag, const std::vector<std::string> &labels)
{
	// std::unique_lock<std::mutex> lock(mutex);
	// // std::unique_lock<std::mutex> lock(mutex);
	if (!broker)
		throw std::logic_error("broker had not been initialized");
	broker->declare_set(id, label, tag, labels);
	// std::cout << "TRN4CPP : sucessful call to " << __FUNCTION__ << std::endl;
}
void TRN4CPP::setup_states(const unsigned int &id, const std::function<void(const unsigned int &id, const std::string &phase, const std::string &label, const std::vector<float> &samples, const std::size_t &rows, const std::size_t &cols)> &functor, const bool &train, const bool &prime, const bool &generate)
{
	// std::unique_lock<std::mutex> lock(mutex);
	// // std::unique_lock<std::mutex> lock(mutex);
	if (!broker)
		throw std::logic_error("broker had not been initialized");
	broker->setup_states(id, std::bind(functor, id, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5), train, prime, generate);
	// std::cout << "TRN4CPP : sucessful call to " << __FUNCTION__ << std::endl;
}
void TRN4CPP::setup_weights(const unsigned int &id, const std::function<void(const unsigned int &id, const std::string &phase, const std::string &label, const std::vector<float> &weights, const std::size_t &rows, const std::size_t &cols)> &functor, const bool &initialize, const bool &train)
{
	// std::unique_lock<std::mutex> lock(mutex);
	// // std::unique_lock<std::mutex> lock(mutex);
	if (!broker)
		throw std::logic_error("broker had not been initialized");
	broker->setup_weights(id, std::bind(functor, id, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5), initialize, train);
	// std::cout << "TRN4CPP : sucessful call to " << __FUNCTION__ << std::endl;
}
void TRN4CPP::setup_performances(const unsigned int &id, const std::function<void(const unsigned int &id, const std::string &phase, const std::size_t &batch_size, const size_t &cycles, const float &gflops, const float &seconds)> &functor, const bool &train, const bool &prime, const bool &generate)
{
	// std::unique_lock<std::mutex> lock(mutex);
	// // std::unique_lock<std::mutex> lock(mutex);
	if (!broker)
		throw std::logic_error("broker had not been initialized");
	broker->setup_performances(id, std::bind(functor, id, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5), train, prime, generate);
	// std::cout << "TRN4CPP : sucessful call to " << __FUNCTION__ << std::endl;
}

void TRN4CPP::setup_scheduling(const unsigned int &id, const std::function<void(const unsigned int &id, const std::vector<int> &offsets, const std::vector<int> &durations)> &functor)
{
	if (!broker)
		throw std::logic_error("broker had not been initialized");
	broker->setup_scheduling(id, std::bind(functor, id, std::placeholders::_1, std::placeholders::_2));
}


void TRN4CPP::configure_begin(const unsigned int &id)
{
	// std::unique_lock<std::mutex> lock(mutex);
	// // std::unique_lock<std::mutex> lock(mutex);
	if (!broker)
		throw std::logic_error("broker had not been initialized");
	broker->configure_begin(id);
	// std::cout << "TRN4CPP : sucessful call to " << __FUNCTION__ << std::endl;
}
void TRN4CPP::configure_end(const unsigned int &id)
{
	// std::unique_lock<std::mutex> lock(mutex);
	// // std::unique_lock<std::mutex> lock(mutex);
	if (!broker)
		throw std::logic_error("broker had not been initialized");
	broker->configure_end(id);
	// std::cout << "TRN4CPP : sucessful call to " << __FUNCTION__ << std::endl;
}

void 	TRN4CPP::configure_measurement_readout_mean_square_error(const unsigned int &id, const std::size_t &batch_size, const std::function<void(const unsigned int &id, const std::vector<float> &values, const std::size_t &rows, const std::size_t &cols)> &functor)
{
	// std::unique_lock<std::mutex> lock(mutex);
	if (!broker)
		throw std::logic_error("broker had not been initialized");
	broker->configure_measurement_readout_mean_square_error(id, batch_size, std::bind(functor, id, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
}

void 	TRN4CPP::configure_measurement_readout_frechet_distance(const unsigned int &id, const std::size_t &batch_size, const std::function<void(const unsigned int &id, const std::vector<float> &values, const std::size_t &rows, const std::size_t &cols)> &functor)
{
	// std::unique_lock<std::mutex> lock(mutex);
	if (!broker)
		throw std::logic_error("broker had not been initialized");
	broker->configure_measurement_readout_frechet_distance(id, batch_size, std::bind(functor, id, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
}
void 	TRN4CPP::configure_measurement_readout_custom(const unsigned int &id, const std::size_t &batch_size, const std::function<void(const unsigned int &id, const std::vector<float> &primed, const std::vector<float> &predicted, const std::vector<float> &expected, const std::size_t &preamble, const std::size_t &pages, const std::size_t &rows, const std::size_t &cols)> &functor)
{
	// std::unique_lock<std::mutex> lock(mutex);
	if (!broker)
		throw std::logic_error("broker had not been initialized");
	broker->configure_measurement_readout_custom(id, batch_size, std::bind(functor, id, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5, std::placeholders::_6, std::placeholders::_7));
}

void 	TRN4CPP::configure_measurement_position_mean_square_error(const unsigned int &id, const std::size_t &batch_size, const std::function<void(const unsigned int &id, const std::vector<float> &values, const std::size_t &rows, const std::size_t &cols)> &functor)
{
	// std::unique_lock<std::mutex> lock(mutex);
	if (!broker)
		throw std::logic_error("broker had not been initialized");
	broker->configure_measurement_position_mean_square_error(id, batch_size, std::bind(functor, id, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
}

void 	TRN4CPP::configure_measurement_position_frechet_distance(const unsigned int &id, const std::size_t &batch_size, const std::function<void(const unsigned int &id, const std::vector<float> &values, const std::size_t &rows, const std::size_t &cols)> &functor)
{
	// std::unique_lock<std::mutex> lock(mutex);
	if (!broker)
		throw std::logic_error("broker had not been initialized");
	broker->configure_measurement_position_frechet_distance(id, batch_size, std::bind(functor, id, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
}
void 	TRN4CPP::configure_measurement_position_custom(const unsigned int &id, const std::size_t &batch_size, const std::function<void(const unsigned int &id, const std::vector<float> &primed, const std::vector<float> &predicted, const std::vector<float> &expected, const std::size_t &preamble, const std::size_t &pages, const std::size_t &rows, const std::size_t &cols)> &functor)
{
	// std::unique_lock<std::mutex> lock(mutex);
	if (!broker)
		throw std::logic_error("broker had not been initialized");
	broker->configure_measurement_position_custom(id, batch_size, std::bind(functor, id, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5, std::placeholders::_6, std::placeholders::_7 ));
}

void TRN4CPP::configure_reservoir_widrow_hoff(const unsigned int &id, const std::size_t &stimulus_size, const std::size_t &prediction_size, const std::size_t &reservoir_size, const float &leak_rate, const float &initial_state_scale, const float &learning_rate, const unsigned long &seed, const std::size_t &batch_size)
{
	// std::unique_lock<std::mutex> lock(mutex);
	// // std::unique_lock<std::mutex> lock(mutex);
	if (!broker)
		throw std::logic_error("broker had not been initialized");
	broker->configure_reservoir_widrow_hoff(id, stimulus_size, prediction_size, reservoir_size, leak_rate, initial_state_scale, learning_rate, seed, batch_size);
	// std::cout << "TRN4CPP : sucessful call to " << __FUNCTION__ << std::endl;
}

void TRN4CPP::configure_loop_copy(const unsigned int &id, const std::size_t &batch_size, const std::size_t &stimulus_size)
{
	// std::unique_lock<std::mutex> lock(mutex);
	// // std::unique_lock<std::mutex> lock(mutex);
	if (!broker)
		throw std::logic_error("broker had not been initialized");
	broker->configure_loop_copy(id, batch_size, stimulus_size);
	// std::cout << "TRN4CPP : sucessful call to " << __FUNCTION__ << std::endl;
}
void TRN4CPP::configure_loop_spatial_filter(const unsigned int &id, const std::size_t &batch_size, const std::size_t &stimulus_size,
	const std::function<void(const unsigned int &id, const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols)> &predicted_position,
	std::function<void(const unsigned int &id, const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols)> &estimated_position,
	const std::function<void(const unsigned int &id, const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols)> &predicted_stimulus,
	std::function<void(const unsigned int &id, const std::vector<float> &stimulus, const std::size_t &rows, const std::size_t &cols)> &perceived_stimulus,
	const std::size_t &rows, const std::size_t &cols, const std::pair<float, float> &x, const std::pair<float, float> &y,
	const std::vector<float> response, const float &sigma, const float &radius, const float &scale, const std::string &tag
	
	)
{
	std::unique_lock<std::mutex> lock(mutex);
	// // std::unique_lock<std::mutex> lock(mutex);
	if (!broker)
		throw std::logic_error("broker had not been initialized");

	broker->configure_loop_spatial_filter(id, batch_size, stimulus_size, 
		std::bind(predicted_position, id, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3), id_estimated_position[id],
		std::bind(predicted_stimulus, id, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3), id_perceived_stimulus[id],
		rows, cols, x, y, response, sigma, radius, scale, tag);

	
	estimated_position = [=](const unsigned int &id, const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols)
	{
		std::unique_lock<std::mutex> lock(mutex);
		id_estimated_position[id](position, rows, cols);
	};
	perceived_stimulus = [=](const unsigned int &id, const std::vector<float> &stimulus, const std::size_t &rows, const std::size_t &cols)
	{
		std::unique_lock<std::mutex> lock(mutex);
		id_perceived_stimulus[id](stimulus, rows, cols);
	};
	//perceived_stimulus = std::bind(_perceived_stimulus, id, std::placeholders::_1);
	// std::cout << "TRN4CPP : sucessful call to " << __FUNCTION__ << std::endl;
}
void TRN4CPP::configure_loop_custom(const unsigned int &id, const std::size_t &batch_size, const std::size_t &stimulus_size,
	const std::function<void(const unsigned int &id, const std::vector<float> &prediction, const std::size_t &rows, const std::size_t &cols)> &request,
	std::function<void(const unsigned int &id, const std::vector<float> &stimulus, const std::size_t &rows, const std::size_t &cols)> &reply
	)
{
	 std::unique_lock<std::mutex> lock(mutex);
	// // std::unique_lock<std::mutex> lock(mutex);
	if (!broker)
		throw std::logic_error("broker had not been initialized");
	broker->configure_loop_custom(id, batch_size, stimulus_size, std::bind(request, id, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3), id_estimated_position[id]);
	reply = [=](const unsigned int &id, const std::vector<float> &stimulus, const std::size_t &rows, const std::size_t &cols)
	{
		std::unique_lock<std::mutex> lock(mutex);
		id_perceived_stimulus[id](stimulus, rows, cols);
	};
	
	// std::cout << "TRN4CPP : sucessful call to " << __FUNCTION__ << std::endl;
}

void TRN4CPP::configure_scheduler_tiled(const unsigned int &id, const unsigned int &epochs)
{
	// std::unique_lock<std::mutex> lock(mutex);
	// // std::unique_lock<std::mutex> lock(mutex);
	if (!broker)
		throw std::logic_error("broker had not been initialized");
	broker->configure_scheduler_tiled(id, epochs);
	// std::cout << "TRN4CPP : sucessful call to " << __FUNCTION__ << std::endl;
}
void TRN4CPP::configure_scheduler_snippets(const unsigned int &id, const unsigned int &snippets_size, const unsigned int &time_budget, const std::string &tag)
{
	// std::unique_lock<std::mutex> lock(mutex);
	// // std::unique_lock<std::mutex> lock(mutex);
	if (!broker)
		throw std::logic_error("broker had not been initialized");
	broker->configure_scheduler_snippets(id, snippets_size, time_budget, tag);
	// std::cout << "TRN4CPP : sucessful call to " << __FUNCTION__ << std::endl;
}


void TRN4CPP::configure_scheduler_custom(const unsigned int &id, 
	const std::function<void(const unsigned int &id, const std::vector<float> &elements, const std::size_t &rows, const std::size_t &cols, const std::vector<int> &offsets, const std::vector<int> &durations)> &request,
	std::function<void(const unsigned int &id, const std::vector<int> &offsets, const std::vector<int> &durations)> &reply, const std::string &tag)
{
	//std::unique_lock<std::mutex> lock(mutex);
	// // std::unique_lock<std::mutex> lock(mutex);
	if (!broker)
		throw std::logic_error("broker had not been initialized");

	broker->configure_scheduler_custom(id, std::bind(request, id, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5), id_scheduler[id], tag);
	reply = [=](const unsigned int &id, const std::vector<int> &offsets, const std::vector<int> &durations)
	{
		std::unique_lock<std::mutex> lock(mutex);
		id_scheduler[id](offsets, durations);
	};
	// std::cout << "TRN4CPP : sucessful call to " << __FUNCTION__ << std::endl;
}

void TRN4CPP::configure_mutator_shuffle(const unsigned int &id)
{
	if (!broker)
		throw std::logic_error("broker had not been initialized");
	broker->configure_mutator_shuffle(id);
}
void TRN4CPP::configure_mutator_reverse(const unsigned int &id, const float &rate, const std::size_t &size)
{
	if (!broker)
		throw std::logic_error("broker had not been initialized");
	broker->configure_mutator_reverse(id, rate, size);
}
void TRN4CPP::configure_mutator_punch(const unsigned int &id, const float &rate, const std::size_t &size, const std::size_t &number)
{
	if (!broker)
		throw std::logic_error("broker had not been initialized");
	broker->configure_mutator_punch(id, rate, size, number);
}
void TRN4CPP::configure_mutator_custom(const unsigned int &id, 
	const std::function<void(const unsigned int &id, const std::vector<int> &offsets, const std::vector<int> &durations)> &request,
	std::function<void(const unsigned int &id, const std::vector<int> &offsets, const std::vector<int> &durations)> &reply
)
{
	if (!broker)
		throw std::logic_error("broker had not been initialized");

	broker->configure_mutator_custom(id, std::bind(request, id, std::placeholders::_1, std::placeholders::_2), id_mutator[id]);
	reply = [=](const unsigned int &id, const std::vector<int> &offsets, const std::vector<int> &durations)
	{
		std::unique_lock<std::mutex> lock(mutex);
		id_mutator[id](offsets, durations);
	};
}


void TRN4CPP::configure_readout_uniform(const unsigned int &id, const float &a, const float &b, const float &sparsity)
{
	// std::unique_lock<std::mutex> lock(mutex);
	// // std::unique_lock<std::mutex> lock(mutex);
	if (!broker)
		throw std::logic_error("broker had not been initialized");
	broker->configure_readout_uniform(id, a, b, sparsity);
	// std::cout << "TRN4CPP : sucessful call to " << __FUNCTION__ << std::endl;
}
void TRN4CPP::configure_readout_gaussian(const unsigned int &id, const float &mu, const float &sigma)
{
	// std::unique_lock<std::mutex> lock(mutex);
	// // std::unique_lock<std::mutex> lock(mutex);
	if (!broker)
		throw std::logic_error("broker had not been initialized");
	broker->configure_readout_gaussian(id, mu, sigma);
	// std::cout << "TRN4CPP : sucessful call to " << __FUNCTION__ << std::endl;
}
void TRN4CPP::configure_readout_custom(const unsigned int &id, const std::function<void(const unsigned int &id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &request,
	std::function<void(const unsigned int &id, const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &reply)
{
	std::unique_lock<std::mutex> lock(mutex);
	// // std::unique_lock<std::mutex> lock(mutex);
	if (!broker)
		throw std::logic_error("broker had not been initialized");
	broker->configure_readout_custom(id, std::bind(request, id, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4), id_readout_weights[id]);
	reply = [=](const unsigned int &id, const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)
	{
		std::unique_lock<std::mutex> lock(mutex);
		id_readout_weights[id](weights, matrices, rows, cols);
	};
	// std::cout << "TRN4CPP : sucessful call to " << __FUNCTION__ << std::endl;
}

void TRN4CPP::configure_feedback_uniform(const unsigned int &id, const float &a, const float &b, const float &sparsity)
{
	// std::unique_lock<std::mutex> lock(mutex);
	// // std::unique_lock<std::mutex> lock(mutex);
	if (!broker)
		throw std::logic_error("broker had not been initialized");
	broker->configure_feedback_uniform(id, a, b, sparsity);
	// std::cout << "TRN4CPP : sucessful call to " << __FUNCTION__ << std::endl;
}
void TRN4CPP::configure_feedback_gaussian(const unsigned int &id, const float &mu, const float &sigma)
{
	// std::unique_lock<std::mutex> lock(mutex);
	// // std::unique_lock<std::mutex> lock(mutex);
	if (!broker)
		throw std::logic_error("broker had not been initialized");
	broker->configure_feedback_gaussian(id, mu, sigma);
	// std::cout << "TRN4CPP : sucessful call to " << __FUNCTION__ << std::endl;
}
void TRN4CPP::configure_feedback_custom(const unsigned int &id, const std::function<void(const unsigned int &id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &request,
	std::function<void(const unsigned int &id, const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &reply)
{
	std::unique_lock<std::mutex> lock(mutex);
	// // std::unique_lock<std::mutex> lock(mutex);
	if (!broker)
		throw std::logic_error("broker had not been initialized");
	broker->configure_feedback_custom(id, std::bind(request, id, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4), id_feedback_weights[id]);
	reply = [=](const unsigned int &id, const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)
	{
		std::unique_lock<std::mutex> lock(mutex);
		id_feedback_weights[id](weights, matrices, rows, cols);
	};
	// std::cout << "TRN4CPP : sucessful call to " << __FUNCTION__ << std::endl;
}

void TRN4CPP::configure_recurrent_uniform(const unsigned int &id, const float &a, const float &b, const float &sparsity)
{
	// std::unique_lock<std::mutex> lock(mutex);
	// // std::unique_lock<std::mutex> lock(mutex);
	if (!broker)
		throw std::logic_error("broker had not been initialized");
	broker->configure_recurrent_uniform(id, a, b, sparsity);
	// std::cout << "TRN4CPP : sucessful call to " << __FUNCTION__ << std::endl;
}
void TRN4CPP::configure_recurrent_gaussian(const unsigned int &id, const float &mu, const float &sigma)
{
	// std::unique_lock<std::mutex> lock(mutex);
	// // std::unique_lock<std::mutex> lock(mutex);
	if (!broker)
		throw std::logic_error("broker had not been initialized");
	broker->configure_recurrent_gaussian(id, mu, sigma);
	// std::cout << "TRN4CPP : sucessful call to " << __FUNCTION__ << std::endl;
}
void TRN4CPP::configure_recurrent_custom(const unsigned int &id, const std::function<void(const unsigned int &id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &request,
	std::function<void(const unsigned int &id, const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &reply)
{
	 std::unique_lock<std::mutex> lock(mutex);
	// // std::unique_lock<std::mutex> lock(mutex);
	if (!broker)
		throw std::logic_error("broker had not been initialized");
	broker->configure_recurrent_custom(id, std::bind(request, id, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4), id_recurrent_weights[id]);
	reply = [=](const unsigned int &id, const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)
	{
		std::unique_lock<std::mutex> lock(mutex);
		id_recurrent_weights[id](weights, matrices, rows, cols);
	};
	// std::cout << "TRN4CPP : sucessful call to " << __FUNCTION__ << std::endl;
}

void TRN4CPP::configure_feedforward_uniform(const unsigned int &id, const float &a, const float &b, const float &sparsity)
{
	// std::unique_lock<std::mutex> lock(mutex);
	// // std::unique_lock<std::mutex> lock(mutex);
	if (!broker)
		throw std::logic_error("broker had not been initialized");
	broker->configure_feedforward_uniform(id, a, b, sparsity);
	// std::cout << "TRN4CPP : sucessful call to " << __FUNCTION__ << std::endl;
}
void TRN4CPP::configure_feedforward_gaussian(const unsigned int &id, const float &mu, const float &sigma)
{
	// std::unique_lock<std::mutex> lock(mutex);
	// // std::unique_lock<std::mutex> lock(mutex);
	if (!broker)
		throw std::logic_error("broker had not been initialized");
	broker->configure_feedforward_gaussian(id, mu, sigma);
	// std::cout << "TRN4CPP : sucessful call to " << __FUNCTION__ << std::endl;
}
void TRN4CPP::configure_feedforward_custom(const unsigned int &id, const std::function<void(const unsigned int &id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &request,
	std::function<void(const unsigned int &id, const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &reply)
{
	 std::unique_lock<std::mutex> lock(mutex);
	// // std::unique_lock<std::mutex> lock(mutex);
	if (!broker)
		throw std::logic_error("broker had not been initialized");
	broker->configure_feedforward_custom(id, std::bind(request, id, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4), id_feedforward_weights[id]);
	reply = [=](const unsigned int &id, const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)
	{
		std::unique_lock<std::mutex> lock(mutex);
		id_feedforward_weights[id](weights, matrices, rows, cols);
	};
	// std::cout << "TRN4CPP : sucessful call to " << __FUNCTION__ << std::endl;
}