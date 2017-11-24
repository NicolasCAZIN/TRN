#include "stdafx.h"
#include "Dispatcher_impl.h"
#include "Broker_impl.h"

TRN::Engine::Dispatcher::Dispatcher(const std::shared_ptr<TRN::Engine::Communicator> &to_workers) :
	TRN::Engine::Broker(to_workers),
	handle(std::make_unique<Handle>())
{
}
TRN::Engine::Dispatcher::~Dispatcher()
{
	handle.reset();
}





static void local_id(const unsigned long long &global_id, unsigned long long &local_id, unsigned short &frontend_number)
{
	unsigned short condition_number;
	unsigned int simulation_number;

	TRN::Engine::decode(global_id, frontend_number, condition_number, simulation_number);
	TRN::Engine::encode(0, condition_number, simulation_number, local_id);
}

void TRN::Engine::Dispatcher::register_frontend(const unsigned short &frontend, const std::shared_ptr<TRN::Engine::Communicator> &communicator)
{
	if (handle->to_frontend.find(frontend) != handle->to_frontend.end())
		throw std::invalid_argument("Communicator for frontend #" + std::to_string(frontend) + " is already registered");
	handle->to_frontend[frontend] = communicator;
}
void TRN::Engine::Dispatcher::unregister_frontend(const unsigned short &frontend)
{
	if (handle->to_frontend.find(frontend) == handle->to_frontend.end())
		throw std::invalid_argument("Communicator for frontend #" + std::to_string(frontend) + " is not registered");
	handle->to_frontend.erase(frontend);
}

void TRN::Engine::Dispatcher::callback_ack(const unsigned long long &id, const std::size_t &counter, const bool &success, const std::string &cause)
{
	/*TRN::Engine::Message<TRN::Engine::Tag::ACK> message;

	message.id = id;
	message.number = number;
	message.success = success;
	message.cause = cause;

	TRN::Engine::Broker::handle->communicator->send(message, 0);*/
}

template <typename Message>
static void send_to_frontend(std::map<unsigned short, std::shared_ptr<TRN::Engine::Communicator>> &to_frontend, Message &message)
{
	for (auto p : to_frontend)
	{
		to_frontend[p.first]->send(message, 0);
	}
}

template <typename Message>
static void send_to_frontend(std::map<unsigned short, std::shared_ptr<TRN::Engine::Communicator>> &to_frontend,  Message &message, const unsigned long long &id)
{
	unsigned short frontend_number;
	local_id(id, message.id, frontend_number);
	if (to_frontend.find(frontend_number) == to_frontend.end())
		throw std::invalid_argument("Frontend #" + std::to_string(frontend_number) + "is not registered");
	to_frontend[frontend_number]->send(message, 0);
}

void TRN::Engine::Dispatcher::callback_configured(const unsigned long long &id)
{
	TRN::Engine::Message<TRN::Engine::Tag::CONFIGURED> message;

	send_to_frontend(handle->to_frontend,message, id);
}
void TRN::Engine::Dispatcher::callback_processor(const int &rank, const std::string &host, const unsigned int &index, const std::string &name)
{
	TRN::Engine::Message<TRN::Engine::Tag::WORKER> message;

	message.host = host;
	message.index = index;
	message.name = name;
	message.rank = rank;

	send_to_frontend(handle->to_frontend, message);
}
void TRN::Engine::Dispatcher::callback_allocated(const unsigned long long &id, const int &rank)
{
	TRN::Engine::Message<TRN::Engine::Tag::ALLOCATED> message;

	send_to_frontend(handle->to_frontend,message, id);
}
void TRN::Engine::Dispatcher::callback_deallocated(const unsigned long long &id, const int &rank)
{
	TRN::Engine::Message<TRN::Engine::Tag::DEALLOCATED> message;

	send_to_frontend(handle->to_frontend,message, id);
}
void TRN::Engine::Dispatcher::callback_exit(const int &rank, const bool &terminated)
{
	TRN::Engine::Message<TRN::Engine::Tag::EXIT> message;

	message.rank = rank;
	message.terminated = terminated;
	
	send_to_frontend(handle->to_frontend, message);
}

void TRN::Engine::Dispatcher::callback_completed()
{
	std::cout << "SIMULATIONS COMPLETED" << std::endl;
}
void TRN::Engine::Dispatcher::callback_trained(const unsigned long long &id)
{
	TRN::Engine::Message<TRN::Engine::Tag::TRAINED> message;

	send_to_frontend(handle->to_frontend,message, id);
}
void TRN::Engine::Dispatcher::callback_primed(const unsigned long long &id)
{
	TRN::Engine::Message<TRN::Engine::Tag::PRIMED> message;

	send_to_frontend(handle->to_frontend,message, id);
}
void TRN::Engine::Dispatcher::callback_tested(const unsigned long long &id)
{
	TRN::Engine::Message<TRN::Engine::Tag::TESTED> message;

	send_to_frontend(handle->to_frontend,message, id);
}
void TRN::Engine::Dispatcher::callback_error(const std::string &message)
{
	std::cerr << "ERROR : " << message << std::endl;
}
void TRN::Engine::Dispatcher::callback_information(const std::string &message)
{
	std::cout << "INFORMATION : " << message << std::endl;
}
void TRN::Engine::Dispatcher::callback_warning(const std::string &message)
{
	std::cerr << "WARNING : " << message << std::endl;
}
void TRN::Engine::Dispatcher::callback_measurement_readout_mean_square_error(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols)
{
	TRN::Engine::Message<TRN::Engine::MEASUREMENT_READOUT_MEAN_SQUARE_ERROR> message;

	message.trial = trial;
	message.evaluation = evaluation;
	message.elements = values;

	message.rows = rows;
	message.cols = cols;

	send_to_frontend(handle->to_frontend,message, id);
}
void TRN::Engine::Dispatcher::callback_measurement_readout_frechet_distance(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols)
{
	TRN::Engine::Message<TRN::Engine::MEASUREMENT_READOUT_FRECHET_DISTANCE> message;

	message.trial = trial;
	message.evaluation = evaluation;
	message.elements = values;
	message.rows = rows;
	message.cols = cols;

	send_to_frontend(handle->to_frontend,message, id);
}
void TRN::Engine::Dispatcher::callback_measurement_readout_custom(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &primed, const std::vector<float> &predicted, const std::vector<float> &expected, const std::size_t &preamble, const std::size_t &pages, const std::size_t &rows, const  std::size_t &cols)
{
	TRN::Engine::Message<TRN::Engine::MEASUREMENT_READOUT_CUSTOM> message;

	message.trial = trial;
	message.evaluation = evaluation;
	message.primed = primed;
	message.elements = predicted;
	message.expected = expected;
	message.preamble = preamble;
	message.matrices = pages;
	message.rows = rows;
	message.cols = cols;

	send_to_frontend(handle->to_frontend,message, id);
}
void TRN::Engine::Dispatcher::callback_measurement_position_mean_square_error(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols)
{
	TRN::Engine::Message<TRN::Engine::MEASUREMENT_POSITION_MEAN_SQUARE_ERROR> message;

	message.trial = trial;
	message.evaluation = evaluation;
	message.elements = values;
	message.rows = rows;
	message.cols = cols;

	send_to_frontend(handle->to_frontend,message, id);
}
void TRN::Engine::Dispatcher::callback_measurement_position_frechet_distance(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols)
{
	TRN::Engine::Message<TRN::Engine::MEASUREMENT_POSITION_FRECHET_DISTANCE> message;

	message.trial = trial;
	message.evaluation = evaluation;
	message.elements = values;
	message.rows = rows;
	message.cols = cols;

	send_to_frontend(handle->to_frontend,message, id);
}
void TRN::Engine::Dispatcher::callback_measurement_position_custom(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &primed, const std::vector<float> &predicted, const std::vector<float> &expected, const std::size_t &preamble, const std::size_t &pages, const std::size_t &rows, const  std::size_t &cols)
{
	TRN::Engine::Message<TRN::Engine::MEASUREMENT_POSITION_CUSTOM> message;

	message.trial = trial;
	message.evaluation = evaluation;
	message.elements = predicted;
	message.expected = expected;
	message.primed = primed;
	message.matrices = pages;
	message.rows = rows;
	message.cols = cols;
	message.preamble = preamble;
	send_to_frontend(handle->to_frontend,message, id);
}
void TRN::Engine::Dispatcher::callback_performances(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::string &phase, const float &cycles_per_second, const float &gflops_per_second)
{
	TRN::Engine::Message<TRN::Engine::PERFORMANCES> message;

	message.phase = phase;

	message.trial = trial;
	message.evaluation = evaluation;
	message.cycles_per_second = cycles_per_second;
	message.gflops_per_second = gflops_per_second;
	send_to_frontend(handle->to_frontend,message, id);
}
void TRN::Engine::Dispatcher::callback_states(const unsigned long long &id, const std::string &phase, const std::string &label, const std::size_t &batch, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &samples, const std::size_t &rows, const std::size_t &cols)
{
	TRN::Engine::Message<TRN::Engine::STATES> message;

	message.label = label;
	message.phase = phase;
	message.batch = batch;
	message.trial = trial;
	message.evaluation = evaluation;
	message.elements = samples;
	message.rows = rows;
	message.cols = cols;

	send_to_frontend(handle->to_frontend,message, id);
}
void TRN::Engine::Dispatcher::callback_weights(const unsigned long long &id, const std::string &phase, const std::string &label, const std::size_t &batch, const std::size_t &trial, const std::vector<float> &samples, const std::size_t &rows, const std::size_t &cols)
{
	TRN::Engine::Message<TRN::Engine::WEIGHTS> message;

	message.phase = phase;
	message.label = label;
	message.trial = trial;
	message.batch = batch;
	message.elements = samples;
	message.rows = rows;
	message.cols = cols;

	send_to_frontend(handle->to_frontend,message, id);
}
void TRN::Engine::Dispatcher::callback_position(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols)
{
	TRN::Engine::Message<TRN::Engine::POSITION> message;

	message.trial = trial;
	message.evaluation = evaluation;
	message.elements = position;
	message.rows = rows;
	message.cols = cols;
	send_to_frontend(handle->to_frontend,message, id);
}
void TRN::Engine::Dispatcher::callback_stimulus(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &stimulus, const std::size_t &rows, const std::size_t &cols)
{
	TRN::Engine::Message<TRN::Engine::STIMULUS> message;

	message.trial = trial;
	message.evaluation = evaluation;
	message.elements = stimulus;
	message.rows = rows;
	message.cols = cols;
	send_to_frontend(handle->to_frontend,message, id);
}
void TRN::Engine::Dispatcher::callback_mutator(const unsigned long long &id, const unsigned long &seed, const std::size_t &trial, const std::vector<int> &offsets, const std::vector<int> &durations)
{
	TRN::Engine::Message<TRN::Engine::Tag::MUTATOR_CUSTOM> message;

	message.trial = trial;
	message.offsets = offsets;
	message.durations = durations;
	message.seed = seed;
	send_to_frontend(handle->to_frontend,message, id);
}
void TRN::Engine::Dispatcher::callback_scheduler(const unsigned long long &id, const unsigned long &seed, const std::size_t &trial, const std::vector<float> &elements, const std::size_t &rows, const std::size_t &cols, const std::vector<int> &offsets, const std::vector<int> &durations)
{
	TRN::Engine::Message<TRN::Engine::Tag::SCHEDULER_CUSTOM> message;
	message.trial = trial;
	message.seed = seed;
	message.elements = elements;
	message.rows = rows;
	message.cols = cols;
	message.offsets = offsets;
	message.durations = durations;
	send_to_frontend(handle->to_frontend,message, id);
}
void TRN::Engine::Dispatcher::callback_scheduling(const unsigned long long &id, const std::size_t &trial, const std::vector<int> &offsets, const std::vector<int> &durations)
{
	TRN::Engine::Message<TRN::Engine::SCHEDULING> message;

	message.trial = trial;
	message.offsets = offsets;
	message.durations = durations;
	message.is_from_mutator = false;
	send_to_frontend(handle->to_frontend,message, id);
}
void TRN::Engine::Dispatcher::callback_feedforward(const unsigned long long &id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const  std::size_t &cols)
{
	TRN::Engine::Message<TRN::Engine::Tag::FEEDFORWARD_DIMENSIONS> message;

	message.matrices = matrices;
	message.rows = rows;
	message.cols = cols;
	message.seed = seed;
	send_to_frontend(handle->to_frontend,message, id);
}
void TRN::Engine::Dispatcher::callback_feedback(const unsigned long long &id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const  std::size_t &cols)
{
	TRN::Engine::Message<TRN::Engine::Tag::FEEDBACK_DIMENSIONS> message;

	message.matrices = matrices;
	message.rows = rows;
	message.cols = cols;
	message.seed = seed;
	send_to_frontend(handle->to_frontend,message, id);
}
void TRN::Engine::Dispatcher::callback_readout(const unsigned long long &id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const  std::size_t &cols)
{
	TRN::Engine::Message<TRN::Engine::Tag::READOUT_DIMENSIONS> message;

	message.matrices = matrices;
	message.rows = rows;
	message.cols = cols;
	message.seed = seed;
	send_to_frontend(handle->to_frontend,message, id);
}
void TRN::Engine::Dispatcher::callback_recurrent(const unsigned long long &id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)
{
	TRN::Engine::Message<TRN::Engine::Tag::RECURRENT_DIMENSIONS> message;

	message.matrices = matrices;
	message.rows = rows;
	message.cols = cols;
	message.seed = seed;
	send_to_frontend(handle->to_frontend,message, id);
}


std::shared_ptr<TRN::Engine::Dispatcher> TRN::Engine::Dispatcher::create(const std::shared_ptr<TRN::Engine::Communicator> &to_workers)
{
	return std::make_shared<TRN::Engine::Dispatcher>(to_workers);
}