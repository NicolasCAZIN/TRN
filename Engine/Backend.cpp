#include "stdafx.h"
#include "Backend_impl.h"
#include "Broker_impl.h"

TRN::Engine::Backend::Backend(const std::shared_ptr<TRN::Engine::Communicator> &to_frontend, const std::shared_ptr<TRN::Engine::Communicator> &to_workers) :
	TRN::Engine::Broker(to_workers),
	handle(std::make_unique<Handle>())
{
	handle->to_frontend = to_frontend;
}
TRN::Engine::Backend::~Backend()
{
	handle.reset();
}


void TRN::Engine::Backend::callback_ack(const unsigned long long &id, const std::size_t &number, const bool &success, const std::string &cause)
{
	/*TRN::Engine::Message<TRN::Engine::Tag::ACK> message;

	message.id = id;
	message.number = number;
	message.success = success;
	message.cause = cause;

	TRN::Engine::Broker::handle->communicator->send(message, 0);*/
}
void TRN::Engine::Backend::callback_configured(const unsigned long long &id)
{
	TRN::Engine::Message<TRN::Engine::Tag::CONFIGURED> message;

	message.id = id;

	handle->to_frontend->send(message, 0);

}
void TRN::Engine::Backend::callback_processor(const int &rank, const std::string &host, const unsigned int &index, const std::string &name)
{
	TRN::Engine::Message<TRN::Engine::Tag::WORKER> message;

	message.host = host;
	message.index = index;
	message.name = name;
	message.rank = rank;

	handle->to_frontend->send(message, 0);
}
void TRN::Engine::Backend::callback_allocated(const unsigned long long &id, const int &rank)
{
	TRN::Engine::Message<TRN::Engine::Tag::ALLOCATED> message;

	message.id = id;

	handle->to_frontend->send(message, 0);
}
void TRN::Engine::Backend::callback_deallocated(const unsigned long long &id, const int &rank)
{
	TRN::Engine::Message<TRN::Engine::Tag::DEALLOCATED> message;

	message.id = id;

	handle->to_frontend->send(message, 0);
}
void TRN::Engine::Backend::callback_quit(const int &rank)
{
	TRN::Engine::Message<TRN::Engine::Tag::QUIT> message;

	message.rank = rank;

	handle->to_frontend->send(message, 0);
}

void TRN::Engine::Backend::callback_completed()
{
	std::cout << "SIMULATIONS COMPLETED" << std::endl;
}
void TRN::Engine::Backend::callback_trained(const unsigned long long &id)
{
	TRN::Engine::Message<TRN::Engine::Tag::TRAINED> message;

	message.id = id;
	handle->to_frontend->send(message, 0);
}
void TRN::Engine::Backend::callback_primed(const unsigned long long &id)
{
	TRN::Engine::Message<TRN::Engine::Tag::PRIMED> message;

	message.id = id;
	handle->to_frontend->send(message, 0);
}
void TRN::Engine::Backend::callback_tested(const unsigned long long &id)
{
	TRN::Engine::Message<TRN::Engine::Tag::TESTED> message;

	message.id = id;
	handle->to_frontend->send(message, 0);
}
void TRN::Engine::Backend::callback_error(const std::string &message)
{
	std::cerr << "ERROR : " << message << std::endl;
}
void TRN::Engine::Backend::callback_information(const std::string &message)
{
	std::cout << "INFORMATION : " << message << std::endl;
}
void TRN::Engine::Backend::callback_warning(const std::string &message)
{
	std::cerr << "WARNING : " << message << std::endl;
}
void TRN::Engine::Backend::callback_measurement_readout_mean_square_error(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols)
{
	TRN::Engine::Message<TRN::Engine::MEASUREMENT_READOUT_MEAN_SQUARE_ERROR> message;

	message.id = id;
	message.trial = trial;
	message.evaluation = evaluation;
	message.elements = values;

	message.rows = rows;
	message.cols = cols;

	handle->to_frontend->send(message, 0);
}
void TRN::Engine::Backend::callback_measurement_readout_frechet_distance(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols)
{
	TRN::Engine::Message<TRN::Engine::MEASUREMENT_READOUT_FRECHET_DISTANCE> message;

	message.id = id;
	message.trial = trial;
	message.evaluation = evaluation;
	message.elements = values;
	message.rows = rows;
	message.cols = cols;

	handle->to_frontend->send(message, 0);
}
void TRN::Engine::Backend::callback_measurement_readout_custom(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &primed, const std::vector<float> &predicted, const std::vector<float> &expected, const std::size_t &preamble, const std::size_t &pages, const std::size_t &rows, const  std::size_t &cols)
{
	TRN::Engine::Message<TRN::Engine::MEASUREMENT_READOUT_CUSTOM> message;

	message.id = id;
	message.trial = trial;
	message.evaluation = evaluation;
	message.primed = primed;
	message.elements = predicted;
	message.expected = expected;
	message.preamble = preamble;
	message.matrices = pages;
	message.rows = rows;
	message.cols = cols;

	handle->to_frontend->send(message, 0);
}
void TRN::Engine::Backend::callback_measurement_position_mean_square_error(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols)
{
	TRN::Engine::Message<TRN::Engine::MEASUREMENT_POSITION_MEAN_SQUARE_ERROR> message;

	message.id = id;
	message.trial = trial;
	message.evaluation = evaluation;
	message.elements = values;
	message.rows = rows;
	message.cols = cols;

	handle->to_frontend->send(message, 0);
}
void TRN::Engine::Backend::callback_measurement_position_frechet_distance(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols)
{
	TRN::Engine::Message<TRN::Engine::MEASUREMENT_POSITION_FRECHET_DISTANCE> message;

	message.id = id;
	message.trial = trial;
	message.evaluation = evaluation;
	message.elements = values;
	message.rows = rows;
	message.cols = cols;

	handle->to_frontend->send(message, 0);
}
void TRN::Engine::Backend::callback_measurement_position_custom(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &primed, const std::vector<float> &predicted, const std::vector<float> &expected, const std::size_t &preamble, const std::size_t &pages, const std::size_t &rows, const  std::size_t &cols)
{
	TRN::Engine::Message<TRN::Engine::MEASUREMENT_POSITION_CUSTOM> message;

	message.id = id;
	message.trial = trial;
	message.evaluation = evaluation;
	message.elements = predicted;
	message.expected = expected;
	message.primed = primed;
	message.matrices = pages;
	message.rows = rows;
	message.cols = cols;
	message.preamble = preamble;
	handle->to_frontend->send(message, 0);
}
void TRN::Engine::Backend::callback_performances(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::string &phase, const float &cycles_per_second, const float &gflops_per_second)
{
	TRN::Engine::Message<TRN::Engine::PERFORMANCES> message;

	message.id = id;
	message.phase = phase;

	message.trial = trial;
	message.evaluation = evaluation;
	message.cycles_per_second = cycles_per_second;
	message.gflops_per_second = gflops_per_second;
	handle->to_frontend->send(message, 0);
}
void TRN::Engine::Backend::callback_states(const unsigned long long &id, const std::string &phase, const std::string &label, const std::size_t &batch, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &samples, const std::size_t &rows, const std::size_t &cols)
{
	TRN::Engine::Message<TRN::Engine::STATES> message;

	message.id = id;
	message.label = label;
	message.phase = phase;
	message.batch = batch;
	message.trial = trial;
	message.evaluation = evaluation;
	message.elements = samples;
	message.rows = rows;
	message.cols = cols;

	handle->to_frontend->send(message, 0);
}
void TRN::Engine::Backend::callback_weights(const unsigned long long &id, const std::string &phase, const std::string &label, const std::size_t &batch, const std::size_t &trial, const std::vector<float> &samples, const std::size_t &rows, const std::size_t &cols)
{
	TRN::Engine::Message<TRN::Engine::WEIGHTS> message;

	message.id = id;
	message.phase = phase;
	message.label = label;
	message.trial = trial;
	message.batch = batch;
	message.elements = samples;
	message.rows = rows;
	message.cols = cols;

	handle->to_frontend->send(message, 0);
}
void TRN::Engine::Backend::callback_position(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols)
{
	TRN::Engine::Message<TRN::Engine::POSITION> message;
	message.id = id;
	message.trial = trial;
	message.evaluation = evaluation;
	message.elements = position;
	message.rows = rows;
	message.cols = cols;
	handle->to_frontend->send(message, 0);
}
void TRN::Engine::Backend::callback_stimulus(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &stimulus, const std::size_t &rows, const std::size_t &cols)
{
	TRN::Engine::Message<TRN::Engine::STIMULUS> message;
	message.id = id;
	message.trial = trial;
	message.evaluation = evaluation;
	message.elements = stimulus;
	message.rows = rows;
	message.cols = cols;
	handle->to_frontend->send(message, 0);
}
void TRN::Engine::Backend::callback_mutator(const unsigned long long &id, const unsigned long &seed, const std::size_t &trial, const std::vector<int> &offsets, const std::vector<int> &durations)
{
	TRN::Engine::Message<TRN::Engine::Tag::MUTATOR_CUSTOM> message;

	message.trial = trial;
	message.id = id;
	message.offsets = offsets;
	message.durations = durations;
	message.seed = seed;
	handle->to_frontend->send(message, 0);
}
void TRN::Engine::Backend::callback_scheduler(const unsigned long long &id, const unsigned long &seed, const std::size_t &trial, const std::vector<float> &elements, const std::size_t &rows, const std::size_t &cols, const std::vector<int> &offsets, const std::vector<int> &durations)
{
	TRN::Engine::Message<TRN::Engine::Tag::SCHEDULER_CUSTOM> message;
	message.trial = trial;
	message.seed = seed;
	message.id = id;
	message.elements = elements;
	message.rows = rows;
	message.cols = cols;
	message.offsets = offsets;
	message.durations = durations;
	handle->to_frontend->send(message, 0);
}
void TRN::Engine::Backend::callback_scheduling(const unsigned long long &id, const std::size_t &trial, const std::vector<int> &offsets, const std::vector<int> &durations)
{
	TRN::Engine::Message<TRN::Engine::SCHEDULING> scheduling;

	scheduling.id = id;
	scheduling.trial = trial;
	scheduling.offsets = offsets;
	scheduling.durations = durations;
	scheduling.is_from_mutator = false;

	handle->to_frontend->send(scheduling, 0);
}
void TRN::Engine::Backend::callback_feedforward(const unsigned long long &id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const  std::size_t &cols)
{
	TRN::Engine::Message<TRN::Engine::Tag::FEEDFORWARD_DIMENSIONS> message;

	message.id = id;
	message.matrices = matrices;
	message.rows = rows;
	message.cols = cols;
	message.seed = seed;
	handle->to_frontend->send(message, 0);
}
void TRN::Engine::Backend::callback_feedback(const unsigned long long &id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const  std::size_t &cols)
{
	TRN::Engine::Message<TRN::Engine::Tag::FEEDBACK_DIMENSIONS> message;

	message.id = id;
	message.matrices = matrices;
	message.rows = rows;
	message.cols = cols;
	message.seed = seed;
	handle->to_frontend->send(message, 0);
}
void TRN::Engine::Backend::callback_readout(const unsigned long long &id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const  std::size_t &cols)
{
	TRN::Engine::Message<TRN::Engine::Tag::READOUT_DIMENSIONS> message;

	message.id = id;
	message.matrices = matrices;
	message.rows = rows;
	message.cols = cols;
	message.seed = seed;
	handle->to_frontend->send(message, 0);
}
void TRN::Engine::Backend::callback_recurrent(const unsigned long long &id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)
{
	TRN::Engine::Message<TRN::Engine::Tag::RECURRENT_DIMENSIONS> message;

	message.id = id;
	message.matrices = matrices;
	message.rows = rows;
	message.cols = cols;
	message.seed = seed;
	handle->to_frontend->send(message, 0);
}

std::shared_ptr<TRN::Engine::Backend> TRN::Engine::Backend::create(const std::shared_ptr<TRN::Engine::Communicator> &to_frontend, const std::shared_ptr<TRN::Engine::Communicator> &to_workers)
{
	return std::make_shared<TRN::Engine::Backend>(to_frontend, to_workers);
}