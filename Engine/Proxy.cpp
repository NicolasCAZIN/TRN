#include "stdafx.h"
#include "Proxy_impl.h"
#include "Node_impl.h"
#include "Broker_impl.h"

#include "NonBlocking.h"

TRN::Engine::Proxy::Proxy(const std::shared_ptr<TRN::Engine::Communicator> &frontend_proxy, const std::shared_ptr<TRN::Engine::Communicator> &proxy_workers, const std::shared_ptr<TRN::Helper::Visitor<TRN::Engine::Proxy>> &visitor) :
	TRN::Engine::Node(frontend_proxy, -1),
	TRN::Engine::Broker(proxy_workers, TRN::Engine::NonBlocking::create()),
	handle(std::make_unique<Handle>())
{
	handle->frontend_proxy = frontend_proxy;
	handle->visitor = visitor;

}

TRN::Engine::Proxy::~Proxy()
{
	handle.reset();
}
void TRN::Engine::Proxy::initialize()
{
	TRN::Engine::Broker::initialize();

}
void TRN::Engine::Proxy::uninitialize()
{
	auto uptime = (std::clock() - handle->start) / (float)CLOCKS_PER_SEC;
	TRN::Engine::Broker::handle->to_caller->post([=]() 
	{
		callback_information("Proxy uptime : " + std::to_string(uptime) + " seconds");
	//	handle->visitor->visit(shared_from_this());
	});

	TRN::Engine::Broker::uninitialize();
}
void TRN::Engine::Proxy::start()
{

	handle->start = std::clock();
	TRN::Engine::Node::start();
	TRN::Engine::Broker::start();
}

void TRN::Engine::Proxy::callback_ack(const unsigned int &id, const std::size_t &number, const bool &success, const std::string &cause)
{
	/*TRN::Engine::Message<TRN::Engine::Tag::ACK> message;

	message.id = id;
	message.number = number;
	message.success = success;
	message.cause = cause;

	handle->frontend_proxy->send(message, 0);*/
}
void TRN::Engine::Proxy::callback_configured(const unsigned int &id)
{

	TRN::Engine::Message<TRN::Engine::Tag::CONFIGURED> message;

	message.id = id;

	handle->frontend_proxy->send(message, 0);

}
void TRN::Engine::Proxy::callback_processor(const int &rank, const std::string &host, const unsigned int &index, const std::string &name)
{
	TRN::Engine::Message<TRN::Engine::Tag::WORKER> message;

	message.host = host;
	message.index = index;
	message.name = name;
	message.rank = rank;

	handle->frontend_proxy->send(message, 0);
}
void TRN::Engine::Proxy::callback_allocated(const unsigned int &id, const int &rank)
{
	TRN::Engine::Message<TRN::Engine::Tag::ALLOCATED> message;

	message.id = id;

	handle->frontend_proxy->send(message, 0);
}
void TRN::Engine::Proxy::callback_deallocated(const unsigned int &id, const int &rank)
{
	TRN::Engine::Message<TRN::Engine::Tag::DEALLOCATED> message;

	message.id = id;

	handle->frontend_proxy->send(message, 0);
}
void TRN::Engine::Proxy::callback_quit(const int &rank)
{
	TRN::Engine::Message<TRN::Engine::Tag::QUIT> message;

	message.rank = rank;

	handle->frontend_proxy->send(message, 0);
}

void TRN::Engine::Proxy::callback_completed()
{
	std::cout << "SIMULATIONS COMPLETED" << std::endl;
}
void TRN::Engine::Proxy::callback_trained(const unsigned int &id)
{
	TRN::Engine::Message<TRN::Engine::Tag::TRAINED> message;

	message.id = id;
	handle->frontend_proxy->send(message, 0);
}
void TRN::Engine::Proxy::callback_primed(const unsigned int &id)
{
	TRN::Engine::Message<TRN::Engine::Tag::PRIMED> message;

	message.id = id;
	handle->frontend_proxy->send(message, 0);
}
void TRN::Engine::Proxy::callback_tested(const unsigned int &id)
{
	TRN::Engine::Message<TRN::Engine::Tag::TESTED> message;

	message.id = id;
	handle->frontend_proxy->send(message, 0);
}
void TRN::Engine::Proxy::callback_error(const std::string &message)
{
	std::cerr << "ERROR : " << message << std::endl;
}
void TRN::Engine::Proxy::callback_information(const std::string &message)
{
	std::cout << "INFORMATION : " << message << std::endl;
}
void TRN::Engine::Proxy::callback_warning(const std::string &message)
{
	std::cerr << "WARNING : " << message << std::endl;
}
void TRN::Engine::Proxy::callback_measurement_readout_mean_square_error(const unsigned int &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols)
{
	TRN::Engine::Message<TRN::Engine::MEASUREMENT_READOUT_MEAN_SQUARE_ERROR> message;

	message.id = id;
	message.trial = trial;
	message.evaluation = evaluation;
	message.elements = values;

	message.rows = rows;
	message.cols = cols;

	auto locked = TRN::Engine::Node::implementor.lock();
	if (locked)
		locked->send(message, 0);
}
void TRN::Engine::Proxy::callback_measurement_readout_frechet_distance(const unsigned int &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols)
{
	TRN::Engine::Message<TRN::Engine::MEASUREMENT_READOUT_FRECHET_DISTANCE> message;

	message.id = id;
	message.trial = trial;
	message.evaluation = evaluation;
	message.elements = values;
	message.rows = rows;
	message.cols = cols;

	auto locked = TRN::Engine::Node::implementor.lock();
	if (locked)
		locked->send(message, 0);
}
void TRN::Engine::Proxy::callback_measurement_readout_custom(const unsigned int &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &primed, const std::vector<float> &predicted, const std::vector<float> &expected, const std::size_t &preamble, const std::size_t &pages, const std::size_t &rows, const  std::size_t &cols)
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

	auto locked = TRN::Engine::Node::implementor.lock();
	if (locked)
		locked->send(message, 0);
}
void TRN::Engine::Proxy::callback_measurement_position_mean_square_error(const unsigned int &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols)
{
	TRN::Engine::Message<TRN::Engine::MEASUREMENT_POSITION_MEAN_SQUARE_ERROR> message;

	message.id = id;
	message.trial = trial;
	message.evaluation = evaluation;
	message.elements = values;
	message.rows = rows;
	message.cols = cols;

	auto locked = TRN::Engine::Node::implementor.lock();
	if (locked)
		locked->send(message, 0);
}
void TRN::Engine::Proxy::callback_measurement_position_frechet_distance(const unsigned int &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols)
{
	TRN::Engine::Message<TRN::Engine::MEASUREMENT_POSITION_FRECHET_DISTANCE> message;

	message.id = id;
	message.trial = trial;
	message.evaluation = evaluation;
	message.elements = values;
	message.rows = rows;
	message.cols = cols;

	auto locked = TRN::Engine::Node::implementor.lock();
	if (locked)
		locked->send(message, 0);
}
void TRN::Engine::Proxy::callback_measurement_position_custom(const unsigned int &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &primed, const std::vector<float> &predicted, const std::vector<float> &expected, const std::size_t &preamble, const std::size_t &pages, const std::size_t &rows, const  std::size_t &cols)
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
	auto locked = TRN::Engine::Node::implementor.lock();
	if (locked)
		locked->send(message, 0);
}
void TRN::Engine::Proxy::callback_performances(const unsigned int &id, const std::string &phase, const size_t &batch_size, const size_t &cycles, const float &gflops, const float &seconds)
{
	TRN::Engine::Message<TRN::Engine::PERFORMANCES> message;

	message.id = id;
	message.phase = phase;

	message.cycles = cycles;
	message.gflops = gflops;
	message.seconds = seconds;
	message.batch_size = batch_size;
	auto locked = TRN::Engine::Node::implementor.lock();
	if (locked)
		locked->send(message, 0);
}
void TRN::Engine::Proxy::callback_states(const unsigned int &id, const std::string &phase, const std::string &label, const std::size_t &batch, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &samples, const std::size_t &rows, const std::size_t &cols)
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

	auto locked = TRN::Engine::Node::implementor.lock();
	if (locked)
		locked->send(message, 0);
}
void TRN::Engine::Proxy::callback_weights(const unsigned int &id, const std::string &phase, const std::string &label, const std::size_t &batch, const std::size_t &trial, const std::vector<float> &samples, const std::size_t &rows, const std::size_t &cols)
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

	auto locked = TRN::Engine::Node::implementor.lock();
	if (locked)
		locked->send(message, 0);
}
void TRN::Engine::Proxy::callback_position(const unsigned int &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols)
{
	TRN::Engine::Message<TRN::Engine::POSITION> message;
	message.id = id;
	message.trial = trial;
	message.evaluation = evaluation;
	message.elements = position;
	message.rows = rows;
	message.cols = cols;
	auto locked = TRN::Engine::Node::implementor.lock();
	if (locked)
		locked->send(message, 0);
}
void TRN::Engine::Proxy::callback_stimulus(const unsigned int &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &stimulus, const std::size_t &rows, const std::size_t &cols)
{
	TRN::Engine::Message<TRN::Engine::STIMULUS> message;
	message.id = id;
	message.trial = trial;
	message.evaluation = evaluation;
	message.elements = stimulus;
	message.rows = rows;
	message.cols = cols;
	auto locked = TRN::Engine::Node::implementor.lock();
	if (locked)
		locked->send(message, 0);
}
void TRN::Engine::Proxy::callback_mutator(const unsigned int &id, const unsigned long &seed, const std::size_t &trial, const std::vector<int> &offsets, const std::vector<int> &durations)
{
	TRN::Engine::Message<TRN::Engine::Tag::MUTATOR_CUSTOM> message;

	message.trial = trial;
	message.id = id;
	message.offsets = offsets;
	message.durations = durations;
	message.seed = seed;
	auto locked = TRN::Engine::Node::implementor.lock();
	if (locked)
		locked->send(message, 0);
}
void TRN::Engine::Proxy::callback_scheduler(const unsigned int &id, const unsigned long &seed, const std::size_t &trial, const std::vector<float> &elements, const std::size_t &rows, const std::size_t &cols, const std::vector<int> &offsets, const std::vector<int> &durations)
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
	auto locked = TRN::Engine::Node::implementor.lock();
	if (locked)
		locked->send(message, 0);
}
void TRN::Engine::Proxy::callback_scheduling(const unsigned int &id, const std::size_t &trial, const std::vector<int> &offsets, const std::vector<int> &durations)
{
	TRN::Engine::Message<TRN::Engine::SCHEDULING> scheduling;

	scheduling.id = id;
	scheduling.trial = trial;
	scheduling.offsets = offsets;
	scheduling.durations = durations;
	scheduling.is_from_mutator = false;

	auto locked = TRN::Engine::Node::implementor.lock();
	if (locked)
		locked->send(scheduling, 0);
}
void TRN::Engine::Proxy::callback_feedforward(const unsigned int &id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const  std::size_t &cols)
{
	TRN::Engine::Message<TRN::Engine::Tag::FEEDFORWARD_DIMENSIONS> message;

	message.id = id;
	message.matrices = matrices;
	message.rows = rows;
	message.cols = cols;
	message.seed = seed;
	auto locked = TRN::Engine::Node::implementor.lock();
	if (locked)
		locked->send(message, 0);
}
void TRN::Engine::Proxy::callback_feedback(const unsigned int &id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const  std::size_t &cols)
{
	TRN::Engine::Message<TRN::Engine::Tag::FEEDBACK_DIMENSIONS> message;

	message.id = id;
	message.matrices = matrices;
	message.rows = rows;
	message.cols = cols;
	message.seed = seed;
	auto locked = TRN::Engine::Node::implementor.lock();
	if (locked)
		locked->send(message, 0);
}
void TRN::Engine::Proxy::callback_readout(const unsigned int &id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const  std::size_t &cols)
{
	TRN::Engine::Message<TRN::Engine::Tag::READOUT_DIMENSIONS> message;

	message.id = id;
	message.matrices = matrices;
	message.rows = rows;
	message.cols = cols;
	message.seed = seed;
	auto locked = TRN::Engine::Node::implementor.lock();
	if (locked)
		locked->send(message, 0);
}
void TRN::Engine::Proxy::callback_recurrent(const unsigned int &id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)
{
	TRN::Engine::Message<TRN::Engine::Tag::RECURRENT_DIMENSIONS> message;

	message.id = id;
	message.matrices = matrices;
	message.rows = rows;
	message.cols = cols;
	message.seed = seed;
	auto locked = TRN::Engine::Node::implementor.lock();
	if (locked)
		locked->send(message, 0);
}
/*void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::READY> &message)
{
	ready(message.id);
	auto processor = TRN::Engine::Broker::handle->manager->retrieve(message.id);
	processor->ready();


}*/
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::COMPLETED> &message)
{
	
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::ALLOCATE> &message)
{
	allocate(message.id);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::DEALLOCATE> &message)
{
	deallocate(message.id);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::TRAIN> &message)
{

	train(message.id, message.label, message.incoming, message.expected);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::TEST> &message)
{
	test(message.id, message.label, message.incoming, message.expected, message.preamble, message.supplementary_generations);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::DECLARE_SEQUENCE> &message)
{
	declare_sequence(message.id, message.label, message.tag, message.sequence, message.observations);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::DECLARE_SET> &message)
{
	declare_set(message.id, message.label, message.tag, message.labels);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::SETUP_STATES> &message)
{
	setup_states(message.id, message.train, message.prime, message.generate);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::SETUP_WEIGHTS> &message)
{
	setup_weights(message.id, message.initialization, message.train);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::SETUP_PERFORMANCES> &message)
{
	setup_performances(message.id, message.train, message.prime, message.generate);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::SETUP_SCHEDULING> &message)
{
	setup_scheduling(message.id);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_BEGIN> &message)
{
	configure_begin(message.id);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_END> &message)
{
	configure_end(message.id);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_MEASUREMENT_READOUT_MEAN_SQUARE_ERROR> &message)
{
	configure_measurement_readout_mean_square_error(message.id, message.batch_size);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_MEASUREMENT_READOUT_FRECHET_DISTANCE> &message)
{
	configure_measurement_readout_frechet_distance(message.id, message.batch_size);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_MEASUREMENT_READOUT_CUSTOM> &message)
{
	configure_measurement_readout_custom(message.id, message.batch_size);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_MEASUREMENT_POSITION_MEAN_SQUARE_ERROR> &message)
{
	configure_measurement_position_mean_square_error(message.id, message.batch_size);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_MEASUREMENT_POSITION_FRECHET_DISTANCE> &message)
{
	configure_measurement_position_frechet_distance(message.id, message.batch_size);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_MEASUREMENT_POSITION_CUSTOM> &message)
{
	configure_measurement_position_custom(message.id, message.batch_size);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_RESERVOIR_WIDROW_HOFF> &message)
{
	configure_reservoir_widrow_hoff(message.id, message.stimulus_size, message.prediction_size, message.reservoir_size, message.leak_rate, message.initial_state_scale, message.learning_rate, message.seed, message.batch_size);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_LOOP_COPY> &message)
{
	configure_loop_copy(message.id, message.batch_size, message.stimulus_size);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_LOOP_SPATIAL_FILTER> &message)
{
	configure_loop_spatial_filter(message.id, message.batch_size, message.stimulus_size, message.seed, message.rows, message.cols, message.x, message.y, message.response, message.sigma, message.radius, message.scale, message.tag);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_LOOP_CUSTOM> &message)
{
	configure_loop_custom(message.id, message.batch_size, message.stimulus_size);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_SCHEDULER_TILED> &message)
{
	configure_scheduler_tiled(message.id, message.epochs);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_SCHEDULER_SNIPPETS> &message)
{
	configure_scheduler_snippets(message.id, message.seed, message.snippets_size, message.time_budget, message.tag);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_SCHEDULER_CUSTOM> &message)
{
	configure_scheduler_custom(message.id, message.seed, message.tag);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_MUTATOR_SHUFFLE> &message)
{
	configure_mutator_shuffle(message.id, message.seed);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_MUTATOR_REVERSE> &message)
{
	configure_mutator_reverse(message.id, message.seed, message.rate, message.size);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_MUTATOR_PUNCH> &message)
{
	configure_mutator_punch(message.id, message.seed, message.rate, message.size, message.repetition);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_MUTATOR_CUSTOM> &message)
{
	configure_mutator_custom(message.id, message.seed);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_FEEDFORWARD_UNIFORM> &message)
{
	configure_feedforward_uniform(message.id, message.a, message.b, message.sparsity);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_FEEDFORWARD_GAUSSIAN> &message)
{
	configure_feedforward_gaussian(message.id, message.mu, message.sigma);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_FEEDFORWARD_CUSTOM> &message)
{
	configure_feedforward_custom(message.id);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_FEEDBACK_UNIFORM> &message)
{
	configure_feedback_uniform(message.id, message.a, message.b, message.sparsity);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_FEEDBACK_GAUSSIAN> &message)
{
	configure_feedback_gaussian(message.id, message.mu, message.sigma);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_FEEDBACK_CUSTOM> &message)
{
	configure_feedback_custom(message.id);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_RECURRENT_UNIFORM> &message)
{
	configure_recurrent_uniform(message.id, message.a, message.b, message.sparsity);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_RECURRENT_GAUSSIAN> &message)
{
	configure_recurrent_gaussian(message.id, message.mu, message.sigma);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_RECURRENT_CUSTOM> &message)
{
	configure_recurrent_custom(message.id);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_READOUT_UNIFORM> &message)
{
	configure_readout_uniform(message.id, message.a, message.b, message.sparsity);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_READOUT_GAUSSIAN> &message)
{
	configure_readout_gaussian(message.id, message.mu, message.sigma);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_READOUT_CUSTOM> &message)
{
	configure_readout_custom(message.id);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::POSITION> &message)
{
	notify_position(message.id, message.trial, message.evaluation, message.elements, message.rows, message.cols);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::STIMULUS> &message)
{
	notify_stimulus(message.id, message.trial, message.evaluation, message.elements, message.rows, message.cols);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::SCHEDULING> &message)
{
	if (message.is_from_mutator)
	{
		notify_mutator(message.id, message.trial, message.offsets, message.durations);
	}
	else
	{
		notify_scheduler(message.id, message.trial, message.offsets, message.durations);
	}
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::FEEDFORWARD_WEIGHTS> &message)
{
	notify_feedforward(message.id, message.elements, message.matrices, message.rows, message.cols);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::RECURRENT_WEIGHTS> &message)
{
	notify_recurrent(message.id, message.elements, message.matrices, message.rows, message.cols);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::FEEDBACK_WEIGHTS> &message)
{
	notify_feedback(message.id, message.elements, message.matrices, message.rows, message.cols);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::READOUT_WEIGHTS> &message)
{
	notify_readout(message.id, message.elements, message.matrices, message.rows, message.cols);
}
std::shared_ptr<TRN::Engine::Proxy> TRN::Engine::Proxy::create(const std::shared_ptr<TRN::Engine::Communicator> &frontend_proxy, const std::shared_ptr<TRN::Engine::Communicator> &proxy_workers, const std::shared_ptr<TRN::Helper::Visitor<TRN::Engine::Proxy>> &visitor)
{
	return std::make_shared<TRN::Engine::Proxy>(frontend_proxy, proxy_workers, visitor);
}