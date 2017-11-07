#include "stdafx.h"
#include "Proxy_impl.h"
#include "Node_impl.h"
#include "Broker_impl.h"


TRN::Engine::Proxy::Proxy(const std::shared_ptr<TRN::Engine::Communicator> &frontend_proxy, const std::shared_ptr<TRN::Engine::Communicator> &proxy_workers, const std::shared_ptr<TRN::Helper::Visitor<TRN::Engine::Proxy>> &visitor) :
	TRN::Engine::Node(frontend_proxy, -1),
	handle(std::make_unique<Handle>())
{
	handle->backend = TRN::Engine::Backend::create(frontend_proxy, proxy_workers);
	handle->visitor = visitor;

}

TRN::Engine::Proxy::~Proxy()
{	
	handle.reset();
}

void TRN::Engine::Proxy::initialize()
{
	TRN::Engine::Node::initialize();
	handle->start = std::clock();
	handle->backend->start();

}
void TRN::Engine::Proxy::uninitialize()
{
	handle->backend->join();
	auto uptime = (std::clock() - handle->start) / (float)CLOCKS_PER_SEC;
	std::cout << "Proxy uptime : " + std::to_string(uptime) + " seconds" << std::endl;
	
	TRN::Engine::Node::uninitialize();
	
	handle->visitor->visit(shared_from_this());
}




void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::COMPLETED> &message)
{
	handle->backend->join();
	std::cout << "BACKEND JOINED" << std::endl;
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::ALLOCATE> &message)
{
	handle->backend->allocate(message.id);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::DEALLOCATE> &message)
{
	handle->backend->deallocate(message.id);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::TRAIN> &message)
{
	handle->backend->train(message.id, message.label, message.incoming, message.expected);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::TEST> &message)
{
	handle->backend->test(message.id, message.label, message.incoming, message.expected, message.preamble, message.autonomous, message.supplementary_generations);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::DECLARE_SEQUENCE> &message)
{
	handle->backend->declare_sequence(message.id, message.label, message.tag, message.sequence, message.observations);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::DECLARE_SET> &message)
{
	handle->backend->declare_set(message.id, message.label, message.tag, message.labels);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::SETUP_STATES> &message)
{
	handle->backend->setup_states(message.id, message.train, message.prime, message.generate);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::SETUP_WEIGHTS> &message)
{
	handle->backend->setup_weights(message.id, message.initialization, message.train);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::SETUP_PERFORMANCES> &message)
{
	handle->backend->setup_performances(message.id, message.train, message.prime, message.generate);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::SETUP_SCHEDULING> &message)
{
	handle->backend->setup_scheduling(message.id);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_BEGIN> &message)
{
	handle->backend->configure_begin(message.id);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_END> &message)
{
	handle->backend->configure_end(message.id);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_MEASUREMENT_READOUT_MEAN_SQUARE_ERROR> &message)
{
	handle->backend->configure_measurement_readout_mean_square_error(message.id, message.batch_size);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_MEASUREMENT_READOUT_FRECHET_DISTANCE> &message)
{
	handle->backend->configure_measurement_readout_frechet_distance(message.id, message.batch_size);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_MEASUREMENT_READOUT_CUSTOM> &message)
{
	handle->backend->configure_measurement_readout_custom(message.id, message.batch_size);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_MEASUREMENT_POSITION_MEAN_SQUARE_ERROR> &message)
{
	handle->backend->configure_measurement_position_mean_square_error(message.id, message.batch_size);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_MEASUREMENT_POSITION_FRECHET_DISTANCE> &message)
{
	handle->backend->configure_measurement_position_frechet_distance(message.id, message.batch_size);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_MEASUREMENT_POSITION_CUSTOM> &message)
{
	handle->backend->configure_measurement_position_custom(message.id, message.batch_size);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_RESERVOIR_WIDROW_HOFF> &message)
{
	handle->backend->configure_reservoir_widrow_hoff(message.id, message.stimulus_size, message.prediction_size, message.reservoir_size, message.leak_rate, message.initial_state_scale, message.learning_rate, message.seed, message.batch_size);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_LOOP_COPY> &message)
{
	handle->backend->configure_loop_copy(message.id, message.batch_size, message.stimulus_size);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_LOOP_SPATIAL_FILTER> &message)
{
	handle->backend->configure_loop_spatial_filter(message.id, message.batch_size, message.stimulus_size, message.seed, message.rows, message.cols, message.x, message.y, message.response, message.sigma, message.radius, message.scale, message.tag);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_LOOP_CUSTOM> &message)
{
	handle->backend->configure_loop_custom(message.id, message.batch_size, message.stimulus_size);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_SCHEDULER_TILED> &message)
{
	handle->backend->configure_scheduler_tiled(message.id, message.epochs);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_SCHEDULER_SNIPPETS> &message)
{
	handle->backend->configure_scheduler_snippets(message.id, message.seed, message.snippets_size, message.time_budget, message.tag);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_SCHEDULER_CUSTOM> &message)
{
	handle->backend->configure_scheduler_custom(message.id, message.seed, message.tag);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_MUTATOR_SHUFFLE> &message)
{
	handle->backend->configure_mutator_shuffle(message.id, message.seed);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_MUTATOR_REVERSE> &message)
{
	handle->backend->configure_mutator_reverse(message.id, message.seed, message.rate, message.size);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_MUTATOR_PUNCH> &message)
{
	handle->backend->configure_mutator_punch(message.id, message.seed, message.rate, message.size, message.repetition);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_MUTATOR_CUSTOM> &message)
{
	handle->backend->configure_mutator_custom(message.id, message.seed);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_FEEDFORWARD_UNIFORM> &message)
{
	handle->backend->configure_feedforward_uniform(message.id, message.a, message.b, message.sparsity);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_FEEDFORWARD_GAUSSIAN> &message)
{
	handle->backend->configure_feedforward_gaussian(message.id, message.mu, message.sigma);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_FEEDFORWARD_CUSTOM> &message)
{
	handle->backend->configure_feedforward_custom(message.id);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_FEEDBACK_UNIFORM> &message)
{
	handle->backend->configure_feedback_uniform(message.id, message.a, message.b, message.sparsity);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_FEEDBACK_GAUSSIAN> &message)
{
	handle->backend->configure_feedback_gaussian(message.id, message.mu, message.sigma);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_FEEDBACK_CUSTOM> &message)
{
	handle->backend->configure_feedback_custom(message.id);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_RECURRENT_UNIFORM> &message)
{
	handle->backend->configure_recurrent_uniform(message.id, message.a, message.b, message.sparsity);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_RECURRENT_GAUSSIAN> &message)
{
	handle->backend->configure_recurrent_gaussian(message.id, message.mu, message.sigma);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_RECURRENT_CUSTOM> &message)
{
	handle->backend->configure_recurrent_custom(message.id);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_READOUT_UNIFORM> &message)
{
	handle->backend->configure_readout_uniform(message.id, message.a, message.b, message.sparsity);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_READOUT_GAUSSIAN> &message)
{
	handle->backend->configure_readout_gaussian(message.id, message.mu, message.sigma);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_READOUT_CUSTOM> &message)
{
	handle->backend->configure_readout_custom(message.id);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::POSITION> &message)
{
	handle->backend->notify_position(message.id, message.trial, message.evaluation, message.elements, message.rows, message.cols);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::STIMULUS> &message)
{
	handle->backend->notify_stimulus(message.id, message.trial, message.evaluation, message.elements, message.rows, message.cols);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::SCHEDULING> &message)
{
	if (message.is_from_mutator)
	{
		handle->backend->notify_mutator(message.id, message.trial, message.offsets, message.durations);
	}
	else
	{
		handle->backend->notify_scheduler(message.id, message.trial, message.offsets, message.durations);
	}
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::FEEDFORWARD_WEIGHTS> &message)
{
	handle->backend->notify_feedforward(message.id, message.elements, message.matrices, message.rows, message.cols);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::RECURRENT_WEIGHTS> &message)
{
	handle->backend->notify_recurrent(message.id, message.elements, message.matrices, message.rows, message.cols);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::FEEDBACK_WEIGHTS> &message)
{
	handle->backend->notify_feedback(message.id, message.elements, message.matrices, message.rows, message.cols);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::READOUT_WEIGHTS> &message)
{
	handle->backend->notify_readout(message.id, message.elements, message.matrices, message.rows, message.cols);
}
std::shared_ptr<TRN::Engine::Proxy> TRN::Engine::Proxy::create(const std::shared_ptr<TRN::Engine::Communicator> &frontend_proxy, const std::shared_ptr<TRN::Engine::Communicator> &proxy_workers, const std::shared_ptr<TRN::Helper::Visitor<TRN::Engine::Proxy>> &visitor)
{
	return std::make_shared<TRN::Engine::Proxy>(frontend_proxy, proxy_workers, visitor);
}