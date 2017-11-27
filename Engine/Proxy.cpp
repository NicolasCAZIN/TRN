#include "stdafx.h"
#include "Proxy_impl.h"
#include "Node_impl.h"
#include "Broker_impl.h"


TRN::Engine::Proxy::Proxy(
	const std::shared_ptr<TRN::Engine::Communicator> &frontend_proxy,
	const std::shared_ptr<TRN::Engine::Communicator> &proxy_workers,
	const std::shared_ptr<TRN::Engine::Dispatcher> &dispatcher, 
	const std::shared_ptr<TRN::Helper::Visitor<TRN::Engine::Proxy>> &visitor, const unsigned short &number) :
	TRN::Engine::Node(frontend_proxy, -1),
	handle(std::make_unique<Handle>())
{
	
	for (int rank = 1; rank < proxy_workers->size(); rank++)
		handle->stopped[rank] = false;
	handle->to_workers = proxy_workers;
	handle->to_frontend = frontend_proxy;
	handle->number = number;
	handle->dispatcher = dispatcher;
	handle->visitor = visitor;
	handle->quit_broadcasted = false;
}

TRN::Engine::Proxy::~Proxy()
{	
	handle.reset();
}

unsigned long long TRN::Engine::Proxy::global_id(const unsigned long long &local_id)
{
	unsigned short number;
	unsigned short condition_number;
	unsigned int simulation_number;

	TRN::Engine::decode(local_id, number, condition_number, simulation_number);

	if (number != 0)
		throw std::invalid_argument("Fontend number must be 0");

	unsigned long long global_id;

	TRN::Engine::encode(handle->number, condition_number, simulation_number, global_id);

	return global_id;
}


void TRN::Engine::Proxy::initialize()
{
	TRN::Engine::Message<TRN::Engine::START> start;
	start.number = handle->number;

	handle->to_workers->broadcast(start);
	handle->start = std::clock();
}
void TRN::Engine::Proxy::uninitialize()
{
	TRN::Engine::Node::uninitialize();

	auto uptime = (std::clock() - handle->start) / (float)CLOCKS_PER_SEC;
	std::cout << "Proxy uptime : " + std::to_string(uptime) + " seconds" << std::endl;
	
	handle->visitor->visit(shared_from_this());
}

void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::QUIT> &message)
{
	if (!handle->quit_broadcasted)
	{
		TRN::Engine::Message<TRN::Engine::Tag::QUIT> quit;

		quit.terminate = false;
		quit.number = handle->number;
		handle->to_workers->broadcast(quit);
		handle->quit_broadcasted = true;
		std::cout << "PROXY #" << std::to_string(handle->number) << "QUIT BROADCASTED" << std::endl;
	}
}

void TRN::Engine::Proxy::update(const TRN::Engine::Message<TRN::Engine::EXIT> &message)
{
	if (message.number == handle->number)
	{
		std::cout << "PROXY #" << std::to_string(handle->number) << "RECEIVED EXIT RESPONSE FROM RANK " << message.rank << " QUIT BROADCASTED " << handle->quit_broadcasted << std::endl;
		if (handle->quit_broadcasted && !handle->stopped[message.rank])
		{

			handle->stopped[message.rank] = true;

			TRN::Engine::Message<TRN::Engine::Tag::EXIT> exit;

			exit.rank = message.rank;
			exit.number = 0;

			handle->to_frontend->send(exit, 0);


			TRN::Engine::Message<TRN::Engine::STOP> stop;
			stop.number = handle->number;
			handle->to_workers->broadcast(stop);


			if (std::all_of(handle->stopped.begin(), handle->stopped.end(), [](const std::pair<int, bool> &p)
			{
				return p.second;
			}))
			{
				std::cout << "PROXY #" << std::to_string(handle->number) << "RECEIVED EXIT RESPONSES" << std::endl;
			}
		}
	}	
}

/*


*/

void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::START> &message)
{
	if (handle->frontends.find(message.number) == handle->frontends.end())
		handle->frontends[message.number] = 0;
	handle->frontends[message.number]++;
	if (handle->frontends.size() != 1)
		throw std::invalid_argument("Only one frontend per proxy is allowed");
}

void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::STOP> &message)
{
	if (handle->frontends.find(message.number) == handle->frontends.end())
		throw std::invalid_argument("Frontend # " + std::to_string(message.number) + "was not started");
	handle->frontends[message.number]--;
	if (handle->frontends[message.number] == 0)
	{
		for (auto p : handle->stopped)
		{
			if (!p.second)
				throw std::runtime_error("Stopped not declared to worker# " + std::to_string(p.first));
			TRN::Engine::Message<TRN::Engine::Tag::TERMINATED> terminated;

			terminated.rank = p.first;

			handle->to_frontend->send(message, 0);
		}
		handle->to_frontend->dispose();
		//handle->dispatcher->dispose();
		stop();
		std::cout << "PROXY #"  << std::to_string(handle->number) << "JOINED" << std::endl;

	}
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::ALLOCATE> &message)
{
	handle->dispatcher->allocate(global_id(message.id));
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::DEALLOCATE> &message)
{
	handle->dispatcher->deallocate(global_id(message.id));
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::TRAIN> &message)
{
	handle->dispatcher->train(global_id(message.id), message.label, message.incoming, message.expected);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::TEST> &message)
{
	handle->dispatcher->test(global_id(message.id), message.label, message.incoming, message.expected, message.preamble, message.autonomous, message.supplementary_generations);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::DECLARE_SEQUENCE> &message)
{
	handle->dispatcher->declare_sequence(global_id(message.id), message.label, message.tag, message.sequence, message.observations);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::DECLARE_SET> &message)
{
	handle->dispatcher->declare_set(global_id(message.id), message.label, message.tag, message.labels);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::SETUP_STATES> &message)
{
	handle->dispatcher->setup_states(global_id(message.id), message.train, message.prime, message.generate);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::SETUP_WEIGHTS> &message)
{
	handle->dispatcher->setup_weights(global_id(message.id), message.initialization, message.train);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::SETUP_PERFORMANCES> &message)
{
	handle->dispatcher->setup_performances(global_id(message.id), message.train, message.prime, message.generate);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::SETUP_SCHEDULING> &message)
{
	handle->dispatcher->setup_scheduling(global_id(message.id));
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_BEGIN> &message)
{
	handle->dispatcher->configure_begin(global_id(message.id));
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_END> &message)
{
	handle->dispatcher->configure_end(global_id(message.id));
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_MEASUREMENT_READOUT_MEAN_SQUARE_ERROR> &message)
{
	handle->dispatcher->configure_measurement_readout_mean_square_error(global_id(message.id), message.batch_size);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_MEASUREMENT_READOUT_FRECHET_DISTANCE> &message)
{
	handle->dispatcher->configure_measurement_readout_frechet_distance(global_id(message.id), message.batch_size);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_MEASUREMENT_READOUT_CUSTOM> &message)
{
	handle->dispatcher->configure_measurement_readout_custom(global_id(message.id), message.batch_size);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_MEASUREMENT_POSITION_MEAN_SQUARE_ERROR> &message)
{
	handle->dispatcher->configure_measurement_position_mean_square_error(global_id(message.id), message.batch_size);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_MEASUREMENT_POSITION_FRECHET_DISTANCE> &message)
{
	handle->dispatcher->configure_measurement_position_frechet_distance(global_id(message.id), message.batch_size);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_MEASUREMENT_POSITION_CUSTOM> &message)
{
	handle->dispatcher->configure_measurement_position_custom(global_id(message.id), message.batch_size);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_RESERVOIR_WIDROW_HOFF> &message)
{
	handle->dispatcher->configure_reservoir_widrow_hoff(global_id(message.id), message.stimulus_size, message.prediction_size, message.reservoir_size, message.leak_rate, message.initial_state_scale, message.learning_rate, message.seed, message.batch_size);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_LOOP_COPY> &message)
{
	handle->dispatcher->configure_loop_copy(global_id(message.id), message.batch_size, message.stimulus_size);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_LOOP_SPATIAL_FILTER> &message)
{
	handle->dispatcher->configure_loop_spatial_filter(global_id(message.id), message.batch_size, message.stimulus_size, message.seed, message.rows, message.cols, message.x, message.y, message.response, message.sigma, message.radius, message.scale, message.tag);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_LOOP_CUSTOM> &message)
{
	handle->dispatcher->configure_loop_custom(global_id(message.id), message.batch_size, message.stimulus_size);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_SCHEDULER_TILED> &message)
{
	handle->dispatcher->configure_scheduler_tiled(global_id(message.id), message.epochs);
}

void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_SCHEDULER_SNIPPETS> &message)
{
	handle->dispatcher->configure_scheduler_snippets(global_id(message.id), message.seed, message.snippets_size, message.time_budget, message.tag);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_SCHEDULER_CUSTOM> &message)
{
	handle->dispatcher->configure_scheduler_custom(global_id(message.id), message.seed, message.tag);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_MUTATOR_SHUFFLE> &message)
{
	handle->dispatcher->configure_mutator_shuffle(global_id(message.id), message.seed);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_MUTATOR_REVERSE> &message)
{
	handle->dispatcher->configure_mutator_reverse(global_id(message.id), message.seed, message.rate, message.size);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_MUTATOR_PUNCH> &message)
{
	handle->dispatcher->configure_mutator_punch(global_id(message.id), message.seed, message.rate, message.size, message.repetition);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_MUTATOR_CUSTOM> &message)
{
	handle->dispatcher->configure_mutator_custom(global_id(message.id), message.seed);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_FEEDFORWARD_UNIFORM> &message)
{
	handle->dispatcher->configure_feedforward_uniform(global_id(message.id), message.a, message.b, message.sparsity);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_FEEDFORWARD_GAUSSIAN> &message)
{
	handle->dispatcher->configure_feedforward_gaussian(global_id(message.id), message.mu, message.sigma);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_FEEDFORWARD_CUSTOM> &message)
{
	handle->dispatcher->configure_feedforward_custom(global_id(message.id));
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_FEEDBACK_UNIFORM> &message)
{
	handle->dispatcher->configure_feedback_uniform(global_id(message.id), message.a, message.b, message.sparsity);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_FEEDBACK_GAUSSIAN> &message)
{
	handle->dispatcher->configure_feedback_gaussian(global_id(message.id), message.mu, message.sigma);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_FEEDBACK_CUSTOM> &message)
{
	handle->dispatcher->configure_feedback_custom(global_id(message.id));
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_RECURRENT_UNIFORM> &message)
{
	handle->dispatcher->configure_recurrent_uniform(global_id(message.id), message.a, message.b, message.sparsity);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_RECURRENT_GAUSSIAN> &message)
{
	handle->dispatcher->configure_recurrent_gaussian(global_id(message.id), message.mu, message.sigma);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_RECURRENT_CUSTOM> &message)
{
	handle->dispatcher->configure_recurrent_custom(global_id(message.id));
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_READOUT_UNIFORM> &message)
{
	handle->dispatcher->configure_readout_uniform(global_id(message.id), message.a, message.b, message.sparsity);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_READOUT_GAUSSIAN> &message)
{
	handle->dispatcher->configure_readout_gaussian(global_id(message.id), message.mu, message.sigma);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_READOUT_CUSTOM> &message)
{
	handle->dispatcher->configure_readout_custom(global_id(message.id));
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::POSITION> &message)
{
	handle->dispatcher->notify_position(global_id(message.id), message.trial, message.evaluation, message.elements, message.rows, message.cols);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::STIMULUS> &message)
{
	handle->dispatcher->notify_stimulus(global_id(message.id), message.trial, message.evaluation, message.elements, message.rows, message.cols);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::SCHEDULING> &message)
{
	if (message.is_from_mutator)
	{
		handle->dispatcher->notify_mutator(global_id(message.id), message.trial, message.offsets, message.durations);
	}
	else
	{
		handle->dispatcher->notify_scheduler(global_id(message.id), message.trial, message.offsets, message.durations);
	}
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::FEEDFORWARD_WEIGHTS> &message)
{
	handle->dispatcher->notify_feedforward(global_id(message.id), message.elements, message.matrices, message.rows, message.cols);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::RECURRENT_WEIGHTS> &message)
{
	handle->dispatcher->notify_recurrent(global_id(message.id), message.elements, message.matrices, message.rows, message.cols);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::FEEDBACK_WEIGHTS> &message)
{
	handle->dispatcher->notify_feedback(global_id(message.id), message.elements, message.matrices, message.rows, message.cols);
}
void TRN::Engine::Proxy::process(const TRN::Engine::Message<TRN::Engine::Tag::READOUT_WEIGHTS> &message)
{
	handle->dispatcher->notify_readout(global_id(message.id), message.elements, message.matrices, message.rows, message.cols);
}
std::shared_ptr<TRN::Engine::Proxy> TRN::Engine::Proxy::create(
	const std::shared_ptr<TRN::Engine::Communicator> &frontend_proxy,
	const std::shared_ptr<TRN::Engine::Communicator> &proxy_workers,
	const std::shared_ptr<TRN::Engine::Dispatcher> &dispatcher, const std::shared_ptr<TRN::Helper::Visitor<TRN::Engine::Proxy>> &visitor, const unsigned short &number)
{
	return std::make_shared<TRN::Engine::Proxy>(frontend_proxy, proxy_workers, dispatcher, visitor, number);
}