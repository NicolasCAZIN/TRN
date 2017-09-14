#include "stdafx.h"
#include "Broker_impl.h"

TRN::Engine::Broker::Broker(const std::shared_ptr<TRN::Engine::Communicator> &communicator) :
	handle(std::make_unique<Handle>())
{
	handle->communicator = communicator;
	handle->manager = TRN::Engine::Manager::create(communicator->size());
	handle->count = 0;
	handle->on_ack = [](const unsigned int &id, const std::size_t &number, const bool &success, const std::string &cause) {};
	handle->on_processor = [&](const int &rank, const std::string &host, const unsigned int &index, const std::string &name) {};
	handle->on_allocation = [&](const unsigned int &id, const int &rank) {};
	handle->on_deallocation = [&](const unsigned int &id, const int &rank) {};
}

void TRN::Engine::Broker::start()
{
	handle->receive = std::thread(&TRN::Engine::Broker::receive, this);
}

void TRN::Engine::Broker::stop()
{
	handle->manager->wait_not_allocated();
	TRN::Engine::Message<TRN::Engine::Tag::QUIT> quit;
	for (auto processor : handle->manager->get_processors())
	{
		handle->communicator->send(quit, processor->get_rank());
	}

	if (handle->receive.joinable())
		handle->receive.join();
}

TRN::Engine::Broker::~Broker()
{
	handle.reset();
}


void TRN::Engine::Broker::receive()
{
	TRN::Engine::Tag tag;
	auto active = handle->manager->get_processors().size();
	while (active > 0)
	{
		auto tag = handle->communicator->probe(0);
		PrintThread{} << "BROKER received tag " << tag << " " << __FUNCTION__ << std::endl;
		switch (tag)
		{
				case TRN::Engine::QUIT:
				{
					auto message = handle->communicator->receive<TRN::Engine::QUIT>(0);
					active--;
				}
				break;
				case TRN::Engine::WORKER:
				{
					auto message = handle->communicator->receive<TRN::Engine::WORKER>(0);
					auto processor = handle->manager->get_processors()[message.rank - 1];
					processor->set_name(message.name);
					processor->set_host(message.host);
					processor->set_index(message.index);
					handle->on_processor(message.rank, message.host, message.index, message.name);
				}
				break;
				case TRN::Engine::ACK:
				{
					auto message = handle->communicator->receive<TRN::Engine::ACK>(0);
					handle->on_ack(message.id, message.number, message.success, message.cause);
					if (message.success)
					{
						std::unique_lock<std::mutex> lock(handle->ack);
						if (handle->on_ack_map.find(message.number) == handle->on_ack_map.end())
							throw std::runtime_error("Ack functor for message #" + std::to_string(message.number) + " is not setup");
						handle->on_ack_map[message.number]();
						handle->on_ack_map.erase(message.number);
						lock.unlock();
					}
					else
					{
						std::cerr << "Simulator #" << message.id << "ACK : " << message.cause << std::endl;
					}
				}
				break;

				case TRN::Engine::LOG_INFORMATION:
				{
					auto message = handle->communicator->receive<TRN::Engine::LOG_INFORMATION>(0);
					//PrintThread{} << "Simulator #" << message.id << "INFORMATION : " << message.message << std::endl;
				}
				break;

				case TRN::Engine::LOG_WARNING:
				{
					auto message = handle->communicator->receive<TRN::Engine::LOG_WARNING>(0);
					std::cerr << "Simulator #" << message.id << "WARNING : " << message.message << std::endl;
				}
				break;

				case TRN::Engine::LOG_ERROR:
				{
					auto message = handle->communicator->receive<TRN::Engine::LOG_ERROR>(0);
					std::cerr << "Simulator #" << message.id << "ERROR : " << message.message << std::endl;
				}
				break;

				case TRN::Engine::POSITION:
				{
					auto message = handle->communicator->receive<TRN::Engine::POSITION>(0);
					std::unique_lock<std::mutex> lock(handle->functors);
					if (handle->predicted_position.find(message.id) == handle->predicted_position.end())
						throw std::runtime_error("Predicted position functor for message #" + std::to_string(message.id) + " is not setup");
					handle->predicted_position[message.id](message.trial, message.evaluation, message.elements, message.rows, message.cols);
					lock.unlock();
				}
				break;

				case TRN::Engine::STIMULUS:
				{
					auto message = handle->communicator->receive<TRN::Engine::STIMULUS>(0);
					std::unique_lock<std::mutex> lock(handle->functors);
					if (handle->predicted_stimulus.find(message.id) == handle->predicted_stimulus.end())
						throw std::runtime_error("Predicted Stimulus functor for message #" + std::to_string(message.id) + " is not setup");
					handle->predicted_stimulus[message.id](message.trial, message.evaluation, message.elements, message.rows, message.cols);
					lock.unlock();
				}
				break;

				case TRN::Engine::STATES:
				{
					auto message = handle->communicator->receive<TRN::Engine::STATES>(0);
					std::unique_lock<std::mutex> lock(handle->functors);
					if (handle->states.find(message.id) == handle->states.end())
						throw std::runtime_error("States functor for message #" + std::to_string(message.id) + " is not setup");
					handle->states[message.id](message.phase, message.label, message.batch, message.trial, message.evaluation, message.elements, message.rows, message.cols);
					lock.unlock();
				}
				break;

				case TRN::Engine::WEIGHTS:
				{
					auto message = handle->communicator->receive<TRN::Engine::WEIGHTS>(0);
					std::unique_lock<std::mutex> lock(handle->functors);
					if (handle->weights.find(message.id) == handle->weights.end())
						throw std::runtime_error("Weights functor for message #" + std::to_string(message.id) + " is not setup");
					handle->weights[message.id](message.phase, message.label, message.batch, message.trial, message.elements, message.rows, message.cols);
					lock.unlock();
				}
				break;

				case TRN::Engine::PERFORMANCES:
				{
					auto message = handle->communicator->receive<TRN::Engine::PERFORMANCES>(0);
					std::unique_lock<std::mutex> lock(handle->functors);
					if (handle->performances.find(message.id) == handle->performances.end())
						throw std::runtime_error("Performances functor for message #" + std::to_string(message.id) + " is not setup");
					handle->performances[message.id](message.phase, message.batch_size, message.cycles, message.gflops, message.seconds);
					lock.unlock();
				}
				break;

				case TRN::Engine::SCHEDULER_CUSTOM:
				{
					auto message = handle->communicator->receive<TRN::Engine::SCHEDULER_CUSTOM>(0);
					std::unique_lock<std::mutex> lock(handle->functors);
					if (handle->scheduler.find(message.id) == handle->scheduler.end())
						throw std::runtime_error("Scheduler functor for message #" + std::to_string(message.id) + " is not setup");
					handle->scheduler[message.id](message.seed, message.trial, message.elements, message.rows, message.cols, message.offsets, message.durations);
					lock.unlock();
				}
				break;
				case TRN::Engine::SCHEDULING:
				{
					auto message = handle->communicator->receive<TRN::Engine::SCHEDULING>(0);
					std::unique_lock<std::mutex> lock(handle->functors);
					if (handle->scheduling.find(message.id) == handle->scheduling.end())
						throw std::runtime_error("Scheduler functor for message #" + std::to_string(message.id) + " is not setup");
					handle->scheduling[message.id](message.trial, message.offsets, message.durations);
					lock.unlock();
				}
				break;
				case TRN::Engine::MUTATOR_CUSTOM:
				{
					auto message = handle->communicator->receive<TRN::Engine::MUTATOR_CUSTOM>(0);
					std::unique_lock<std::mutex> lock(handle->functors);
					if (handle->mutator.find(message.id) == handle->mutator.end())
						throw std::runtime_error("Scheduler functor for message #" + std::to_string(message.id) + " is not setup");
					handle->mutator[message.id](message.seed, message.trial, message.offsets, message.durations);
					lock.unlock();
				}
				break;
				case TRN::Engine::FEEDFORWARD_DIMENSIONS:
				{
					auto message = handle->communicator->receive<TRN::Engine::FEEDFORWARD_DIMENSIONS>(0);
					std::unique_lock<std::mutex> lock(handle->functors);
					if (handle->feedforward.find(message.id) == handle->feedforward.end())
						throw std::runtime_error("Feedforward functor for message #" + std::to_string(message.id) + " is not setup");
					handle->feedforward[message.id](message.seed, message.matrices, message.rows, message.cols);
					lock.unlock();
				}
				break;

				case TRN::Engine::RECURRENT_DIMENSIONS:
				{
					auto message = handle->communicator->receive<TRN::Engine::RECURRENT_DIMENSIONS>(0);
					std::unique_lock<std::mutex> lock(handle->functors);
					if (handle->recurrent.find(message.id) == handle->recurrent.end())
						throw std::runtime_error("Recurrent functor for message #" + std::to_string(message.id) + " is not setup");
					handle->recurrent[message.id](message.seed, message.matrices, message.rows, message.cols);
					lock.unlock();
				}
				break;

				case TRN::Engine::FEEDBACK_DIMENSIONS:
				{
					auto message = handle->communicator->receive<TRN::Engine::FEEDBACK_DIMENSIONS>(0);
					std::unique_lock<std::mutex> lock(handle->functors);
					if (handle->feedback.find(message.id) == handle->feedback.end())
						throw std::runtime_error("Feedback functor for message #" + std::to_string(message.id) + " is not setup");
					handle->feedback[message.id](message.seed, message.matrices, message.rows, message.cols);
					lock.unlock();
				}
				break;

				case TRN::Engine::READOUT_DIMENSIONS:
				{
					auto message = handle->communicator->receive<TRN::Engine::READOUT_DIMENSIONS>(0);
					std::unique_lock<std::mutex> lock(handle->functors);
					if (handle->readout.find(message.id) == handle->readout.end())
						throw std::runtime_error("Readout functor for message #" + std::to_string(message.id) + " is not setup");
					handle->readout[message.id](message.seed, message.matrices, message.rows, message.cols);
					lock.unlock();
				}
				break;

				case TRN::Engine::TRAINED:
				{
					auto message = handle->communicator->receive<TRN::Engine::TRAINED>(0);
					auto processor = handle->manager->retrieve(message.id);
					processor->trained();
					processor->set_t1(std::clock());
				}
				break;
				case TRN::Engine::PRIMED:
				{
					auto message = handle->communicator->receive<TRN::Engine::PRIMED>(0);
					auto processor = handle->manager->retrieve(message.id);
					processor->primed();
				}
				break;
				case TRN::Engine::TESTED:
				{
					auto message = handle->communicator->receive<TRN::Engine::TESTED>(0);
					auto processor = handle->manager->retrieve(message.id);
					processor->tested();
				}
				break;
				case TRN::Engine::MEASUREMENT_READOUT_MEAN_SQUARE_ERROR:
				{
					auto message = handle->communicator->receive<TRN::Engine::MEASUREMENT_READOUT_MEAN_SQUARE_ERROR>(0);
					std::unique_lock<std::mutex> lock(handle->functors);
					if (handle->measurement_readout_mean_square_error.find(message.id) == handle->measurement_readout_mean_square_error.end())
						throw std::runtime_error("Prediction mean square error functor for message #" + std::to_string(message.id) + " is not setup");
					handle->measurement_readout_mean_square_error[message.id](message.trial, message.evaluation, message.elements, message.rows, message.cols);
					lock.unlock();
				}
				break;
				case TRN::Engine::MEASUREMENT_READOUT_FRECHET_DISTANCE:
				{
					auto message = handle->communicator->receive<TRN::Engine::MEASUREMENT_READOUT_FRECHET_DISTANCE>(0);
					std::unique_lock<std::mutex> lock(handle->functors);
					if (handle->measurement_readout_frechet_distance.find(message.id) == handle->measurement_readout_frechet_distance.end())
						throw std::runtime_error("Prediction Frechet distance functor for message #" + std::to_string(message.id) + " is not setup");
					handle->measurement_readout_frechet_distance[message.id](message.trial, message.evaluation, message.elements, message.rows, message.cols);
					lock.unlock();
				}
				break;
				case TRN::Engine::MEASUREMENT_READOUT_CUSTOM:
				{
					auto message = handle->communicator->receive<TRN::Engine::MEASUREMENT_READOUT_CUSTOM>(0);
					std::unique_lock<std::mutex> lock(handle->functors);
					if (handle->measurement_readout_custom.find(message.id) == handle->measurement_readout_custom.end())
						throw std::runtime_error("Prediction custom functor for message #" + std::to_string(message.id) + " is not setup");
					handle->measurement_readout_custom[message.id](message.trial, message.evaluation, message.primed, message.elements, message.expected, message.preamble, message.matrices, message.rows, message.cols);
					lock.unlock();
				}
				break;
				case TRN::Engine::MEASUREMENT_POSITION_MEAN_SQUARE_ERROR:
				{
					auto message = handle->communicator->receive<TRN::Engine::MEASUREMENT_POSITION_MEAN_SQUARE_ERROR>(0);
					std::unique_lock<std::mutex> lock(handle->functors);
					if (handle->measurement_position_mean_square_error.find(message.id) == handle->measurement_position_mean_square_error.end())
						throw std::runtime_error("Position mean square error functor for message #" + std::to_string(message.id) + " is not setup");
					handle->measurement_position_mean_square_error[message.id](message.trial, message.evaluation, message.elements, message.rows, message.cols);
					lock.unlock();
				}
				break;
				case TRN::Engine::MEASUREMENT_POSITION_FRECHET_DISTANCE:
				{
					auto message = handle->communicator->receive<TRN::Engine::MEASUREMENT_POSITION_FRECHET_DISTANCE>(0);
					std::unique_lock<std::mutex> lock(handle->functors);
					if (handle->measurement_position_frechet_distance.find(message.id) == handle->measurement_position_frechet_distance.end())
						throw std::runtime_error("Position Frechet distance functor for message #" + std::to_string(message.id) + " is not setup");
					handle->measurement_position_frechet_distance[message.id](message.trial, message.evaluation, message.elements, message.rows, message.cols);
					lock.unlock();
				}
				break;
				case TRN::Engine::MEASUREMENT_POSITION_CUSTOM:
				{
					auto message = handle->communicator->receive<TRN::Engine::MEASUREMENT_POSITION_CUSTOM>(0);
					std::unique_lock<std::mutex> lock(handle->functors);
					if (handle->measurement_position_custom.find(message.id) == handle->measurement_position_custom.end())
						throw std::runtime_error("Position custom functor for message #" + std::to_string(message.id) + " is not setup");
					handle->measurement_position_custom[message.id](message.trial, message.evaluation, message.primed, message.elements, message.expected, message.preamble, message.matrices, message.rows, message.cols);
					lock.unlock();
				}
				break;



				default:
					throw std::invalid_argument("Unexpected tag " + std::to_string(tag));
			}
		
	}
	//PrintThread{} << "Broker #0 quitted" << std::endl;
}


template<TRN::Engine::Tag tag>
void TRN::Engine::Broker::send(const int &rank, TRN::Engine::Message<tag> &message, const std::function<void()> &functor)
{
	std::unique_lock<std::mutex> lock(handle->ack);
	message.number = handle->count;
	handle->count++;

	////PrintThread{} << "acquiring functor lock for id " << message.id << std::endl;

	if (handle->on_ack_map.find(message.number) != handle->on_ack_map.end())
		throw std::runtime_error("Ack functor for message #" + std::to_string(message.number) + " is already setup");
	handle->on_ack_map[message.number] = functor;
	lock.unlock();

	handle->communicator->send(message, rank);
}

void TRN::Engine::Broker::install_ack(const std::function<void(const unsigned int &id, const std::size_t &number, const bool &success, const std::string &cause)> &on_ack)
{
	handle->on_ack = on_ack;
}

void TRN::Engine::Broker::install_processor(const std::function<void(const int &rank, const std::string &host, const unsigned int &index, const std::string &name)> &on_processor)
{
	handle->on_processor = on_processor;
}

void    TRN::Engine::Broker::install_allocation(const std::function<void(const unsigned int &id, const int &rank)> &on_allocation)
{
	handle->on_allocation = on_allocation;
}

void    TRN::Engine::Broker::install_deallocation(const std::function<void(const unsigned int &id, const int &rank)> &on_deallocation)
{
	handle->on_deallocation = on_deallocation;
}

void TRN::Engine::Broker::allocate(const unsigned int &id)
{
	//PrintThread{} << "entering " << id << " " << __FUNCTION__ << std::endl;
	auto processor = handle->manager->allocate(id);
	processor->post([=]()
	{
		TRN::Engine::Message<ALLOCATE> message;
		message.id = id;
		processor->allocating();
		send(processor->get_rank(), message, [=]()
		{
			processor->allocated();
			handle->on_allocation(id, handle->manager->retrieve(id)->get_rank());
			////PrintThread{} << "id " << id << " allocate ack" << std::endl;
		});
	});
	//PrintThread{} << "exiting " << id << " " << __FUNCTION__ << std::endl;
}

void TRN::Engine::Broker::deallocate(const unsigned int &id)
{
	//PrintThread{} << "entering " << id << " " << __FUNCTION__ << std::endl;
	auto processor = handle->manager->retrieve(id);
	processor->post([=]()
	{
		TRN::Engine::Message<DEALLOCATE> message;
		message.id = id;
		//PrintThread{} << "id " << id << " deallocate BEFORE" << std::endl;
		processor->deallocating();

		//PrintThread{} << "id " << id << " deallocate SEND" << std::endl;

		send(processor->get_rank(), message, [=]()
		{
			handle->manager->deallocate(id);
			//PrintThread{} << "id " << id << " deallocate ack" << std::endl;
			processor->deallocated();
			//PrintThread{} << "id " << id << " manager deallocate" << std::endl;
		

			handle->on_deallocation(id, processor->get_rank());
			//PrintThread{} << "id " << id << " deallocate DONE" << std::endl;
		});
	});
	//PrintThread{} << "exiting " << id << " " << __FUNCTION__ << std::endl;
}


void TRN::Engine::Broker::train(const unsigned int &id, const std::string &label, const std::string &incoming, const std::string &expected)
{
	auto processor = handle->manager->retrieve(id);
	processor->post([=]()
	{
		processor->training();
		processor->set_t0(std::clock());
		TRN::Engine::Message<TRAIN> message;
		message.id = id;
		message.label = label;
		message.incoming = incoming;
		message.expected = expected;

		auto start = std::clock();
		send(processor->get_rank(), message, [this, id, start]()
		{
			////PrintThread{} << "id " << id << " train ack" << std::endl;
		});
	});

}
void TRN::Engine::Broker::test(const unsigned int &id, const std::string &label, const std::string &incoming, const std::string &expected, const unsigned int &preamble, const unsigned int &supplementary_generations)
{
	auto processor = handle->manager->retrieve(id);
	processor->post([=]()
	{
		processor->testing();
		TRN::Engine::Message<TEST> message;
		message.id = id;
		message.label = label;
		message.incoming = incoming;
		message.expected = expected;
		message.preamble = preamble;
		message.supplementary_generations = supplementary_generations;
		send(processor->get_rank(), message, [this, id]()
		{
			////PrintThread{} << "id " << id << " test ack" << std::endl;
		});
	});
}
void TRN::Engine::Broker::declare_sequence(const unsigned int &id, const std::string &label, const std::string &tag, const std::vector<float> &sequence, const std::size_t &observations)
{
	auto processor = handle->manager->retrieve(id);
	processor->post([=]()
	{
		processor->declare();
		TRN::Engine::Message<DECLARE_SEQUENCE> message;
		message.id = id;
		message.label = label;
		message.tag = tag;
		message.sequence = sequence;
		message.observations = observations;
		send(processor->get_rank(), message, [id]()
		{
			////PrintThread{} << "id " << id << " sequence declare ack" << std::endl;
		});
	});
}
void TRN::Engine::Broker::declare_set(const unsigned int &id, const std::string &label, const std::string &tag, const std::vector<std::string> &labels)
{
	auto processor = handle->manager->retrieve(id);
	processor->post([=]()
	{
		processor->declare();
		TRN::Engine::Message<DECLARE_SET> message;
		message.id = id;
		message.label = label;
		message.tag = tag;
		message.labels = labels;
		send(processor->get_rank(), message, [id]()
		{
			////PrintThread{} << "id " << id << " batch declare ack" << std::endl;
		});
	});
}
void TRN::Engine::Broker::setup_states(const unsigned int &id, const std::function<void(const std::string &phase, const std::string &label, const std::size_t &batch, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &samples, const std::size_t &rows, const std::size_t &cols)> &functor, const bool &train, const bool &prime, const bool &generate)
{
	auto processor = handle->manager->retrieve(id);

	std::unique_lock<std::mutex> lock(handle->functors);
	if (handle->states.find(id) != handle->states.end())
		throw std::invalid_argument("Simulator #" + std::to_string(id) + " already have a states functor");
	handle->states[id] = functor;
	lock.unlock();

	processor->post([=]()
	{
		TRN::Engine::Message<SETUP_STATES> message;
		message.id = id;
		message.train = train;
		message.prime = prime;
		message.generate = generate;
		send(processor->get_rank(), message, [id]()
		{
			////PrintThread{} << "id " << id << " states setup acked" << std::endl;
		});
	});
}
void TRN::Engine::Broker::setup_weights(const unsigned int &id, const std::function<void(const std::string &phase, const std::string &label, const std::size_t &batch, const std::size_t &trial, const std::vector<float> &weights, const std::size_t &rows, const std::size_t &cols)> &functor, const bool &initalization, const bool &train)
{
	auto processor = handle->manager->retrieve(id);
	std::unique_lock<std::mutex> lock(handle->functors);
	if (handle->weights.find(id) != handle->weights.end())
		throw std::invalid_argument("Simulator #" + std::to_string(id) + " already have a weights functor");
	handle->weights[id] = functor;
	lock.unlock();
	processor->post([=]()
	{
		TRN::Engine::Message<SETUP_WEIGHTS> message;
		message.id = id;
		message.train = train;
		message.initialization = initalization;
	
		send(processor->get_rank(), message, [id]()
		{
			////PrintThread{} << "id " << id << " weights setup acked" << std::endl;
		});
	});
}
void TRN::Engine::Broker::setup_performances(const unsigned int &id, const std::function<void(const std::string &phase, const size_t &batch_size, const size_t &cycles, const float &gflops, const float &seconds)> &functor, const bool &train, const bool &prime, const bool &generate)
{
	auto processor = handle->manager->retrieve(id);
	std::unique_lock<std::mutex> lock(handle->functors);
	if (handle->performances.find(id) != handle->performances.end())
		throw std::invalid_argument("Simulator #" + std::to_string(id) + " already have a performances functor");
	handle->performances[id] = functor;
	lock.unlock();
	processor->post([=]()
	{
		TRN::Engine::Message<SETUP_PERFORMANCES> message;
		message.id = id;
		message.train = train;
		message.prime = prime;
		message.generate = generate;
		send(processor->get_rank(), message, [id]()
		{
			////PrintThread{} << "id " << id << " performances setup acked" << std::endl;
		});
	});
}


void TRN::Engine::Broker::setup_scheduling(const unsigned int &id, const std::function<void(const std::size_t &trial, const std::vector<int> &offsets, const std::vector<int> &durations)> &functor)
{
	auto processor = handle->manager->retrieve(id);
	std::unique_lock<std::mutex> lock(handle->functors);
	if (handle->scheduling.find(id) != handle->scheduling.end())
		throw std::invalid_argument("Simulator #" + std::to_string(id) + " already have a scheduling functor");
	handle->scheduling[id] = functor;
	lock.unlock();
	processor->post([=]()
	{
		TRN::Engine::Message<SETUP_SCHEDULING> message;
		message.id = id;
		
		send(processor->get_rank(), message, [id]()
		{
			////PrintThread{} << "id " << id << " performances setup acked" << std::endl;
		});
	});
}

void TRN::Engine::Broker::configure_begin(const unsigned int &id)
{
	auto processor = handle->manager->retrieve(id);
	processor->post([=]()
	{
		processor->configuring();

		TRN::Engine::Message<CONFIGURE_BEGIN> message;
		message.id = id;
		send(processor->get_rank(), message, [id]()
		{
			////PrintThread{} << "id " << id << " configure begin acked" << std::endl;
		});
	});
}
void TRN::Engine::Broker::configure_end(const unsigned int &id)
{
	auto processor = handle->manager->retrieve(id);
	processor->post([=]()
	{
		TRN::Engine::Message<CONFIGURE_END> message;
		message.id = id;
		send(processor->get_rank(), message, [=]()
		{
			processor->configured();
			////PrintThread{} << "id " << id << " configure end acked" << std::endl;
		});
	});
}

void 	TRN::Engine::Broker::configure_measurement_readout_mean_square_error(const unsigned int &id, const std::size_t &batch_size, const std::function<void(const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const std::size_t &cols)> &functor)
{
	auto processor = handle->manager->retrieve(id);
	std::unique_lock<std::mutex> lock(handle->functors);
	if (handle->measurement_readout_mean_square_error.find(id) != handle->measurement_readout_mean_square_error.end())
		throw std::invalid_argument("Simulator #" + std::to_string(id) + " already have a performances functor");
	handle->measurement_readout_mean_square_error[id] = functor;
	lock.unlock();
	processor->post([=]()
	{
		processor->configuring();
		TRN::Engine::Message<CONFIGURE_MEASUREMENT_READOUT_MEAN_SQUARE_ERROR> message;
		message.id = id;
		message.batch_size = batch_size;
		send(processor->get_rank(), message, [this, id]()
		{
			////PrintThread{} << "id " << id << " configure_measurement_readout_mean_square_error end acked" << std::endl;
		});
	});
}

void  	TRN::Engine::Broker::configure_measurement_readout_frechet_distance(const unsigned int &id, const std::size_t &batch_size, const std::function<void(const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const std::size_t &cols)> &functor)
{
	auto processor = handle->manager->retrieve(id);
	std::unique_lock<std::mutex> lock(handle->functors);
	if (handle->measurement_readout_frechet_distance.find(id) != handle->measurement_readout_frechet_distance.end())
		throw std::invalid_argument("Simulator #" + std::to_string(id) + " already have a performances functor");
	handle->measurement_readout_frechet_distance[id] = functor;
	lock.unlock();
	processor->post([=]()
	{
		processor->configuring();
		TRN::Engine::Message<CONFIGURE_MEASUREMENT_READOUT_FRECHET_DISTANCE> message;
		message.id = id;
		message.batch_size = batch_size;
		send(processor->get_rank(), message, [this, id]()
		{
			////PrintThread{} << "id " << id << " configure_measurement_readout_frechet_distance end acked" << std::endl;
		});
	});
}
void  	TRN::Engine::Broker::configure_measurement_readout_custom(const unsigned int &id, const std::size_t &batch_size, const std::function<void(const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &primed, const std::vector<float> &predicted, const std::vector<float> &expected, const std::size_t &preamble,const std::size_t &pages, const std::size_t &rows, const std::size_t &cols)> &functor)
{
	auto processor = handle->manager->retrieve(id);
	std::unique_lock<std::mutex> lock(handle->functors);
	if (handle->measurement_readout_custom.find(id) != handle->measurement_readout_custom.end())
		throw std::invalid_argument("Simulator #" + std::to_string(id) + " already have a performances functor");
	handle->measurement_readout_custom[id] = functor;
	lock.unlock();
	processor->post([=]()
	{
		processor->configuring();
		TRN::Engine::Message<CONFIGURE_MEASUREMENT_READOUT_CUSTOM> message;
		message.id = id;
		message.batch_size = batch_size;
		send(processor->get_rank(), message, [this, id]()
		{
			////PrintThread{} << "id " << id << " configure_measurement_readout_custom end acked" << std::endl;
		});
	});
}

void  	TRN::Engine::Broker::configure_measurement_position_mean_square_error(const unsigned int &id, const std::size_t &batch_size, const std::function<void(const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const std::size_t &cols)> &functor)
{
	auto processor = handle->manager->retrieve(id);
	std::unique_lock<std::mutex> lock(handle->functors);
	if (handle->measurement_position_mean_square_error.find(id) != handle->measurement_position_mean_square_error.end())
		throw std::invalid_argument("Simulator #" + std::to_string(id) + " already have a performances functor");
	handle->measurement_position_mean_square_error[id] = functor;
	lock.unlock();
	processor->post([=]()
	{
		processor->configuring();
		TRN::Engine::Message<CONFIGURE_MEASUREMENT_POSITION_MEAN_SQUARE_ERROR> message;
		message.id = id;
		message.batch_size = batch_size;
		send(processor->get_rank(), message, [this, id]()
		{
			////PrintThread{} << "id " << id << " configure_measurement_position_mean_square_error end acked" << std::endl;
		});
	});
}

void  	TRN::Engine::Broker::configure_measurement_position_frechet_distance(const unsigned int &id, const std::size_t &batch_size, const std::function<void(const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const std::size_t &cols)> &functor)
{
	auto processor = handle->manager->retrieve(id);
	std::unique_lock<std::mutex> lock(handle->functors);
	if (handle->measurement_position_frechet_distance.find(id) != handle->measurement_position_frechet_distance.end())
		throw std::invalid_argument("Simulator #" + std::to_string(id) + " already have a performances functor");
	handle->measurement_position_frechet_distance[id] = functor;
	lock.unlock();
	processor->post([=]()
	{
		processor->configuring();

		TRN::Engine::Message<CONFIGURE_MEASUREMENT_POSITION_FRECHET_DISTANCE> message;
		message.id = id;
		message.batch_size = batch_size;
		send(processor->get_rank(), message, [this, id]()
		{
			////PrintThread{} << "id " << id << " configure_measurement_position_frechet_distance end acked" << std::endl;
		});
	});
}
void  	TRN::Engine::Broker::configure_measurement_position_custom(const unsigned int &id, const std::size_t &batch_size, const std::function<void(const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &primed, const std::vector<float> &predicted, const std::vector<float> &expected, const std::size_t &preamble, const std::size_t &pages, const std::size_t &rows, const std::size_t &cols)> &functor)
{
	auto processor = handle->manager->retrieve(id);
	std::unique_lock<std::mutex> lock(handle->functors);
	if (handle->measurement_position_custom.find(id) != handle->measurement_position_custom.end())
		throw std::invalid_argument("Simulator #" + std::to_string(id) + " already have a performances functor");
	handle->measurement_position_custom[id] = functor;
	lock.unlock();

	processor->post([=]()
	{
		processor->configuring();
		TRN::Engine::Message<CONFIGURE_MEASUREMENT_POSITION_CUSTOM> message;
		message.id = id;
		message.batch_size = batch_size;
		send(processor->get_rank(), message, [this, id]()
		{
			////PrintThread{} << "id " << id << " configure_measurement_position_custom end acked" << std::endl;
		});
	});
}


void TRN::Engine::Broker::configure_reservoir_widrow_hoff(const unsigned int &id, const std::size_t &stimulus_size, const std::size_t &prediction_size, const std::size_t &reservoir_size, const float &leak_rate, const float &initial_state_scale, const float &learning_rate, const unsigned long &seed, const std::size_t &batch_size)
{
	auto processor = handle->manager->retrieve(id);
	processor->post([=]()
	{
		processor->configuring();
		TRN::Engine::Message<CONFIGURE_RESERVOIR_WIDROW_HOFF> message;
		message.id = id;
		message.stimulus_size = stimulus_size;
		message.prediction_size = prediction_size;
		message.reservoir_size = reservoir_size;
		message.leak_rate = leak_rate;
		message.initial_state_scale = initial_state_scale;
		message.learning_rate = learning_rate;
		message.batch_size = batch_size;
		message.seed = seed;
		send(processor->get_rank(), message, [id]()
		{
			////PrintThread{} << "id " << id << " configure reservoir widrow hoff acked" << std::endl;
		});
	});
}
void TRN::Engine::Broker::configure_loop_copy(const unsigned int &id, const std::size_t &batch_size, const std::size_t &stimulus_size)
{
	auto processor = handle->manager->retrieve(id);
	processor->post([=]()
	{
		processor->configuring();
		TRN::Engine::Message<CONFIGURE_LOOP_COPY> message;
		message.id = id;
		message.stimulus_size = stimulus_size;
		message.batch_size = batch_size;
		send(processor->get_rank(), message, [id]()
		{
			////PrintThread{} << "id " << id << " configure loop copy acked" << std::endl;
		});
	});
}

void TRN::Engine::Broker::configure_loop_spatial_filter(const unsigned int &id, const std::size_t &batch_size, const std::size_t &stimulus_size, const unsigned long &seed,
	const std::function<void(const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols)> &predicted_position,
	std::function<void(const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols)> &estimated_position,
	const std::function<void(const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols)> &predicted_stimulus,
	std::function<void(const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &stimulus, const std::size_t &rows, const std::size_t &cols)> &perceived_stimulus,
	const std::size_t &rows, const std::size_t &cols,
	const std::pair<float, float> &x, const std::pair<float, float> &y,
	const std::vector<float> &response,
	const float &sigma,
	const float &radius,
	const float &scale,
	const std::string &tag)
{
	auto processor = handle->manager->retrieve(id);
	std::unique_lock<std::mutex> lock(handle->functors);
	if (handle->predicted_position.find(id) != handle->predicted_position.end())
		throw std::invalid_argument("Simulator #" + std::to_string(id) + " already have a predicted position functor");
	handle->predicted_position[id] = predicted_position;
	if (handle->predicted_stimulus.find(id) != handle->predicted_stimulus.end())
		throw std::invalid_argument("Simulator #" + std::to_string(id) + " already have a predicted stimulus functor");
	handle->predicted_stimulus[id] = predicted_stimulus;
	lock.unlock();

	estimated_position = [this, id, processor](const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols)
	{
		TRN::Engine::Message<POSITION> message;
		message.id = id;
		message.elements = position;
		message.rows = rows;
		message.cols = cols;
		message.trial = trial;
		message.evaluation = evaluation;
		send(processor->get_rank(), message, [id]()
		{
			////PrintThread{} << "id " << id << " position acked" << std::endl;
		});
	};
	perceived_stimulus = [this, id, processor](const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &stimulus, const std::size_t &rows, const std::size_t &cols)
	{
		TRN::Engine::Message<STIMULUS> message;
		message.id = id;
		message.elements = stimulus;
		message.rows = rows;
		message.cols = cols;
		message.trial = trial;
		message.evaluation = evaluation;
		send(processor->get_rank(), message, [id]()
		{
			////PrintThread{} << "id " << id << " stimulus acked" << std::endl;
		});
	};

	processor->post([=]()
	{
		processor->configuring();

		TRN::Engine::Message<CONFIGURE_LOOP_SPATIAL_FILTER> message;

		message.id = id;
		message.stimulus_size = stimulus_size;
		message.batch_size = batch_size;
		message.rows = rows;
		message.cols = cols;
		message.x = x;
		message.y = y;
		message.response = response;
		message.sigma = sigma;
		message.radius = radius;
		message.scale = scale;
		message.tag = tag;
		message.seed = seed;
		send(processor->get_rank(), message, [id]()
		{
			////PrintThread{} << "id " << id << " configure loop spatial filter acked" << std::endl;
		});
	});
}
void TRN::Engine::Broker::configure_loop_custom(const unsigned int &id, const std::size_t &batch_size, const std::size_t &stimulus_size,
	const std::function<void(const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &prediction, const std::size_t &rows, const std::size_t &cols)> &request,
	std::function<void(const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &stimulus, const std::size_t &rows, const std::size_t &cols)> &reply)
{
	auto processor = handle->manager->retrieve(id);
	std::unique_lock<std::mutex> lock(handle->functors);
	if (handle->predicted_position.find(id) != handle->predicted_position.end())
		throw std::invalid_argument("Simulator #" + std::to_string(id) + " already have a predicted position functor");
	handle->predicted_stimulus[id] = request;
	lock.unlock();
	
	reply = [this, id, processor](const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &stimulus, const std::size_t &rows, const std::size_t &cols)
	{
		TRN::Engine::Message<STIMULUS> message;
		message.id = id;
		message.elements = stimulus;
		message.rows = rows;
		message.cols = cols;
		message.trial = trial;
		message.evaluation = evaluation;
		send(processor->get_rank(), message, [id]()
		{
			////PrintThread{} << "id " << id << " stimulus acked" << std::endl;
		});
	};
	processor->post([=]()
	{
		processor->configuring();
		TRN::Engine::Message<CONFIGURE_LOOP_CUSTOM> message;

		message.id = id;
		message.batch_size = batch_size;
		message.stimulus_size = stimulus_size;
		send(processor->get_rank(), message, [id]()
		{
			////PrintThread{} << "id " << id << " configure loop custom acked" << std::endl;
		});
	});
}

void TRN::Engine::Broker::configure_scheduler_tiled(const unsigned int &id, const unsigned int &epochs)
{
	auto processor = handle->manager->retrieve(id);
	processor->post([=]()
	{
		processor->configuring();
		TRN::Engine::Message<CONFIGURE_SCHEDULER_TILED> message;
		message.id = id;
		message.epochs = epochs;

		send(processor->get_rank(), message, [id]()
		{
			////PrintThread{} << "id " << id << " configure scheduler tiled acked" << std::endl;
		});
	});
}

void TRN::Engine::Broker::configure_scheduler_snippets(const unsigned int &id, const unsigned long &seed, const unsigned int &snippets_size, const unsigned int &time_budget,  const std::string &tag)
{
	auto processor = handle->manager->retrieve(id);
	processor->post([=]()
	{
		processor->configuring();
		TRN::Engine::Message<CONFIGURE_SCHEDULER_SNIPPETS> message;
		message.id = id;
		message.snippets_size = snippets_size;
		message.time_budget = time_budget;
		message.tag = tag;
		message.seed = seed;

		send(processor->get_rank(), message, [id]()
		{
			////PrintThread{} << "id " << id << " configure scheduler snippets acked" << std::endl;
		});
	});
}


void TRN::Engine::Broker::configure_scheduler_custom(const unsigned int &id, const unsigned long &seed,
	const std::function<void(const unsigned long &seed, const std::size_t &trial, const std::vector<float> &elements, const std::size_t &rows, const std::size_t &cols, const std::vector<int> &offsets, const std::vector<int> &durations)> &request,
	std::function<void( const std::size_t &trial, const std::vector<int> &offsets, const std::vector<int> &durations)> &reply, const std::string &tag)
{
	auto processor = handle->manager->retrieve(id);
	std::unique_lock<std::mutex> lock(handle->functors);
	if (handle->scheduler.find(id) != handle->scheduler.end())
		throw std::invalid_argument("Simulator #" + std::to_string(id) + " already have a scheduler functor");
	handle->scheduler[id] = request;
	lock.unlock();

	reply = [this, id, processor]( const std::size_t &trial, const std::vector<int> &offsets, const std::vector<int> &durations)
	{
		TRN::Engine::Message<SCHEDULING> message;
		message.id = id;

		message.is_from_mutator = false;
		message.offsets = offsets;
		message.durations = durations;
		
		send(processor->get_rank(), message, [id]()
		{
			////PrintThread{} << "id " << id << " scheduling acked" << std::endl;
		});
	};

	processor->post([=]()
	{
		processor->configuring();

		TRN::Engine::Message<CONFIGURE_SCHEDULER_CUSTOM> message;
		message.id = id;
		message.seed = seed;
		send(processor->get_rank(), message, [id]()
		{
			////PrintThread{} << "id " << id << " configure scheduler custom acked" << std::endl;
		});
	});
}

void 	TRN::Engine::Broker::configure_mutator_shuffle(const unsigned int &id,const unsigned long &seed)
{
	auto processor = handle->manager->retrieve(id);
	processor->post([=]()
	{
		processor->configuring();
		TRN::Engine::Message<CONFIGURE_MUTATOR_SHUFFLE> message;
		message.id = id;
		message.seed = seed;
		send(processor->get_rank(), message, [id]()
		{
			////PrintThread{} << "id " << id << " configure scheduler snippets acked" << std::endl;
		});
	});
}
void 	TRN::Engine::Broker::configure_mutator_reverse(const unsigned int &id, const unsigned long &seed, const float &rate, const std::size_t &size)
{
	auto processor = handle->manager->retrieve(id);
	processor->post([=]()
	{
		processor->configuring();
		TRN::Engine::Message<CONFIGURE_MUTATOR_REVERSE> message;
		message.id = id;
		message.seed = seed;
		message.rate = rate;
		message.size = size;
		send(processor->get_rank(), message, [id]()
		{
			////PrintThread{} << "id " << id << " configure scheduler snippets acked" << std::endl;
		});
	});
}
void 	TRN::Engine::Broker::configure_mutator_punch(const unsigned int &id, const unsigned long &seed, const float &rate, const std::size_t &size, const std::size_t &number)
{
	auto processor = handle->manager->retrieve(id);
	processor->post([=]()
	{
		processor->configuring();
		TRN::Engine::Message<CONFIGURE_MUTATOR_PUNCH> message;
		message.id = id;
		message.seed = seed;
		message.rate = rate;
		message.size = size;
		message.repetition = number;
		send(processor->get_rank(), message, [id]()
		{
			////PrintThread{} << "id " << id << " configure scheduler snippets acked" << std::endl;
		});
	});
}

void 	TRN::Engine::Broker::configure_mutator_custom(const unsigned int &id, const unsigned long &seed,
	const std::function<void(const unsigned long &seed, const std::size_t &trial, const std::vector<int> &offsets, const std::vector<int> &durations)> &request,
	std::function<void( const std::size_t &trial, const std::vector<int> &offsets, const std::vector<int> &durations)> &reply)
{
	auto processor = handle->manager->retrieve(id);
	std::unique_lock<std::mutex> lock(handle->functors);
	if (handle->mutator.find(id) != handle->mutator.end())
		throw std::invalid_argument("Simulator #" + std::to_string(id) + " already have a mutator/scheduler functor");
	handle->mutator[id] = request;
	lock.unlock();

	reply = [this, id, processor](const std::size_t &trial, const std::vector<int> &offsets, const std::vector< int> &durations)
	{
		TRN::Engine::Message<SCHEDULING> message;
		message.id = id;
		message.is_from_mutator = true;
		message.offsets = offsets;
		message.durations = durations;
		message.trial = trial;
		send(processor->get_rank(), message, [id]()
		{
			////PrintThread{} << "id " << id << " scheduling acked" << std::endl;
		});
	};

	processor->post([=]()
	{
		processor->configuring();

		TRN::Engine::Message<CONFIGURE_MUTATOR_CUSTOM> message;
		message.id = id;
		message.seed = seed;
		send(processor->get_rank(), message, [id]()
		{
			////PrintThread{} << "id " << id << " configure scheduler custom acked" << std::endl;
		});
	});
}





void TRN::Engine::Broker::configure_readout_uniform(const unsigned int &id, const float &a, const float &b, const float &sparsity)
{
	auto processor = handle->manager->retrieve(id);
	processor->post([=]()
	{
		processor->configuring();
		TRN::Engine::Message<CONFIGURE_READOUT_UNIFORM> message;
		message.id = id;
		message.a = a;
		message.b = b;
		message.sparsity = sparsity;
		send(processor->get_rank(), message, [id]()
		{
			////PrintThread{} << "id " << id << " configure readout uniform acked" << std::endl;
		});
	});
}

void TRN::Engine::Broker::configure_readout_gaussian(const unsigned int &id, const float &mu, const float &sigma)
{
	auto processor = handle->manager->retrieve(id);
	processor->post([=]()
	{
		processor->configuring();
		TRN::Engine::Message<CONFIGURE_READOUT_GAUSSIAN> message;
		message.id = id;
		message.mu = mu;
		message.sigma = sigma;

		send(processor->get_rank(), message, [id]()
		{
			////PrintThread{} << "id " << id << " configure readout gaussian acked" << std::endl;
		});
	});
}

void TRN::Engine::Broker::configure_readout_custom(const unsigned int &id,
	const std::function<void(const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &request,
	std::function<void(const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &reply)
{
	auto processor = handle->manager->retrieve(id);
	std::unique_lock<std::mutex> lock(handle->functors);
	if (handle->readout.find(id) != handle->readout.end())
		throw std::invalid_argument("Simulator #" + std::to_string(id) + " already have a readout functor");
	handle->readout[id] = request;
	lock.unlock();

	
	reply = [this, id,processor](const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)
	{
		TRN::Engine::Message<READOUT_WEIGHTS> message;
		message.id = id;
		message.elements = weights;
		message.matrices = matrices;
		message.rows = rows;
		message.cols = cols;
		send(processor->get_rank(), message, [id]()
		{
			////PrintThread{} << "id " << id << " readout weights acked" << std::endl;
		});
	};

	processor->post([=]()
	{
		processor->configuring();

		TRN::Engine::Message<CONFIGURE_READOUT_CUSTOM> message;
		message.id = id;

		send(processor->get_rank(), message, [id]()
		{
			////PrintThread{} << "id " << id << " configure readout custom acked" << std::endl;
		});
	});
}

void TRN::Engine::Broker::configure_feedback_uniform(const unsigned int &id, const float &a, const float &b, const float &sparsity)
{
	auto processor = handle->manager->retrieve(id);
	processor->post([=]()
	{
		processor->configuring();
		TRN::Engine::Message<CONFIGURE_FEEDBACK_UNIFORM> message;
		message.id = id;
		message.a = a;
		message.b = b;
		message.sparsity = sparsity;
		send(processor->get_rank(), message, [id]()
		{
			////PrintThread{} << "id " << id << " configure feedback uniform acked" << std::endl;
		});
	});
}

void TRN::Engine::Broker::configure_feedback_gaussian(const unsigned int &id, const float &mu, const float &sigma)
{
	auto processor = handle->manager->retrieve(id);
	processor->post([=]()
	{
		processor->configuring();
		TRN::Engine::Message<CONFIGURE_FEEDBACK_GAUSSIAN> message;
		message.id = id;
		message.mu = mu;
		message.sigma = sigma;

		send(processor->get_rank(), message, [id]()
		{
			////PrintThread{} << "id " << id << " configure feedback gaussian acked" << std::endl;
		});
	});
}

void TRN::Engine::Broker::configure_feedback_custom(const unsigned int &id,
	const std::function<void(const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &request,
	std::function<void(const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &reply)
{
	auto processor = handle->manager->retrieve(id);
	std::unique_lock<std::mutex> lock(handle->functors);
	if (handle->feedback.find(id) != handle->feedback.end())
		throw std::invalid_argument("Simulator #" + std::to_string(id) + " already have a feedback functor");
	handle->feedback[id] = request;
	lock.unlock();

	reply = [this, id, processor](const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)
	{
		TRN::Engine::Message<FEEDBACK_WEIGHTS> message;
		message.id = id;
		message.elements = weights;
		message.matrices = matrices;
		message.rows = rows;
		message.cols = cols;
		send(processor->get_rank(), message, [id]()
		{
			////PrintThread{} << "id " << id << " feedback weights acked" << std::endl;
		});
	};

	processor->post([=]()
	{
		processor->configuring();

		TRN::Engine::Message<CONFIGURE_FEEDBACK_CUSTOM> message;
		message.id = id;

		send(processor->get_rank(), message, [id]()
		{
			////PrintThread{} << "id " << id << " configure feedback custom acked" << std::endl;
		});
	});
}
void TRN::Engine::Broker::configure_recurrent_uniform(const unsigned int &id, const float &a, const float &b, const float &sparsity)
{
	auto processor = handle->manager->retrieve(id);
	processor->post([=]()
	{
		processor->configuring();
		TRN::Engine::Message<CONFIGURE_RECURRENT_UNIFORM> message;
		message.id = id;
		message.a = a;
		message.b = b;
		message.sparsity = sparsity;
		send(processor->get_rank(), message, [id]()
		{
			////PrintThread{} << "id " << id << " configure recurrent uniform acked" << std::endl;
		});
	});
}

void TRN::Engine::Broker::configure_recurrent_gaussian(const unsigned int &id, const float &mu, const float &sigma)
{
	auto processor = handle->manager->retrieve(id);
	processor->post([=]()
	{
		processor->configuring();
		TRN::Engine::Message<CONFIGURE_RECURRENT_GAUSSIAN> message;
		message.id = id;
		message.mu = mu;
		message.sigma = sigma;

		send(processor->get_rank(), message, [id]()
		{
			////PrintThread{} << "id " << id << " configure recurrent gaussian acked" << std::endl;
		});
	});
}
void TRN::Engine::Broker::configure_recurrent_custom(const unsigned int &id,
	const std::function<void(const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &request,
	std::function<void(const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &reply)
{
	auto processor = handle->manager->retrieve(id);
	std::unique_lock<std::mutex> lock(handle->functors);
	if (handle->recurrent.find(id) != handle->recurrent.end())
		throw std::invalid_argument("Simulator #" + std::to_string(id) + " already have a recurrent functor");
	handle->recurrent[id] = request;
	lock.unlock();
	
	reply = [this, id, processor](const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)
	{
		TRN::Engine::Message<RECURRENT_WEIGHTS> message;
		message.id = id;
		message.elements = weights;
		message.matrices = matrices;
		message.rows = rows;
		message.cols = cols;
		send(processor->get_rank(), message, [id]()
		{
			////PrintThread{} << "id " << id << " recurrent weights acked" << std::endl;
		});
	};

	processor->post([=]()
	{
		processor->configuring();

		TRN::Engine::Message<CONFIGURE_RECURRENT_CUSTOM> message;
		message.id = id;

		send(processor->get_rank(), message, [id]()
		{
			////PrintThread{} << "id " << id << " configure recurrent custom acked" << std::endl;
		});
	});
}

void TRN::Engine::Broker::configure_feedforward_uniform(const unsigned int &id, const float &a, const float &b, const float &sparsity)
{
	auto processor = handle->manager->retrieve(id);
	processor->post([=]()
	{
		processor->configuring();
		TRN::Engine::Message<CONFIGURE_FEEDFORWARD_UNIFORM> message;
		message.id = id;
		message.a = a;
		message.b = b;
		message.sparsity = sparsity;
		send(processor->get_rank(), message, [id]()
		{
			////PrintThread{} << "id " << id << " configure recurrent uniform acked" << std::endl;
		});
	});
}
void TRN::Engine::Broker::configure_feedforward_gaussian(const unsigned int &id, const float &mu, const float &sigma)
{
	auto processor = handle->manager->retrieve(id);
	processor->post([=]()
	{
		processor->configuring();
		TRN::Engine::Message<CONFIGURE_FEEDFORWARD_GAUSSIAN> message;
		message.id = id;
		message.mu = mu;
		message.sigma = sigma;

		send(processor->get_rank(), message, [id]()
		{
			////PrintThread{} << "id " << id << " configure feedforward gaussian acked" << std::endl;
		});
	});
}

void TRN::Engine::Broker::configure_feedforward_custom(const unsigned int &id,
	const std::function<void(const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &request,
	std::function<void(const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &reply)
{
	auto processor = handle->manager->retrieve(id);
	std::unique_lock<std::mutex> lock(handle->functors);
	if (handle->feedforward.find(id) != handle->feedforward.end())
		throw std::invalid_argument("Simulator #" + std::to_string(id) + " already have a feedforward functor");
	handle->feedforward[id] = request;
	lock.unlock();

	reply = [this, id, processor](const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)
	{
		TRN::Engine::Message<FEEDFORWARD_WEIGHTS> message;
		message.id = id;
		message.elements = weights;
		message.matrices = matrices;
		message.rows = rows;
		message.cols = cols;
		send(processor->get_rank(), message, [id]()
		{
			////PrintThread{} << "id " << id << " recurrent weights acked" << std::endl;
		});
	};

	processor->post([=]()
	{
		processor->configuring();

		TRN::Engine::Message<CONFIGURE_FEEDFORWARD_CUSTOM> message;
		message.id = id;

		send(processor->get_rank(), message, [id]()
		{
			////PrintThread{} << "id " << id << " configure feedforward custom acked" << std::endl;
		});
	});
}

std::shared_ptr<TRN::Engine::Broker> TRN::Engine::Broker::create(const std::shared_ptr<TRN::Engine::Communicator> &communicator)
{
	return std::make_shared<TRN::Engine::Broker>(communicator);
}