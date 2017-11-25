#include "stdafx.h"
#include "Broker_impl.h"

TRN::Engine::Broker::Broker(const std::shared_ptr<TRN::Engine::Communicator> &communicator) :
	handle(std::make_unique<Handle>())
{
	handle->communicator = communicator;
	handle->manager = TRN::Engine::Manager::create(communicator->size());
	handle->count = 0;
	handle->completed = false;
	handle->active = handle->manager->get_processors().size();
	handle->to_caller = TRN::Engine::Executor::create();
	
}
TRN::Engine::Broker::~Broker()
{
	handle->communicator->dispose();
	handle.reset();
}



void TRN::Engine::Broker::dispose()
{
	handle->manager->dispose();
	
}
void TRN::Engine::Broker::quit()
{
	dispose();
	TRN::Engine::Message<TRN::Engine::QUIT> message;

	handle->communicator->broadcast(message);
	join();
	handle->to_caller->post([=]()
	{

		//handle->from_caller->terminate();
		//callback_quit();
	});

}
void TRN::Engine::Broker::initialize()
{
	handle->to_caller->start();
	handle->manager->start();


}

void TRN::Engine::Broker::uninitialize()
{
	handle->manager->terminate();
	handle->to_caller->terminate();
	for (auto from_caller : handle->from_caller)
	{
		from_caller.second->terminate();
	}


}
void TRN::Engine::Broker::body()
{
	TRN::Engine::Tag tag = handle->communicator->probe(0);
	//PrintThread{} << "BROKER received tag " << tag << " " << __FUNCTION__ << std::endl;
	switch (tag)
	{
		case TRN::Engine::EXIT:
		{
			auto message = handle->communicator->receive<TRN::Engine::EXIT>(0);
			handle->to_caller->post([=]()
			{
				callback_exit(message.rank, message.terminated);
			});

			if (message.terminated)
			{
				handle->to_caller->post([=]()
				{
					callback_information("Worker #" + std::to_string(message.rank) + " exited");
				});
			}
			else
			{
				handle->to_caller->post([=]()
				{
					callback_information("Worker #" + std::to_string(message.rank) + " took into account quit request but did not quit yet");
				});
			}
			handle->active--;
			if (handle->active == 0)
			{
				handle->to_caller->post([=]()
				{
					callback_information("Broker stopped");
				});


				stop();
			}
	
	

		
		}
		break;
		case TRN::Engine::WORKER:
		{
			auto message = handle->communicator->receive<TRN::Engine::WORKER>(0);


			handle->manager->update_processor(message.rank, message.host, message.index, message.name);
			handle->to_caller->post([=]()
			{
	
				callback_processor(message.rank, message.host, message.index, message.name);
			
			});
		}
		break;
		case TRN::Engine::ACK:
		{
			auto message = handle->communicator->receive<TRN::Engine::ACK>(0);

			if (message.success)
			{
				std::unique_lock<std::recursive_mutex> lock(handle->ack);
				if (handle->on_ack_map.find(message.counter) == handle->on_ack_map.end())
					throw std::runtime_error("Ack functor for message #" + std::to_string(message.counter) + " is not setup");
				handle->on_ack_map[message.counter]();
				handle->on_ack_map.erase(message.counter);
			}
			else
			{
				std::cerr << "Simulator #" << message.id << "ACK : " << message.cause << std::endl;
			}
			handle->to_caller->post([=]()
			{
				callback_ack(message.id, message.counter, message.success, message.cause);
			});
		}
		break;
		case TRN::Engine::LOG_INFORMATION:
		{
			auto message = handle->communicator->receive<TRN::Engine::LOG_INFORMATION>(0);
			handle->to_caller->post([=]()
			{
				callback_information(message.message);
			});
		}
		break;
		case TRN::Engine::LOG_WARNING:
		{
			auto message = handle->communicator->receive<TRN::Engine::LOG_WARNING>(0);
			handle->to_caller->post([=]()
			{
				callback_warning(message.message);
			});
			std::cerr << "Simulator #" << message.id << "WARNING : " << message.message << std::endl;
		}
		break;
		case TRN::Engine::LOG_ERROR:
		{
			auto message = handle->communicator->receive<TRN::Engine::LOG_ERROR>(0);
			handle->to_caller->post([=]()
			{
				callback_error(message.message);
			});
			std::cerr << "Simulator #" << message.id << "ERROR : " << message.message << std::endl;
		}
		break;
		case TRN::Engine::POSITION:
		{
			auto message = handle->communicator->receive<TRN::Engine::POSITION>(0);
			handle->to_caller->post([=]()
			{
				callback_position(message.id, message.trial, message.evaluation, message.elements, message.rows, message.cols);
			});
		}
		break;
		case TRN::Engine::STIMULUS:
		{
			auto message = handle->communicator->receive<TRN::Engine::STIMULUS>(0);
			handle->to_caller->post([=]()
			{
				callback_stimulus(message.id, message.trial, message.evaluation, message.elements, message.rows, message.cols);
			});
		}
		break;
		case TRN::Engine::STATES:
		{
			auto message = handle->communicator->receive<TRN::Engine::STATES>(0);
			handle->to_caller->post([=]()
			{	
				callback_states(message.id, message.phase, message.label, message.batch, message.trial, message.evaluation, message.elements, message.rows, message.cols);
			});
		}
		break;
		case TRN::Engine::WEIGHTS:
		{
			auto message = handle->communicator->receive<TRN::Engine::WEIGHTS>(0);
			handle->to_caller->post([=]()
			{
				callback_weights(message.id, message.phase, message.label, message.batch, message.trial, message.elements, message.rows, message.cols);
			});
		}
		break;
		case TRN::Engine::PERFORMANCES:
		{
			auto message = handle->communicator->receive<TRN::Engine::PERFORMANCES>(0);
			handle->to_caller->post([=]()
			{
				callback_performances(message.id, message.trial, message.evaluation, message.phase, message.cycles_per_second, message.gflops_per_second);
			});
		}
		break;
		case TRN::Engine::SCHEDULER_CUSTOM:
		{
			auto message = handle->communicator->receive<TRN::Engine::SCHEDULER_CUSTOM>(0);
			handle->to_caller->post([=]()
			{
				callback_scheduler(message.id, message.seed, message.trial, message.elements, message.rows, message.cols, message.offsets, message.durations);
			});
		}
		break;
		case TRN::Engine::SCHEDULING:
		{
			auto message = handle->communicator->receive<TRN::Engine::SCHEDULING>(0);
			handle->to_caller->post([=]()
			{
				callback_scheduling(message.id, message.trial, message.offsets, message.durations);
			});
		}
		break;
		case TRN::Engine::MUTATOR_CUSTOM:
		{
			auto message = handle->communicator->receive<TRN::Engine::MUTATOR_CUSTOM>(0);
			handle->to_caller->post([=]()
			{
				callback_mutator(message.id, message.seed, message.trial, message.offsets, message.durations);
			});
		}
		break;
		case TRN::Engine::FEEDFORWARD_DIMENSIONS:
		{
			auto message = handle->communicator->receive<TRN::Engine::FEEDFORWARD_DIMENSIONS>(0);
			handle->to_caller->post([=]()
			{
				callback_feedforward(message.id, message.seed, message.matrices, message.rows, message.cols);
			});
		}
		break;
		case TRN::Engine::RECURRENT_DIMENSIONS:
		{	
			auto message = handle->communicator->receive<TRN::Engine::RECURRENT_DIMENSIONS>(0);
			handle->to_caller->post([=]()
			{		
				callback_recurrent(message.id, message.seed, message.matrices, message.rows, message.cols);
			});
		}
		break;
		case TRN::Engine::FEEDBACK_DIMENSIONS:
		{
			auto message = handle->communicator->receive<TRN::Engine::FEEDBACK_DIMENSIONS>(0);
			handle->to_caller->post([=]()
			{
				callback_feedback(message.id, message.seed, message.matrices, message.rows, message.cols);
			});
		}
		break;
		case TRN::Engine::READOUT_DIMENSIONS:
		{
			auto message = handle->communicator->receive<TRN::Engine::READOUT_DIMENSIONS>(0);
			handle->to_caller->post([=]()
			{
				callback_readout(message.id, message.seed, message.matrices, message.rows, message.cols);
			});
		}
		break;

		case TRN::Engine::ALLOCATED:
		{
			auto message = handle->communicator->receive<TRN::Engine::ALLOCATED>(0);
			auto processor = handle->manager->retrieve(message.id);
			/*retrieve_simulation(message.id)->post([=]()
			{*/
				processor->allocated();
		/*	});*/
		
			handle->to_caller->post([=]()
			{
				callback_allocated(message.id, processor->get_rank());
			});
		}
		break;
		case TRN::Engine::DEALLOCATED:
		{
			auto message = handle->communicator->receive<TRN::Engine::DEALLOCATED>(0);
			auto processor = handle->manager->retrieve(message.id);
			/*retrieve_simulation(message.id)->post([=]()
			{*/
		
			/*});*/
			handle->manager->deallocate(message.id);
				//	PrintThread{} << "id " << id << " deallocate ack" << std::endl;
			processor->deallocated();
			remove_simulation(message.id);
			handle->to_caller->post([=]()
			{
				callback_deallocated(message.id, processor->get_rank());
			});
		}
		break;


		case TRN::Engine::CONFIGURED:
		{
			auto message = handle->communicator->receive<TRN::Engine::CONFIGURED>(0);
			auto processor = handle->manager->retrieve(message.id);

/*			retrieve_simulation(message.id)->post([=]()
			{*/
				processor->configured();
		/*	});*/

	

			handle->to_caller->post([=]()
			{
				callback_configured(message.id);
			});
		}
		break;

		case TRN::Engine::TRAINED:
		{
			auto message = handle->communicator->receive<TRN::Engine::TRAINED>(0);
			auto processor = handle->manager->retrieve(message.id);
			/*retrieve_simulation(message.id)->post([=]()
			{*/
				processor->trained();
				processor->set_t1(std::clock());
			/*});*/

			handle->to_caller->post([=]()
			{
				callback_trained(message.id);
			});
		}
		break;
		case TRN::Engine::PRIMED:
		{
			auto message = handle->communicator->receive<TRN::Engine::PRIMED>(0);
			auto processor = handle->manager->retrieve(message.id);
			/*retrieve_simulation(message.id)->post([=]()
			{*/
				processor->primed();
			/*});*/
	
			handle->to_caller->post([=]()
			{
				callback_primed(message.id);
			});
		}
		break;
		case TRN::Engine::TESTED:
		{
			auto message = handle->communicator->receive<TRN::Engine::TESTED>(0);
			auto processor = handle->manager->retrieve(message.id);
		/*	retrieve_simulation(message.id)->post([=]()
			{*/
				processor->tested();
			/*});*/
	
			handle->to_caller->post([=]()
			{
				callback_tested(message.id);
			});
		}
		break;
		case TRN::Engine::MEASUREMENT_READOUT_MEAN_SQUARE_ERROR:
		{
			auto message = handle->communicator->receive<TRN::Engine::MEASUREMENT_READOUT_MEAN_SQUARE_ERROR>(0);
			handle->to_caller->post([=]()
			{
				callback_measurement_readout_mean_square_error(message.id, message.trial, message.evaluation, message.elements, message.rows, message.cols);
			});
		}
		break;
		case TRN::Engine::MEASUREMENT_READOUT_FRECHET_DISTANCE:
		{
			auto message = handle->communicator->receive<TRN::Engine::MEASUREMENT_READOUT_FRECHET_DISTANCE>(0);
			handle->to_caller->post([=]()
			{
				callback_measurement_readout_frechet_distance(message.id, message.trial, message.evaluation, message.elements, message.rows, message.cols);
			});
		}
		break;
		case TRN::Engine::MEASUREMENT_READOUT_CUSTOM:
		{
			auto message = handle->communicator->receive<TRN::Engine::MEASUREMENT_READOUT_CUSTOM>(0);
			handle->to_caller->post([=]()
			{
				callback_measurement_readout_custom(message.id, message.trial, message.evaluation, message.primed, message.elements, message.expected, message.preamble, message.matrices, message.rows, message.cols);
			});
		}
		break;
		case TRN::Engine::MEASUREMENT_POSITION_MEAN_SQUARE_ERROR:
		{
			auto message = handle->communicator->receive<TRN::Engine::MEASUREMENT_POSITION_MEAN_SQUARE_ERROR>(0);
			handle->to_caller->post([=]()
			{
				callback_measurement_position_mean_square_error(message.id, message.trial, message.evaluation, message.elements, message.rows, message.cols);
			});
		}
		break;
		case TRN::Engine::MEASUREMENT_POSITION_FRECHET_DISTANCE:
		{
			auto message = handle->communicator->receive<TRN::Engine::MEASUREMENT_POSITION_FRECHET_DISTANCE>(0);
			handle->to_caller->post([=]()
			{
				callback_measurement_position_frechet_distance(message.id, message.trial, message.evaluation, message.elements, message.rows, message.cols);
			});
		}
		break;
		case TRN::Engine::MEASUREMENT_POSITION_CUSTOM:
		{
			auto message = handle->communicator->receive<TRN::Engine::MEASUREMENT_POSITION_CUSTOM>(0);
			handle->to_caller->post([=]()
			{
				callback_measurement_position_custom(message.id, message.trial, message.evaluation, message.primed, message.elements, message.expected, message.preamble, message.matrices, message.rows, message.cols);
			});
		}
		break;


		default:
			throw std::invalid_argument("Unexpected tag " + std::to_string(tag));
	}
}



void TRN::Engine::Broker::append_simulation(const unsigned long long &id)
{
	std::unique_lock<std::mutex> lock(handle->mutex);
	if (handle->simulations.find(id) != handle->simulations.end())
		throw std::logic_error("Simulation #" + std::to_string(id) + " is already allocated");
	handle->simulations.insert(id);
	handle->from_caller[id] = TRN::Engine::Executor::create();
	handle->from_caller[id]->start();
}

std::shared_ptr<TRN::Engine::Executor> TRN::Engine::Broker::retrieve_simulation(const unsigned long long &id)
{
	std::unique_lock<std::mutex> lock(handle->mutex);

	return handle->from_caller[id];
}


void TRN::Engine::Broker::remove_simulation(const unsigned long long &id)
{
	std::unique_lock<std::mutex> lock(handle->mutex);;
	if (handle->simulations.find(id) == handle->simulations.end())
		throw std::logic_error("Simulation #" + std::to_string(id) + " is not allocated");
	handle->simulations.erase(id);
	handle->from_caller[id]->terminate();
	handle->from_caller.erase(id);
	//	PrintThread{} << "id " << id << " deallocate DONE" << std::endl;
}


std::size_t TRN::Engine::Broker::generate_number()
{
	std::unique_lock<std::mutex> lock(handle->counter);
	std::size_t counter = handle->count;
	handle->count++;
	return counter;
}

template<TRN::Engine::Tag tag>
void TRN::Engine::Broker::send(const int &rank, TRN::Engine::Message<tag> &message, const std::function<void()> &functor)
{
	message.counter = generate_number();

	std::unique_lock<std::recursive_mutex> lock(handle->ack);
	////PrintThread{} << "acquiring functor lock for id " << message.id << std::endl;

	if (handle->on_ack_map.find(message.counter) != handle->on_ack_map.end())
		throw std::runtime_error("Ack functor for message #" + std::to_string(message.counter) + " is already setup");
	handle->on_ack_map[message.counter] = functor;

	handle->communicator->send(message, rank);
}



/*void TRN::Engine::Broker::ready(const unsigned long long &id)
{
	retrieve_simulation(id)->post([=]()
	{
		auto processor = handle->manager->retrieve(id);
		//processor->configured();

		processor->post([=]()
		{	
		
			TRN::Engine::Message<READY> message;
			message.id = id;
			
			send(processor->get_rank(), message, [=]()
			{
			
							////PrintThread{} << "id " << id << " allocate ack" << std::endl;
			});
		});
	});
}*/
void TRN::Engine::Broker::allocate(const unsigned long long &id)
{
	append_simulation(id);

	retrieve_simulation(id)->post([=]() 
	{
		auto processor = handle->manager->allocate(id);
		processor->allocating();
		processor->post([=]()
		{
			TRN::Engine::Message<ALLOCATE> message;
			message.id = id;
		
			send(processor->get_rank(), message, [=]()
			{

				////PrintThread{} << "id " << id << " allocate ack" << std::endl;
			});
		});
	});
}
void TRN::Engine::Broker::deallocate(const unsigned long long &id)
{
	retrieve_simulation(id)->post([=]()
	{
		//PrintThread{} << "entering " << id << " " << __FUNCTION__ << std::endl;
		auto processor = handle->manager->retrieve(id);

		processor->deallocating();
		processor->post([=]()
		{	
			TRN::Engine::Message<DEALLOCATE> message;
			message.id = id;

			send(processor->get_rank(), message, [=]()
			{
	
				//	PrintThread{} << "id " << id << " manager deallocate" << std::endl;
			});
		});
		//PrintThread{} << "exiting " << id << " " << __FUNCTION__ << std::endl;
	});
}
void TRN::Engine::Broker::train(const unsigned long long &id, const std::string &label, const std::string &incoming, const std::string &expected)
{
	retrieve_simulation(id)->post([=]()
	{
		auto processor = handle->manager->retrieve(id);
		processor->training();
		processor->set_t0(std::clock());
		processor->post([=]()
		{
		
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
	});
}
void TRN::Engine::Broker::test(const unsigned long long &id, const std::string &label, const std::string &incoming, const std::string &expected, const unsigned int &preamble, const bool &autonomous, const unsigned int &supplementary_generations)
{
	retrieve_simulation(id)->post([=]()
	{
		auto processor = handle->manager->retrieve(id);
		processor->testing();
		processor->post([=]()
		{
			
			TRN::Engine::Message<TEST> message;
			message.id = id;
			message.label = label;
			message.incoming = incoming;
			message.expected = expected;
			message.preamble = preamble;
			message.autonomous = autonomous;
			message.supplementary_generations = supplementary_generations;
			send(processor->get_rank(), message, [this, id]()
			{
				////PrintThread{} << "id " << id << " test ack" << std::endl;
			});
		});
	});
}
void TRN::Engine::Broker::declare_sequence(const unsigned long long &id, const std::string &label, const std::string &tag, const std::vector<float> &sequence, const std::size_t &observations)
{
	retrieve_simulation(id)->post([=]()
	{
		auto processor = handle->manager->retrieve(id);
		processor->declare();
		processor->post([=]()
		{
	
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
	});
}
void TRN::Engine::Broker::declare_set(const unsigned long long &id, const std::string &label, const std::string &tag, const std::vector<std::string> &labels)
{
	retrieve_simulation(id)->post([=]()
	{
		auto processor = handle->manager->retrieve(id);
		processor->declare();
		processor->post([=]()
		{

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
	});
}
void TRN::Engine::Broker::setup_states(const unsigned long long &id, const bool &train, const bool &prime, const bool &generate)
{
	retrieve_simulation(id)->post([=]()
	{
		auto processor = handle->manager->retrieve(id);
		processor->configuring();
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
	});
}
void TRN::Engine::Broker::setup_weights(const unsigned long long &id, const bool &initalization, const bool &train)
{
	retrieve_simulation(id)->post([=]()
	{
		auto processor = handle->manager->retrieve(id);
		processor->configuring();
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
	});
}
void TRN::Engine::Broker::setup_performances(const unsigned long long &id, const bool &train, const bool &prime, const bool &generate)
{
	retrieve_simulation(id)->post([=]()
	{	
		auto processor = handle->manager->retrieve(id);
		processor->configuring();
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
	});
}
void TRN::Engine::Broker::setup_scheduling(const unsigned long long &id)
{
	retrieve_simulation(id)->post([=]()
	{
		auto processor = handle->manager->retrieve(id);
		processor->configuring();
		processor->post([=]()
		{
		
			TRN::Engine::Message<SETUP_SCHEDULING> message;
			message.id = id;

			send(processor->get_rank(), message, [id]()
			{
				////PrintThread{} << "id " << id << " performances setup acked" << std::endl;
			});
		});
	});
}
void TRN::Engine::Broker::configure_begin(const unsigned long long &id)
{
	retrieve_simulation(id)->post([=]()
	{
		auto processor = handle->manager->retrieve(id);
		processor->configuring();
		processor->post([=]()
		{
			TRN::Engine::Message<CONFIGURE_BEGIN> message;
			message.id = id;
			send(processor->get_rank(), message, [id]()
			{
				////PrintThread{} << "id " << id << " configure begin acked" << std::endl;
			});
		});
	});
}


void TRN::Engine::Broker::configure_end(const unsigned long long &id)
{
	auto processor = handle->manager->retrieve(id);
	retrieve_simulation(id)->post([=]()
	{
		
		processor->configuring();
		processor->post([=]()
		{
		
			TRN::Engine::Message<CONFIGURE_END> message;
			message.id = id;
			send(processor->get_rank(), message, [=]()
			{
				
				////PrintThread{} << "id " << id << " configure end acked" << std::endl;
			});
		});
	});
	//processor->ready();
}
void TRN::Engine::Broker::configure_measurement_readout_mean_square_error(const unsigned long long &id, const std::size_t &batch_size)
{
    retrieve_simulation(id)->post([=]()
	{
		auto processor = handle->manager->retrieve(id);
		processor->configuring();
		processor->post([=]()
		{
		
			TRN::Engine::Message<CONFIGURE_MEASUREMENT_READOUT_MEAN_SQUARE_ERROR> message;
			message.id = id;
			message.batch_size = batch_size;
			send(processor->get_rank(), message, [this, id]()
			{
				////PrintThread{} << "id " << id << " configure_measurement_readout_mean_square_error end acked" << std::endl;
			});
		});
 });
}
void TRN::Engine::Broker::configure_measurement_readout_frechet_distance(const unsigned long long &id, const std::size_t &batch_size)
{
	retrieve_simulation(id)->post([=]()
	{
		auto processor = handle->manager->retrieve(id);
		processor->configuring();
		processor->post([=]()
		{
		
			TRN::Engine::Message<CONFIGURE_MEASUREMENT_READOUT_FRECHET_DISTANCE> message;
			message.id = id;
			message.batch_size = batch_size;
			send(processor->get_rank(), message, [this, id]()
			{
				////PrintThread{} << "id " << id << " configure_measurement_readout_frechet_distance end acked" << std::endl;
			});
		});
	});
}
void TRN::Engine::Broker::configure_measurement_readout_custom(const unsigned long long &id, const std::size_t &batch_size)
{
	retrieve_simulation(id)->post([=]()
	{
		auto processor = handle->manager->retrieve(id);
		processor->configuring();
		processor->post([=]()
		{
		
			TRN::Engine::Message<CONFIGURE_MEASUREMENT_READOUT_CUSTOM> message;
			message.id = id;
			message.batch_size = batch_size;
			send(processor->get_rank(), message, [this, id]()
			{
				////PrintThread{} << "id " << id << " configure_measurement_readout_custom end acked" << std::endl;
			});
		});
	});
}
void TRN::Engine::Broker::configure_measurement_position_mean_square_error(const unsigned long long &id, const std::size_t &batch_size)
{
	retrieve_simulation(id)->post([=]()
	{
		auto processor = handle->manager->retrieve(id);
		processor->configuring();
		processor->post([=]()
		{
	
			TRN::Engine::Message<CONFIGURE_MEASUREMENT_POSITION_MEAN_SQUARE_ERROR> message;
			message.id = id;
			message.batch_size = batch_size;
			send(processor->get_rank(), message, [this, id]()
			{
				////PrintThread{} << "id " << id << " configure_measurement_position_mean_square_error end acked" << std::endl;
			});
		});
	});
}
void TRN::Engine::Broker::configure_measurement_position_frechet_distance(const unsigned long long &id, const std::size_t &batch_size)
{
	retrieve_simulation(id)->post([=]()
	{
		auto processor = handle->manager->retrieve(id);
		processor->configuring();
		processor->post([=]()
		{
		

			TRN::Engine::Message<CONFIGURE_MEASUREMENT_POSITION_FRECHET_DISTANCE> message;
			message.id = id;
			message.batch_size = batch_size;
			send(processor->get_rank(), message, [this, id]()
			{
				////PrintThread{} << "id " << id << " configure_measurement_position_frechet_distance end acked" << std::endl;
			});
		});
	});
}
void TRN::Engine::Broker::configure_measurement_position_custom(const unsigned long long &id, const std::size_t &batch_size)
{
	retrieve_simulation(id)->post([=]()
	{
		auto processor = handle->manager->retrieve(id);
		processor->configuring();
		processor->post([=]()
		{
		
			TRN::Engine::Message<CONFIGURE_MEASUREMENT_POSITION_CUSTOM> message;
			message.id = id;
			message.batch_size = batch_size;
			send(processor->get_rank(), message, [this, id]()
			{
				////PrintThread{} << "id " << id << " configure_measurement_position_custom end acked" << std::endl;
			});
		});
	});
}
void TRN::Engine::Broker::configure_reservoir_widrow_hoff(const unsigned long long &id, const std::size_t &stimulus_size, const std::size_t &prediction_size, const std::size_t &reservoir_size, const float &leak_rate, const float &initial_state_scale, const float &learning_rate, const unsigned long &seed, const std::size_t &batch_size)
{
	retrieve_simulation(id)->post([=]()
	{
		auto processor = handle->manager->retrieve(id);
		processor->configuring();
		processor->post([=]()
		{
	
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
	});
}
void TRN::Engine::Broker::configure_loop_copy(const unsigned long long &id, const std::size_t &batch_size, const std::size_t &stimulus_size)
{
	retrieve_simulation(id)->post([=]()
	{
		auto processor = handle->manager->retrieve(id);
		processor->configuring();
		processor->post([=]()
		{
		
			TRN::Engine::Message<CONFIGURE_LOOP_COPY> message;
			message.id = id;
			message.stimulus_size = stimulus_size;
			message.batch_size = batch_size;
			send(processor->get_rank(), message, [id]()
			{
				////PrintThread{} << "id " << id << " configure loop copy acked" << std::endl;
			});
		});
	});
}
void TRN::Engine::Broker::configure_loop_spatial_filter(const unsigned long long &id, const std::size_t &batch_size, const std::size_t &stimulus_size, const unsigned long &seed,
	const std::size_t &rows, const std::size_t &cols,
	const std::pair<float, float> &x, const std::pair<float, float> &y,
	const std::vector<float> &response,
	const float &sigma,
	const float &radius,
	const float &scale,
	const std::string &tag)
{
	retrieve_simulation(id)->post([=]()
	{
		auto processor = handle->manager->retrieve(id);
		
		processor->configuring();
		processor->post([=]()
		{
			
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
	});
	auto x_resolution = (x.second - x.first) / cols;
	auto y_resolution = (y.second - y.first) / rows;
	auto resolution = x_resolution * y_resolution;
	
	int order = 0;
	while (resolution < 1.0f && order <= 2)
	{
		resolution *= 100.0f;
		order++;
	}
	resolution = sqrtf(resolution);
	std::string unit;
	switch (order)
	{
		case 0 :
			unit = "meter";
			break;
		case 1:
			unit = "decimeter";
			break;
		case 2:
			unit = "centimeter";
			break;
		case 3:
			unit = "millimeter";
			break;
	}
	handle->to_caller->post([=]()
	{
		callback_information("Spatial filter resolution : " + std::to_string(resolution) + " " + unit + " square per stencil element");
	});
}
void TRN::Engine::Broker::configure_loop_custom(const unsigned long long &id, const std::size_t &batch_size, const std::size_t &stimulus_size)
{
	retrieve_simulation(id)->post([=]()
	{
		auto processor = handle->manager->retrieve(id);
		processor->configuring();
		processor->post([=]()
		{
	
			TRN::Engine::Message<CONFIGURE_LOOP_CUSTOM> message;

			message.id = id;
			message.batch_size = batch_size;
			message.stimulus_size = stimulus_size;
			send(processor->get_rank(), message, [id]()
			{
				////PrintThread{} << "id " << id << " configure loop custom acked" << std::endl;
			});
		});
	});
}
void TRN::Engine::Broker::configure_scheduler_tiled(const unsigned long long &id, const unsigned int &epochs)
{
	retrieve_simulation(id)->post([=]()
	{
	 auto processor = handle->manager->retrieve(id);
	 processor->configuring();
		processor->post([=]()
		{
		
			TRN::Engine::Message<CONFIGURE_SCHEDULER_TILED> message;
			message.id = id;
			message.epochs = epochs;

			send(processor->get_rank(), message, [id]()
			{
				////PrintThread{} << "id " << id << " configure scheduler tiled acked" << std::endl;
			});
		});
	});
}
void TRN::Engine::Broker::configure_scheduler_snippets(const unsigned long long &id, const unsigned long &seed, const unsigned int &snippets_size, const unsigned int &time_budget,  const std::string &tag)
{
	retrieve_simulation(id)->post([=]()
	{
		 auto processor = handle->manager->retrieve(id);
		 processor->configuring();
		processor->post([=]()
		{
		
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
	});
}
void TRN::Engine::Broker::configure_scheduler_custom(const unsigned long long &id, const unsigned long &seed, const std::string &tag)
{
	retrieve_simulation(id)->post([=]()
	{
		 auto processor = handle->manager->retrieve(id);
		 processor->configuring();
		processor->post([=]()
		{



			TRN::Engine::Message<CONFIGURE_SCHEDULER_CUSTOM> message;
			message.id = id;
			message.seed = seed;

			send(processor->get_rank(), message, [id]()
			{
				////PrintThread{} << "id " << id << " configure scheduler custom acked" << std::endl;
			});
		});
	});
}
void TRN::Engine::Broker::configure_mutator_shuffle(const unsigned long long &id,const unsigned long &seed)
{
	retrieve_simulation(id)->post([=]()
	{
		auto processor = handle->manager->retrieve(id);
		processor->configuring();
		processor->post([=]()
		{
		
			TRN::Engine::Message<CONFIGURE_MUTATOR_SHUFFLE> message;
			message.id = id;
			message.seed = seed;
			send(processor->get_rank(), message, [id]()
			{
				////PrintThread{} << "id " << id << " configure scheduler snippets acked" << std::endl;
			});
		});
	});
}
void TRN::Engine::Broker::configure_mutator_reverse(const unsigned long long &id, const unsigned long &seed, const float &rate, const std::size_t &size)
{
	retrieve_simulation(id)->post([=]()
	{
		auto processor = handle->manager->retrieve(id);
		processor->configuring();
		processor->post([=]()
		{
	
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
	});
}
void TRN::Engine::Broker::configure_mutator_punch(const unsigned long long &id, const unsigned long &seed, const float &rate, const std::size_t &size, const std::size_t &counter)
{
	retrieve_simulation(id)->post([=]()
	{
		auto processor = handle->manager->retrieve(id);
		processor->configuring();
		processor->post([=]()
		{
	
			TRN::Engine::Message<CONFIGURE_MUTATOR_PUNCH> message;
			message.id = id;
			message.seed = seed;
			message.rate = rate;
			message.size = size;
			message.repetition = counter;
			send(processor->get_rank(), message, [id]()
			{
				////PrintThread{} << "id " << id << " configure scheduler snippets acked" << std::endl;
			});
		});
	});
}
void TRN::Engine::Broker::configure_mutator_custom(const unsigned long long &id, const unsigned long &seed)
{
	retrieve_simulation(id)->post([=]()
	{
		auto processor = handle->manager->retrieve(id);
		processor->configuring();
		processor->post([=]()
		{
	

			TRN::Engine::Message<CONFIGURE_MUTATOR_CUSTOM> message;
			message.id = id;
			message.seed = seed;
			send(processor->get_rank(), message, [id]()
			{
				////PrintThread{} << "id " << id << " configure scheduler custom acked" << std::endl;
			});
		});
	});
}
void TRN::Engine::Broker::configure_readout_uniform(const unsigned long long &id, const float &a, const float &b, const float &sparsity)
{
	retrieve_simulation(id)->post([=]()
	{
		auto processor = handle->manager->retrieve(id);
		processor->configuring();
		processor->post([=]()
		{
		
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
	});
}
void TRN::Engine::Broker::configure_readout_gaussian(const unsigned long long &id, const float &mu, const float &sigma)
{
	retrieve_simulation(id)->post([=]()
	{
		auto processor = handle->manager->retrieve(id);
		processor->configuring();
		processor->post([=]()
		{

			TRN::Engine::Message<CONFIGURE_READOUT_GAUSSIAN> message;
			message.id = id;
			message.mu = mu;
			message.sigma = sigma;

			send(processor->get_rank(), message, [id]()
			{
				////PrintThread{} << "id " << id << " configure readout gaussian acked" << std::endl;
			});
		});
	});
}
void TRN::Engine::Broker::configure_readout_custom(const unsigned long long &id)
{
	retrieve_simulation(id)->post([=]()
	{
		auto processor = handle->manager->retrieve(id);
		processor->configuring();
		processor->post([=]()
		{
	

			TRN::Engine::Message<CONFIGURE_READOUT_CUSTOM> message;
			message.id = id;

			send(processor->get_rank(), message, [id]()
			{
				////PrintThread{} << "id " << id << " configure readout custom acked" << std::endl;
			});
		});
	});
}
void TRN::Engine::Broker::configure_feedback_uniform(const unsigned long long &id, const float &a, const float &b, const float &sparsity)
{
	retrieve_simulation(id)->post([=]()
	{
		auto processor = handle->manager->retrieve(id);
		processor->configuring();
		processor->post([=]()
		{

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
	});
}
void TRN::Engine::Broker::configure_feedback_gaussian(const unsigned long long &id, const float &mu, const float &sigma)
{
 retrieve_simulation(id)->post([=]()
	{
		auto processor = handle->manager->retrieve(id);
		processor->configuring();
		processor->post([=]()
		{
		
			TRN::Engine::Message<CONFIGURE_FEEDBACK_GAUSSIAN> message;
			message.id = id;
			message.mu = mu;
			message.sigma = sigma;

			send(processor->get_rank(), message, [id]()
			{
				////PrintThread{} << "id " << id << " configure feedback gaussian acked" << std::endl;
			});
		});
	});
}
void TRN::Engine::Broker::configure_feedback_custom(const unsigned long long &id)
{
	retrieve_simulation(id)->post([=]()
	{
	 auto processor = handle->manager->retrieve(id);
	 processor->configuring();
		processor->post([=]()
		{
		
			TRN::Engine::Message<CONFIGURE_FEEDBACK_CUSTOM> message;
			message.id = id;

			send(processor->get_rank(), message, [id]()
			{
				////PrintThread{} << "id " << id << " configure feedback custom acked" << std::endl;
			});
		});
	});
}
void TRN::Engine::Broker::configure_recurrent_uniform(const unsigned long long &id, const float &a, const float &b, const float &sparsity)
{
	retrieve_simulation(id)->post([=]()
	{
		auto processor = handle->manager->retrieve(id);
		processor->configuring();
		processor->post([=]()
		{
			
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
	});
}
void TRN::Engine::Broker::configure_recurrent_gaussian(const unsigned long long &id, const float &mu, const float &sigma)
{
	retrieve_simulation(id)->post([=]()
	{
		auto processor = handle->manager->retrieve(id);
		processor->configuring();
		processor->post([=]()
		{
			
			TRN::Engine::Message<CONFIGURE_RECURRENT_GAUSSIAN> message;
			message.id = id;
			message.mu = mu;
			message.sigma = sigma;

			send(processor->get_rank(), message, [id]()
			{
				////PrintThread{} << "id " << id << " configure recurrent gaussian acked" << std::endl;
			});
		});
	});
}
void TRN::Engine::Broker::configure_recurrent_custom(const unsigned long long &id)
{
	retrieve_simulation(id)->post([=]()
	{
		auto processor = handle->manager->retrieve(id);
		processor->configuring();
		processor->post([=]()
		{
			
			TRN::Engine::Message<CONFIGURE_RECURRENT_CUSTOM> message;
			message.id = id;

			send(processor->get_rank(), message, [id]()
			{
				////PrintThread{} << "id " << id << " configure recurrent custom acked" << std::endl;
			});
		});
	 });
}
void TRN::Engine::Broker::configure_feedforward_uniform(const unsigned long long &id, const float &a, const float &b, const float &sparsity)
{
	retrieve_simulation(id)->post([=]()
	{
		auto processor = handle->manager->retrieve(id);
		processor->configuring();
		processor->post([=]()
		{
		
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
	});
}
void TRN::Engine::Broker::configure_feedforward_gaussian(const unsigned long long &id, const float &mu, const float &sigma)
{
	retrieve_simulation(id)->post([=]()
	{
		auto processor = handle->manager->retrieve(id);
		processor->configuring();
		processor->post([=]()
		{
			
			TRN::Engine::Message<CONFIGURE_FEEDFORWARD_GAUSSIAN> message;
			message.id = id;
			message.mu = mu;
			message.sigma = sigma;

			send(processor->get_rank(), message, [id]()
			{
				////PrintThread{} << "id " << id << " configure feedforward gaussian acked" << std::endl;
			});
		});
	});
}
void TRN::Engine::Broker::configure_feedforward_custom(const unsigned long long &id)
{
	retrieve_simulation(id)->post([=]()
	{
		auto processor = handle->manager->retrieve(id);
		processor->configuring();
		processor->post([=]()
		{

			TRN::Engine::Message<CONFIGURE_FEEDFORWARD_CUSTOM> message;
			message.id = id;

			send(processor->get_rank(), message, [id]()
			{
				////PrintThread{} << "id " << id << " configure feedforward custom acked" << std::endl;
			});
		});
	});
}
void TRN::Engine::Broker::notify_stimulus(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &stimulus, const std::size_t &rows, const std::size_t &cols)
{
	/*retrieve_simulation(id)->post([=]()
	{*/
		auto processor = handle->manager->retrieve(id);
		/*processor->post([=]()
		{*/
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
		/*});
	});*/
}
void TRN::Engine::Broker::notify_position(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols)
{
	/*retrieve_simulation(id)->post([=]()
	{*/
		auto processor = handle->manager->retrieve(id);
		/*processor->post([=]()
		{*/
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
		/*});
	});*/
}
void TRN::Engine::Broker::notify_scheduler(const unsigned long long &id, const std::size_t &trial, const std::vector<int> &offsets, const std::vector<int> &durations)
{
	/*retrieve_simulation(id)->post([=]()
	{*/
		auto processor = handle->manager->retrieve(id);
		/*processor->post([=]()
		{*/
			TRN::Engine::Message<SCHEDULING> message;
			message.id = id;

			message.is_from_mutator = false;
			message.offsets = offsets;
			message.durations = durations;

			send(processor->get_rank(), message, [id]()
			{
				////PrintThread{} << "id " << id << " scheduling acked" << std::endl;
			});
		/*});
	});*/
}
void TRN::Engine::Broker::notify_mutator(const unsigned long long &id, const std::size_t &trial, const std::vector<int> &offsets, const std::vector<int> &durations)
{
	/*retrieve_simulation(id)->post([=]()
	{*/
		auto processor = handle->manager->retrieve(id);
		/*processor->post([=]()
		{*/
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
	/*	});
	});*/
}
void TRN::Engine::Broker::notify_feedforward(const unsigned long long &id, const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)
{
	/*retrieve_simulation(id)->post([=]()
	{*/
		auto processor = handle->manager->retrieve(id);
		/*processor->configuring();
		processor->post([=]()
		{
		*/
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
	/*	});
	});*/
}
void TRN::Engine::Broker::notify_feedback(const unsigned long long &id, const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)
{
	/*retrieve_simulation(id)->post([=]()
	{*/
		auto processor = handle->manager->retrieve(id);
		/*processor->configuring();
		processor->post([=]()
		{*/
	
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
	/*	});
	});*/
}
void TRN::Engine::Broker::notify_readout(const unsigned long long &id, const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)
{
	/*retrieve_simulation(id)->post([=]()
	{*/
		auto processor = handle->manager->retrieve(id);
		/*processor->configuring();
		processor->post([=]()
		{*/
	
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
	/*	});
	});*/
}
void TRN::Engine::Broker::notify_recurrent(const unsigned long long &id, const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)
{
	/*retrieve_simulation(id)->post([=]()
	{*/
		auto processor = handle->manager->retrieve(id);
		/*processor->configuring();
		processor->post([=]()
		{
		*/
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
		/*});
	});*/
}