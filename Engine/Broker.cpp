#include "stdafx.h"
#include "Broker_impl.h"
#include "Helper/Logger.h"




TRN::Engine::Broker::Broker(const std::shared_ptr<TRN::Engine::Communicator> &communicator) :
	handle(std::make_unique<Handle>())
{
	TRACE_LOGGER;
	handle->communicator = communicator;
	handle->manager = TRN::Engine::Manager::create(communicator->size());
	handle->count = 0;
	handle->completed = false;
	handle->active = handle->manager->get_processors().size();
	handle->to_caller = TRN::Engine::Executor::create();
	
}
TRN::Engine::Broker::~Broker() noexcept(false)
{
	TRACE_LOGGER;
	handle.reset();
}

void TRN::Engine::Broker::dispose()
{
	TRACE_LOGGER;
	handle->manager->dispose();
	handle->communicator->dispose();
}
void TRN::Engine::Broker::quit()
{
	TRACE_LOGGER;
	handle->manager->dispose();
	TRN::Engine::Message<TRN::Engine::QUIT> message;
	message.terminate = true;
	handle->communicator->broadcast(message);
	handle->communicator->dispose();
	join();
}
void TRN::Engine::Broker::initialize()
{
	TRACE_LOGGER;
	handle->to_caller->start();
	handle->manager->start();
}

void TRN::Engine::Broker::uninitialize()
{
	TRACE_LOGGER;
	handle->manager->terminate();
	handle->to_caller->terminate();
	for (auto from_caller : handle->from_caller)
	{
		from_caller.second->terminate();
	}


}
void TRN::Engine::Broker::body()
{
	TRACE_LOGGER;
	auto probed = handle->communicator->probe(0);
	if (!probed)
	{
		DEBUG_LOGGER << "BROKER encountered an invalid probe.Stopping RX task";
		stop();
	}
	else
	{
		auto tag = *probed;
		TRACE_LOGGER <<   "BROKER received tag " << tag;
		switch (tag)
		{
			case TRN::Engine::EXIT:
			{
				auto message = handle->communicator->receive<TRN::Engine::EXIT>(0);

				handle->to_caller->post([=]()
				{
					callback_exit(message.number, message.rank);
				});
			}
			break;

			case TRN::Engine::TERMINATED:
			{
				auto message = handle->communicator->receive<TRN::Engine::TERMINATED>(0);

				handle->to_caller->post([=]()
				{
					callback_terminated(message.rank);
				});

				handle->active--;

				if (handle->active == 0)
				{
					stop();
				}
			}
			break;
			case TRN::Engine::CACHED:
			{
				auto message = handle->communicator->receive<TRN::Engine::CACHED>(0);
				if (!message.checksums.empty())
				{


					std::unique_lock<std::mutex> guard(handle->cache_mutex);
					auto ranks = handle->host_ranks[handle->rank_host[message.rank]];
					for (auto rank : ranks)
						handle->cached[rank].insert(message.checksums.begin(), message.checksums.end());
				}
			}
			break;
			case TRN::Engine::WORKER:
			{
				auto message = handle->communicator->receive<TRN::Engine::WORKER>(0);

				handle->rank_host[message.rank] = message.host;
				handle->host_ranks[message.host].push_back(message.rank);

				handle->manager->update_processor(message.rank, message.host, message.index, message.name);
				handle->to_caller->post([=]()
				{

					callback_processor(message.rank, message.host, message.index, message.name);

				});
			}
			break;

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
				ERROR_LOGGER << "Simulator #" << message.simulation_id << message.message;
			}
			break;
			case TRN::Engine::LOG_ERROR:
			{
				auto message = handle->communicator->receive<TRN::Engine::LOG_ERROR>(0);
				handle->to_caller->post([=]()
				{
					callback_error(message.message);
				});
				ERROR_LOGGER << "Simulator #" << message.simulation_id << "ERROR : " << message.message;
			}
			break;
			case TRN::Engine::POSITION:
			{
				auto message = handle->communicator->receive<TRN::Engine::POSITION>(0);
				handle->to_caller->post([=]()
				{
					callback_position(message.simulation_id, message.evaluation_id, message.elements, message.rows, message.cols);
				});
			}
			break;
			case TRN::Engine::STIMULUS:
			{
				auto message = handle->communicator->receive<TRN::Engine::STIMULUS>(0);
				handle->to_caller->post([=]()
				{
					callback_stimulus(message.simulation_id, message.evaluation_id, message.elements, message.rows, message.cols);
				});
			}
			break;
			case TRN::Engine::STATES:
			{
				auto message = handle->communicator->receive<TRN::Engine::STATES>(0);
				handle->to_caller->post([=]()
				{
					callback_states(message.simulation_id, message.evaluation_id, message.phase, message.label, message.batch, message.elements, message.rows, message.cols);
				});
			}
			break;
			case TRN::Engine::WEIGHTS:
			{
				auto message = handle->communicator->receive<TRN::Engine::WEIGHTS>(0);
				handle->to_caller->post([=]()
				{
					callback_weights(message.simulation_id, message.evaluation_id, message.phase, message.label, message.batch,  message.elements, message.rows, message.cols);
				});
			}
			break;
			case TRN::Engine::PERFORMANCES:
			{
				auto message = handle->communicator->receive<TRN::Engine::PERFORMANCES>(0);
				handle->to_caller->post([=]()
				{
					callback_performances(message.simulation_id, message.evaluation_id, message.phase, message.cycles_per_second, message.gflops_per_second);
				});
			}
			break;
			case TRN::Engine::SCHEDULER_CUSTOM:
			{
				auto message = handle->communicator->receive<TRN::Engine::SCHEDULER_CUSTOM>(0);
				handle->to_caller->post([=]()
				{
					callback_scheduler(message.simulation_id, message.evaluation_id, message.seed, message.elements, message.rows, message.cols, message.offsets, message.durations);
				});
			}
			break;
			case TRN::Engine::SCHEDULING:
			{
				auto message = handle->communicator->receive<TRN::Engine::SCHEDULING>(0);
				handle->to_caller->post([=]()
				{
					callback_scheduling(message.simulation_id, message.evaluation_id, message.offsets, message.durations);
				});
			}
			break;
			case TRN::Engine::MUTATOR_CUSTOM:
			{
				auto message = handle->communicator->receive<TRN::Engine::MUTATOR_CUSTOM>(0);
				handle->to_caller->post([=]()
				{
					callback_mutator(message.simulation_id, message.evaluation_id, message.seed, message.offsets, message.durations);
				});
			}
			break;
			case TRN::Engine::FEEDFORWARD_DIMENSIONS:
			{
				auto message = handle->communicator->receive<TRN::Engine::FEEDFORWARD_DIMENSIONS>(0);
				handle->to_caller->post([=]()
				{
					callback_feedforward(message.simulation_id, message.seed, message.matrices, message.rows, message.cols);
				});
			}
			break;
			case TRN::Engine::RECURRENT_DIMENSIONS:
			{
				auto message = handle->communicator->receive<TRN::Engine::RECURRENT_DIMENSIONS>(0);
				handle->to_caller->post([=]()
				{
					callback_recurrent(message.simulation_id, message.seed, message.matrices, message.rows, message.cols);
				});
			}
			break;

			case TRN::Engine::READOUT_DIMENSIONS:
			{
				auto message = handle->communicator->receive<TRN::Engine::READOUT_DIMENSIONS>(0);
				handle->to_caller->post([=]()
				{
					callback_readout(message.simulation_id, message.seed, message.matrices, message.rows, message.cols);
				});
			}
			break;

			case TRN::Engine::ALLOCATED:
			{
				auto message = handle->communicator->receive<TRN::Engine::ALLOCATED>(0);
				auto processor = handle->manager->retrieve(message.simulation_id);
				/*retrieve_simulation(message.simulation_id)->post([=]()
				{*/
				processor->allocated();
				/*	});*/

				handle->to_caller->post([=]()
				{
					callback_allocated(message.simulation_id, processor->get_rank());
				});
			}
			break;
			case TRN::Engine::DEALLOCATED:
			{
				auto message = handle->communicator->receive<TRN::Engine::DEALLOCATED>(0);
				auto processor = handle->manager->retrieve(message.simulation_id);
				/*retrieve_simulation(message.simulation_id)->post([=]()
				{*/

				/*});*/
				handle->manager->deallocate(message.simulation_id);
				//	INFORMATION_LOGGER <<   "id " << id << " deallocate ack" ;
				processor->deallocated();
				remove_simulation(message.simulation_id);
				handle->to_caller->post([=]()
				{
					callback_deallocated(message.simulation_id, processor->get_rank());
				});
			}
			break;


			case TRN::Engine::CONFIGURED:
			{
				auto message = handle->communicator->receive<TRN::Engine::CONFIGURED>(0);
				auto processor = handle->manager->retrieve(message.simulation_id);

				/*			retrieve_simulation(message.simulation_id)->post([=]()
							{*/
				processor->configured();
				/*	});*/



				handle->to_caller->post([=]()
				{
					callback_configured(message.simulation_id);
				});
			}
			break;

			case TRN::Engine::TRAINED:
			{
				auto message = handle->communicator->receive<TRN::Engine::TRAINED>(0);
				auto processor = handle->manager->retrieve(message.simulation_id);
				/*retrieve_simulation(message.simulation_id)->post([=]()
				{*/
				processor->trained();
				processor->set_t1(std::clock());
				/*});*/

				handle->to_caller->post([=]()
				{
					callback_trained(message.simulation_id, message.evaluation_id);
				});
			}
			break;
			case TRN::Engine::PRIMED:
			{
				auto message = handle->communicator->receive<TRN::Engine::PRIMED>(0);
				auto processor = handle->manager->retrieve(message.simulation_id);
				/*retrieve_simulation(message.simulation_id)->post([=]()
				{*/
				processor->primed();
				/*});*/

				handle->to_caller->post([=]()
				{
					callback_primed(message.simulation_id, message.evaluation_id);
				});
			}
			break;
			case TRN::Engine::TESTED:
			{
				auto message = handle->communicator->receive<TRN::Engine::TESTED>(0);
				auto processor = handle->manager->retrieve(message.simulation_id);
				/*	retrieve_simulation(message.simulation_id)->post([=]()
					{*/
				processor->tested();
				/*});*/

				handle->to_caller->post([=]()
				{
					callback_tested(message.simulation_id, message.evaluation_id);
				});
			}
			break;
			case TRN::Engine::MEASUREMENT_READOUT_MEAN_SQUARE_ERROR:
			{
				auto message = handle->communicator->receive<TRN::Engine::MEASUREMENT_READOUT_MEAN_SQUARE_ERROR>(0);
				handle->to_caller->post([=]()
				{
					callback_measurement_readout_mean_square_error(message.simulation_id, message.evaluation_id, message.elements, message.rows, message.cols);
				});
			}
			break;
			case TRN::Engine::MEASUREMENT_READOUT_FRECHET_DISTANCE:
			{
				auto message = handle->communicator->receive<TRN::Engine::MEASUREMENT_READOUT_FRECHET_DISTANCE>(0);
				handle->to_caller->post([=]()
				{
					callback_measurement_readout_frechet_distance(message.simulation_id, message.evaluation_id, message.elements, message.rows, message.cols);
				});
			}
			break;
			case TRN::Engine::MEASUREMENT_READOUT_CUSTOM:
			{
				auto message = handle->communicator->receive<TRN::Engine::MEASUREMENT_READOUT_CUSTOM>(0);
				handle->to_caller->post([=]()
				{
					callback_measurement_readout_custom(message.simulation_id, message.evaluation_id, message.primed, message.elements, message.expected, message.preamble, message.matrices, message.rows, message.cols);
				});
			}
			break;
			case TRN::Engine::MEASUREMENT_POSITION_MEAN_SQUARE_ERROR:
			{
				auto message = handle->communicator->receive<TRN::Engine::MEASUREMENT_POSITION_MEAN_SQUARE_ERROR>(0);
				handle->to_caller->post([=]()
				{
					callback_measurement_position_mean_square_error(message.simulation_id, message.evaluation_id, message.elements, message.rows, message.cols);
				});
			}
			break;
			case TRN::Engine::MEASUREMENT_POSITION_FRECHET_DISTANCE:
			{
				auto message = handle->communicator->receive<TRN::Engine::MEASUREMENT_POSITION_FRECHET_DISTANCE>(0);
				handle->to_caller->post([=]()
				{
					callback_measurement_position_frechet_distance(message.simulation_id, message.evaluation_id, message.elements, message.rows, message.cols);
				});
			}
			break;
			case TRN::Engine::MEASUREMENT_POSITION_CUSTOM:
			{
				auto message = handle->communicator->receive<TRN::Engine::MEASUREMENT_POSITION_CUSTOM>(0);
				handle->to_caller->post([=]()
				{
					callback_measurement_position_custom(message.simulation_id, message.evaluation_id, message.primed, message.elements, message.expected, message.preamble, message.matrices, message.rows, message.cols);
				});
			}
			break;


			default:
				throw std::invalid_argument("Unexpected tag " + std::to_string(tag));
		}
	}
}



void TRN::Engine::Broker::append_simulation(const unsigned long long &simulation_id)
{
	TRACE_LOGGER;
	std::unique_lock<std::mutex> lock(handle->mutex);
	if (handle->simulations.find(simulation_id) != handle->simulations.end())
		throw std::logic_error("Simulation #" + std::to_string(simulation_id) + " is already allocated");
	handle->simulations.insert(simulation_id);

	handle->from_caller[simulation_id] = TRN::Engine::Executor::create();
	handle->from_caller[simulation_id]->start();
}

std::shared_ptr<TRN::Engine::Executor> TRN::Engine::Broker::retrieve_simulation(const unsigned long long &simulation_id)
{
	TRACE_LOGGER;
	std::unique_lock<std::mutex> lock(handle->mutex);

	return handle->from_caller[simulation_id];
}


void TRN::Engine::Broker::remove_simulation(const unsigned long long &simulation_id)
{
	TRACE_LOGGER;
	std::unique_lock<std::mutex> lock(handle->mutex);;
	if (handle->simulations.find(simulation_id) == handle->simulations.end())
		throw std::logic_error("Simulation #" + std::to_string(simulation_id) + " is not allocated");
	handle->simulations.erase(simulation_id);
	handle->from_caller[simulation_id]->terminate();
	handle->from_caller.erase(simulation_id);

	//	INFORMATION_LOGGER <<   "id " << id << " deallocate DONE" ;
}



/*void TRN::Engine::Broker::ready(const unsigned long long &simulation_id)
{
	retrieve_simulation(simulation_id)->post([=]()
	{
		auto processor = handle->manager->retrieve(simulation_id);
		//processor->configured();

		processor->post([=]()
		{	
		
			TRN::Engine::Message<READY> message;
			message.simulation_id = simulation_id;
			
			send(processor->get_rank(), message, [=]()
			{
			
							////INFORMATION_LOGGER <<   "id " << id << " allocate ack" ;
			});
		});
	});
}*/
void TRN::Engine::Broker::allocate(const unsigned long long &simulation_id)
{
	TRACE_LOGGER;
	append_simulation(simulation_id);

	retrieve_simulation(simulation_id)->post([=]()
	{
		auto processor = handle->manager->allocate(simulation_id);
		processor->allocating();
		processor->post([=]()
		{
			TRN::Engine::Message<ALLOCATE> message;
			message.simulation_id = simulation_id;
		
			handle->communicator->send(message, processor->get_rank());
		});
	});
}
void TRN::Engine::Broker::deallocate(const unsigned long long &simulation_id)
{
	TRACE_LOGGER;
	retrieve_simulation(simulation_id)->post([=]()
	{
		//INFORMATION_LOGGER <<   "entering " << id << " " << __FUNCTION__ ;
		auto processor = handle->manager->retrieve(simulation_id);

		processor->deallocating();
		processor->post([=]()
		{	
			TRN::Engine::Message<DEALLOCATE> message;
			message.simulation_id = simulation_id;

			handle->communicator->send(message, processor->get_rank());
		});
		//INFORMATION_LOGGER <<   "exiting " << id << " " << __FUNCTION__ ;
	});
}
void TRN::Engine::Broker::train(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::string &label, const std::string &incoming, const std::string &expected, const bool &reset_readout)
{
	TRACE_LOGGER;
	retrieve_simulation(simulation_id)->post([=]()
	{
		auto processor = handle->manager->retrieve(simulation_id);
		processor->training();
		processor->set_t0(std::clock());
		processor->post([=]()
		{
		
			TRN::Engine::Message<TRAIN> message;
			message.simulation_id = simulation_id;
			message.evaluation_id = evaluation_id;
			message.label = label;
			message.incoming = incoming;
			message.expected = expected;
			message.reset_readout = reset_readout;

			handle->communicator->send(message, processor->get_rank());
		});
	});
}
void TRN::Engine::Broker::test(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::string &label, const std::string &incoming, const std::string &expected, const unsigned int &preamble, const bool &autonomous, const unsigned int &supplementary_generations)
{
	TRACE_LOGGER;
	retrieve_simulation(simulation_id)->post([=]()
	{
		auto processor = handle->manager->retrieve(simulation_id);
		processor->testing();
		processor->post([=]()
		{
			
			TRN::Engine::Message<TEST> message;
			message.simulation_id = simulation_id;
			message.evaluation_id = evaluation_id;
			message.label = label;
			message.incoming = incoming;
			message.expected = expected;
			message.preamble = preamble;
			message.autonomous = autonomous;
			message.supplementary_generations = supplementary_generations;
			handle->communicator->send(message, processor->get_rank());
		});
	});
}
void TRN::Engine::Broker::declare_sequence(const unsigned long long &simulation_id, const std::string &label, const std::string &tag, const std::vector<float> &sequence, const std::size_t &observations)
{
	TRACE_LOGGER;

	if (sequence.empty())
		throw std::runtime_error("Sequence with label " + label + " and tag " + tag + " is empty");

	retrieve_simulation(simulation_id)->post([=]()
	{
		auto processor = handle->manager->retrieve(simulation_id);
		processor->declare();
		auto rank = processor->get_rank();

		std::vector<float> to_send;
		auto checksum = TRN::Engine::checksum(sequence);
		std::unique_lock<std::mutex> guard(handle->cache_mutex);
		if (handle->cached[rank].find(checksum) == handle->cached[rank].end())
		{
			DEBUG_LOGGER << "Data having checksum 0x" << std::hex << checksum << " is not cached. Will send it";
			to_send = sequence;
			handle->cached[rank].insert(checksum);
		}
		else
		{
			DEBUG_LOGGER << "Data having checksum 0x" << std::hex << checksum << " is cached. Won't send it";
		}
	
		
		processor->post([=]()
		{
			TRN::Engine::Message<DECLARE_SEQUENCE> message;
			message.simulation_id = simulation_id;
			message.label = label;
			message.tag = tag;
			message.checksum = checksum;
			message.sequence = to_send;
			message.observations = observations;
			handle->communicator->send(message, processor->get_rank());
		});
	});
}
void TRN::Engine::Broker::declare_set(const unsigned long long &simulation_id, const std::string &label, const std::string &tag, const std::vector<std::string> &labels)
{
	TRACE_LOGGER;


	retrieve_simulation(simulation_id)->post([=]()
	{
		auto processor = handle->manager->retrieve(simulation_id);
		processor->declare();
		processor->post([=]()
		{

			TRN::Engine::Message<DECLARE_SET> message;
			message.simulation_id = simulation_id;
			message.label = label;
			message.tag = tag;
			message.labels = labels;
			handle->communicator->send(message, processor->get_rank());
		});
	});
}
void TRN::Engine::Broker::setup_states(const unsigned long long &simulation_id, const bool &train, const bool &prime, const bool &generate)
{
	TRACE_LOGGER;
	retrieve_simulation(simulation_id)->post([=]()
	{
		auto processor = handle->manager->retrieve(simulation_id);
		processor->configuring();
		processor->post([=]()
		{
			
			TRN::Engine::Message<SETUP_STATES> message;
			message.simulation_id = simulation_id;
			message.train = train;
			message.prime = prime;
			message.generate = generate;
			handle->communicator->send(message, processor->get_rank());
		});
	});
}
void TRN::Engine::Broker::setup_weights(const unsigned long long &simulation_id, const bool &initalization, const bool &train)
{
	TRACE_LOGGER;
	retrieve_simulation(simulation_id)->post([=]()
	{
		auto processor = handle->manager->retrieve(simulation_id);
		processor->configuring();
		processor->post([=]()
		{
		
			TRN::Engine::Message<SETUP_WEIGHTS> message;
			message.simulation_id = simulation_id;
			message.train = train;
			message.initialization = initalization;

			handle->communicator->send(message, processor->get_rank());
		});
	});
}
void TRN::Engine::Broker::setup_performances(const unsigned long long &simulation_id, const bool &train, const bool &prime, const bool &generate)
{
	TRACE_LOGGER;
	retrieve_simulation(simulation_id)->post([=]()
	{	
		auto processor = handle->manager->retrieve(simulation_id);
		processor->configuring();
		processor->post([=]()
		{		
			TRN::Engine::Message<SETUP_PERFORMANCES> message;
			message.simulation_id = simulation_id;
			message.train = train;
			message.prime = prime;
			message.generate = generate;
			handle->communicator->send(message, processor->get_rank());
		});
	});
}
void TRN::Engine::Broker::setup_scheduling(const unsigned long long &simulation_id)
{
	TRACE_LOGGER;
	retrieve_simulation(simulation_id)->post([=]()
	{
		auto processor = handle->manager->retrieve(simulation_id);
		processor->configuring();
		processor->post([=]()
		{
		
			TRN::Engine::Message<SETUP_SCHEDULING> message;
			message.simulation_id = simulation_id;

			handle->communicator->send(message, processor->get_rank());
		});
	});
}
void TRN::Engine::Broker::configure_begin(const unsigned long long &simulation_id)
{
	TRACE_LOGGER;
	retrieve_simulation(simulation_id)->post([=]()
	{
		auto processor = handle->manager->retrieve(simulation_id);
		processor->configuring();
		processor->post([=]()
		{
			TRN::Engine::Message<CONFIGURE_BEGIN> message;
			message.simulation_id = simulation_id;
			handle->communicator->send(message, processor->get_rank());
		});
	});
}


void TRN::Engine::Broker::configure_end(const unsigned long long &simulation_id)
{
	TRACE_LOGGER;
	auto processor = handle->manager->retrieve(simulation_id);
	retrieve_simulation(simulation_id)->post([=]()
	{
		
		processor->configuring();
		processor->post([=]()
		{
		
			TRN::Engine::Message<CONFIGURE_END> message;
			message.simulation_id = simulation_id;
			handle->communicator->send(message, processor->get_rank());
		});
	});
	//processor->ready();
}
void TRN::Engine::Broker::configure_measurement_readout_mean_square_error(const unsigned long long &simulation_id, const std::size_t &batch_size)
{
	TRACE_LOGGER;
    retrieve_simulation(simulation_id)->post([=]()
	{
		auto processor = handle->manager->retrieve(simulation_id);
		processor->configuring();
		processor->post([=]()
		{
		
			TRN::Engine::Message<CONFIGURE_MEASUREMENT_READOUT_MEAN_SQUARE_ERROR> message;
			message.simulation_id = simulation_id;
			message.batch_size = batch_size;
			handle->communicator->send(message, processor->get_rank());
		});
 });
}
void TRN::Engine::Broker::configure_measurement_readout_frechet_distance(const unsigned long long &simulation_id,  const std::size_t &batch_size, const std::string &norm, const std::string &aggregator)
{
	TRACE_LOGGER;
	retrieve_simulation(simulation_id)->post([=]()
	{
		auto processor = handle->manager->retrieve(simulation_id);
		processor->configuring();
		processor->post([=]()
		{
		
			TRN::Engine::Message<CONFIGURE_MEASUREMENT_READOUT_FRECHET_DISTANCE> message;
			message.simulation_id = simulation_id;
			message.batch_size = batch_size;
			message.norm = norm;
			message.aggregator = aggregator;
			handle->communicator->send(message, processor->get_rank());
		});
	});
}
void TRN::Engine::Broker::configure_measurement_readout_custom(const unsigned long long &simulation_id, const std::size_t &batch_size)
{
	TRACE_LOGGER;
	retrieve_simulation(simulation_id)->post([=]()
	{
		auto processor = handle->manager->retrieve(simulation_id);
		processor->configuring();
		processor->post([=]()
		{
		
			TRN::Engine::Message<CONFIGURE_MEASUREMENT_READOUT_CUSTOM> message;
			message.simulation_id = simulation_id;
			message.batch_size = batch_size;
			handle->communicator->send(message, processor->get_rank());
		});
	});
}
void TRN::Engine::Broker::configure_measurement_position_mean_square_error(const unsigned long long &simulation_id, const std::size_t &batch_size)
{
	TRACE_LOGGER;
	retrieve_simulation(simulation_id)->post([=]()
	{
		auto processor = handle->manager->retrieve(simulation_id);
		processor->configuring();
		processor->post([=]()
		{
	
			TRN::Engine::Message<CONFIGURE_MEASUREMENT_POSITION_MEAN_SQUARE_ERROR> message;
			message.simulation_id = simulation_id;
			message.batch_size = batch_size;
			handle->communicator->send(message, processor->get_rank());
		});
	});
}
void TRN::Engine::Broker::configure_measurement_position_frechet_distance(const unsigned long long &simulation_id, const std::size_t &batch_size, const std::string &norm, const std::string &aggregator)
{
	TRACE_LOGGER;
	retrieve_simulation(simulation_id)->post([=]()
	{
		auto processor = handle->manager->retrieve(simulation_id);
		processor->configuring();
		processor->post([=]()
		{
		

			TRN::Engine::Message<CONFIGURE_MEASUREMENT_POSITION_FRECHET_DISTANCE> message;
			message.simulation_id = simulation_id;
			message.batch_size = batch_size;
			message.norm = norm;
			message.aggregator = aggregator;
			handle->communicator->send(message, processor->get_rank());
		});
	});
}
void TRN::Engine::Broker::configure_measurement_position_custom(const unsigned long long &simulation_id, const std::size_t &batch_size)
{
	TRACE_LOGGER;
	retrieve_simulation(simulation_id)->post([=]()
	{
		auto processor = handle->manager->retrieve(simulation_id);
		processor->configuring();
		processor->post([=]()
		{
		
			TRN::Engine::Message<CONFIGURE_MEASUREMENT_POSITION_CUSTOM> message;
			message.simulation_id = simulation_id;
			message.batch_size = batch_size;
			handle->communicator->send(message, processor->get_rank());
		});
	});
}
void TRN::Engine::Broker::configure_reservoir_widrow_hoff(const unsigned long long &simulation_id, const std::size_t &stimulus_size, const std::size_t &prediction_size, const std::size_t &reservoir_size, const float &leak_rate, const float &initial_state_scale, const float &learning_rate, const unsigned long &seed, const std::size_t &batch_size, const std::size_t &mini_batch_size)
{
	TRACE_LOGGER;
	retrieve_simulation(simulation_id)->post([=]()
	{
		auto processor = handle->manager->retrieve(simulation_id);
		processor->configuring();
		processor->post([=]()
		{
	
			TRN::Engine::Message<CONFIGURE_RESERVOIR_WIDROW_HOFF> message;
			message.simulation_id = simulation_id;
			message.stimulus_size = stimulus_size;
			message.prediction_size = prediction_size;
			message.reservoir_size = reservoir_size;
			message.leak_rate = leak_rate;
			message.initial_state_scale = initial_state_scale;
			message.learning_rate = learning_rate;
			message.batch_size = batch_size;
			message.mini_batch_size = mini_batch_size;
			message.seed = seed;
			handle->communicator->send(message, processor->get_rank());
		});
	});
}
void TRN::Engine::Broker::configure_loop_copy(const unsigned long long &simulation_id, const std::size_t &batch_size, const std::size_t &stimulus_size)
{
	TRACE_LOGGER;
	retrieve_simulation(simulation_id)->post([=]()
	{
		auto processor = handle->manager->retrieve(simulation_id);
		processor->configuring();
		processor->post([=]()
		{
		
			TRN::Engine::Message<CONFIGURE_LOOP_COPY> message;
			message.simulation_id = simulation_id;
			message.stimulus_size = stimulus_size;
			message.batch_size = batch_size;
			handle->communicator->send(message, processor->get_rank());
		});
	});
}
void	TRN::Engine::Broker::configure_encoder_custom(const unsigned long long &simulation_id, const std::size_t &batch_size, const std::size_t &stimulus_size
)
{
	TRACE_LOGGER;
	retrieve_simulation(simulation_id)->post([=]()
	{
		auto processor = handle->manager->retrieve(simulation_id);
		processor->configuring();
		processor->post([=]()
		{
			TRN::Engine::Message<CONFIGURE_ENCODER_CUSTOM> message;

			message.simulation_id = simulation_id;
			message.batch_size = batch_size;
			message.stimulus_size = stimulus_size;
			handle->communicator->send(message, processor->get_rank());
		});
	});
}
void	TRN::Engine::Broker::configure_encoder_model(const unsigned long long &simulation_id, const std::size_t &batch_size, const std::size_t &stimulus_size,
	const std::vector<float> &cx,
	const std::vector<float> &cy,
	const std::vector<float> &K
)
{
	TRACE_LOGGER;
	retrieve_simulation(simulation_id)->post([=]()
	{
		auto processor = handle->manager->retrieve(simulation_id);
		processor->configuring();
		processor->post([=]()
		{
			TRN::Engine::Message<CONFIGURE_ENCODER_MODEL> message;

			message.simulation_id = simulation_id;
			message.batch_size = batch_size;
			message.stimulus_size = stimulus_size;
			message.cx = cx;
			message.cy = cy;
			message.K = K;
			handle->communicator->send(message, processor->get_rank());
		});
	});
}

void	TRN::Engine::Broker::configure_decoder_linear(const unsigned long long &simulation_id, const std::size_t &batch_size, const std::size_t &stimulus_size,
	const std::vector<float> &cx, const std::vector<float> &cy)
{
	TRACE_LOGGER;
	retrieve_simulation(simulation_id)->post([=]()
	{
		auto processor = handle->manager->retrieve(simulation_id);
		processor->configuring();
		processor->post([=]()
		{

			TRN::Engine::Message<CONFIGURE_DECODER_LINEAR> message;

			message.simulation_id = simulation_id;
			message.batch_size = batch_size;
			message.stimulus_size = stimulus_size;
			message.cx = cx;
			message.cy = cy;
			handle->communicator->send(message, processor->get_rank());
		});
	});
}
void	TRN::Engine::Broker::configure_decoder_kernel_map(const unsigned long long &simulation_id, const std::size_t &batch_size, const std::size_t &stimulus_size,
	const std::size_t &rows, const std::size_t &cols,
	const std::pair<float, float> &x, const std::pair<float, float> &y,
	const float &sigma,
	const float &radius,
	const float &angle,
	const float &scale,
	const unsigned long &seed,
	const std::vector<float> &response
	)
{
	TRACE_LOGGER;
	retrieve_simulation(simulation_id)->post([=]()
	{
		auto processor = handle->manager->retrieve(simulation_id);
		processor->configuring();
		auto rank = processor->get_rank();
		std::vector<float> response_to_send;
		auto response_checksum = TRN::Engine::checksum(response);
		std::unique_lock<std::mutex> guard(handle->cache_mutex);
		if (handle->cached[rank].find(response_checksum) == handle->cached[rank].end())
		{
			DEBUG_LOGGER << "Data having checksum 0x" << std::hex << response_checksum << " is not cached. Will send it";
			response_to_send = response;
			handle->cached[rank].insert(response_checksum);
		}
		else
		{
			DEBUG_LOGGER << "Data having checksum 0x" << std::hex << response_checksum << " is cached. Won't send it";
		}
		processor->post([=]()
		{

			TRN::Engine::Message<CONFIGURE_DECODER_KERNEL_MAP> message;

			message.simulation_id = simulation_id;
			message.batch_size = batch_size;
			message.stimulus_size = stimulus_size;
			message.rows = rows;
			message.cols = cols;
			message.x = x;
			message.y = y;
			message.response.first = response_checksum;
			message.response.second = response_to_send;

			message.sigma = sigma;
			message.radius = radius;
			message.scale = scale;
			message.angle = angle;
			message.seed = seed;

			handle->communicator->send(message, processor->get_rank());
		});
	});
}
void	TRN::Engine::Broker::configure_decoder_kernel_model(const unsigned long long &simulation_id, const std::size_t &batch_size, const std::size_t &stimulus_size,
	const std::size_t &rows, const std::size_t &cols,
	const std::pair<float, float> &x, const std::pair<float, float> &y,
	const float &sigma,
	const float &radius,
	const float &angle,
	const float &scale,
	const unsigned long &seed,
	const std::vector<float> &cx,
	const std::vector<float> &cy,
	const std::vector<float> &K
)
{
	TRACE_LOGGER;
	retrieve_simulation(simulation_id)->post([=]()
	{
		auto processor = handle->manager->retrieve(simulation_id);
		processor->configuring();
		processor->post([=]()
		{

			TRN::Engine::Message<CONFIGURE_DECODER_KERNEL_MODEL> message;

			message.simulation_id = simulation_id;
			message.batch_size = batch_size;
			message.stimulus_size = stimulus_size;
			message.rows = rows;
			message.cols = cols;
			message.x = x;
			message.y = y;
			message.cx = cx;
			message.cy = cy;
			message.K = K;

			message.sigma = sigma;
			message.radius = radius;
			message.scale = scale;
			message.angle = angle;
			message.seed = seed;

			handle->communicator->send(message, processor->get_rank());
		});
	});
}

void TRN::Engine::Broker::configure_loop_spatial_filter(const unsigned long long &simulation_id, const std::size_t &batch_size, const std::size_t &stimulus_size, 
	const std::string &tag)
{
	TRACE_LOGGER;
	retrieve_simulation(simulation_id)->post([=]()
	{
		auto processor = handle->manager->retrieve(simulation_id);
		
		processor->configuring();
		
		processor->post([=]()
		{
			TRN::Engine::Message<CONFIGURE_LOOP_SPATIAL_FILTER> message;

			message.simulation_id = simulation_id;
			message.stimulus_size = stimulus_size;
			message.batch_size = batch_size;
	
			message.tag = tag;
	
			handle->communicator->send(message, processor->get_rank());
		});
	});
}
void TRN::Engine::Broker::configure_loop_custom(const unsigned long long &simulation_id, const std::size_t &batch_size, const std::size_t &stimulus_size)
{
	TRACE_LOGGER;
	retrieve_simulation(simulation_id)->post([=]()
	{
		auto processor = handle->manager->retrieve(simulation_id);
		processor->configuring();
		processor->post([=]()
		{
	
			TRN::Engine::Message<CONFIGURE_LOOP_CUSTOM> message;

			message.simulation_id = simulation_id;
			message.batch_size = batch_size;
			message.stimulus_size = stimulus_size;
			handle->communicator->send(message, processor->get_rank());
		});
	});
}
void TRN::Engine::Broker::configure_scheduler_tiled(const unsigned long long &simulation_id, const unsigned int &epochs)
{
	TRACE_LOGGER;
	retrieve_simulation(simulation_id)->post([=]()
	{
	 auto processor = handle->manager->retrieve(simulation_id);
	 processor->configuring();
		processor->post([=]()
		{
		
			TRN::Engine::Message<CONFIGURE_SCHEDULER_TILED> message;
			message.simulation_id = simulation_id;
			message.epochs = epochs;

			handle->communicator->send(message, processor->get_rank());
		});
	});
}
void TRN::Engine::Broker::configure_scheduler_snippets(const unsigned long long &simulation_id, const unsigned long &seed, const unsigned int &snippets_size, const unsigned int &time_budget,  const std::string &tag)
{
	TRACE_LOGGER;
	retrieve_simulation(simulation_id)->post([=]()
	{
		 auto processor = handle->manager->retrieve(simulation_id);
		 processor->configuring();
		processor->post([=]()
		{
		
			TRN::Engine::Message<CONFIGURE_SCHEDULER_SNIPPETS> message;
			message.simulation_id = simulation_id;
			message.snippets_size = snippets_size;
			message.time_budget = time_budget;
			message.tag = tag;
			message.seed = seed;

			handle->communicator->send(message, processor->get_rank());
		});
	});
}
void TRN::Engine::Broker::configure_scheduler_custom(const unsigned long long &simulation_id, const unsigned long &seed, const std::string &tag)
{
	TRACE_LOGGER;
	retrieve_simulation(simulation_id)->post([=]()
	{
		 auto processor = handle->manager->retrieve(simulation_id);
		 processor->configuring();
		processor->post([=]()
		{



			TRN::Engine::Message<CONFIGURE_SCHEDULER_CUSTOM> message;
			message.simulation_id = simulation_id;
			message.seed = seed;

			handle->communicator->send(message, processor->get_rank());
		});
	});
}
void TRN::Engine::Broker::configure_mutator_shuffle(const unsigned long long &simulation_id,const unsigned long &seed)
{
	TRACE_LOGGER;
	retrieve_simulation(simulation_id)->post([=]()
	{
		auto processor = handle->manager->retrieve(simulation_id);
		processor->configuring();
		processor->post([=]()
		{
		
			TRN::Engine::Message<CONFIGURE_MUTATOR_SHUFFLE> message;
			message.simulation_id = simulation_id;
			message.seed = seed;
			handle->communicator->send(message, processor->get_rank());
		});
	});
}
void TRN::Engine::Broker::configure_mutator_reverse(const unsigned long long &simulation_id, const unsigned long &seed, const float &rate, const std::size_t &size)
{
	TRACE_LOGGER;
	retrieve_simulation(simulation_id)->post([=]()
	{
		auto processor = handle->manager->retrieve(simulation_id);
		processor->configuring();
		processor->post([=]()
		{
	
			TRN::Engine::Message<CONFIGURE_MUTATOR_REVERSE> message;
			message.simulation_id = simulation_id;
			message.seed = seed;
			message.rate = rate;
			message.size = size;
			handle->communicator->send(message, processor->get_rank());
		});
	});
}
void TRN::Engine::Broker::configure_mutator_punch(const unsigned long long &simulation_id, const unsigned long &seed, const float &rate, const std::size_t &size, const std::size_t &counter)
{
	TRACE_LOGGER;
	retrieve_simulation(simulation_id)->post([=]()
	{
		auto processor = handle->manager->retrieve(simulation_id);
		processor->configuring();
		processor->post([=]()
		{
	
			TRN::Engine::Message<CONFIGURE_MUTATOR_PUNCH> message;
			message.simulation_id = simulation_id;
			message.seed = seed;
			message.rate = rate;
			message.size = size;
			message.repetition = counter;
			handle->communicator->send(message, processor->get_rank());
		});
	});
}
void TRN::Engine::Broker::configure_mutator_custom(const unsigned long long &simulation_id, const unsigned long &seed)
{
	TRACE_LOGGER;
	retrieve_simulation(simulation_id)->post([=]()
	{
		auto processor = handle->manager->retrieve(simulation_id);
		processor->configuring();
		processor->post([=]()
		{
	

			TRN::Engine::Message<CONFIGURE_MUTATOR_CUSTOM> message;
			message.simulation_id = simulation_id;
			message.seed = seed;
			handle->communicator->send(message, processor->get_rank());
		});
	});
}
void TRN::Engine::Broker::configure_readout_uniform(const unsigned long long &simulation_id, const float &a, const float &b, const float &sparsity)
{
	TRACE_LOGGER;
	retrieve_simulation(simulation_id)->post([=]()
	{
		auto processor = handle->manager->retrieve(simulation_id);
		processor->configuring();
		processor->post([=]()
		{
		
			TRN::Engine::Message<CONFIGURE_READOUT_UNIFORM> message;
			message.simulation_id = simulation_id;
			message.a = a;
			message.b = b;
			message.sparsity = sparsity;
			handle->communicator->send(message, processor->get_rank());
		});
	});
}
void TRN::Engine::Broker::configure_readout_gaussian(const unsigned long long &simulation_id, const float &mu, const float &sigma, const float &sparsity)
{
	TRACE_LOGGER;
	retrieve_simulation(simulation_id)->post([=]()
	{
		auto processor = handle->manager->retrieve(simulation_id);
		processor->configuring();
		processor->post([=]()
		{

			TRN::Engine::Message<CONFIGURE_READOUT_GAUSSIAN> message;
			message.simulation_id = simulation_id;
			message.mu = mu;
			message.sigma = sigma;
			message.sparsity = sparsity;

			handle->communicator->send(message, processor->get_rank());
		});
	});
}
void TRN::Engine::Broker::configure_readout_custom(const unsigned long long &simulation_id)
{
	TRACE_LOGGER;
	retrieve_simulation(simulation_id)->post([=]()
	{
		auto processor = handle->manager->retrieve(simulation_id);
		processor->configuring();
		processor->post([=]()
		{
	

			TRN::Engine::Message<CONFIGURE_READOUT_CUSTOM> message;
			message.simulation_id = simulation_id;

			handle->communicator->send(message, processor->get_rank());
		});
	});
}

void TRN::Engine::Broker::configure_recurrent_uniform(const unsigned long long &simulation_id, const float &a, const float &b, const float &sparsity)
{
	TRACE_LOGGER;
	retrieve_simulation(simulation_id)->post([=]()
	{
		auto processor = handle->manager->retrieve(simulation_id);
		processor->configuring();
		processor->post([=]()
		{
			
			TRN::Engine::Message<CONFIGURE_RECURRENT_UNIFORM> message;
			message.simulation_id = simulation_id;
			message.a = a;
			message.b = b;
			message.sparsity = sparsity;
			handle->communicator->send(message, processor->get_rank());
		});
	});
}
void TRN::Engine::Broker::configure_recurrent_gaussian(const unsigned long long &simulation_id, const float &mu, const float &sigma, const float &sparsity)
{
	TRACE_LOGGER;
	retrieve_simulation(simulation_id)->post([=]()
	{
		auto processor = handle->manager->retrieve(simulation_id);
		processor->configuring();
		processor->post([=]()
		{
			
			TRN::Engine::Message<CONFIGURE_RECURRENT_GAUSSIAN> message;
			message.simulation_id = simulation_id;
			message.mu = mu;
			message.sigma = sigma;
			message.sparsity = sparsity;

			handle->communicator->send(message, processor->get_rank());
		});
	});
}
void TRN::Engine::Broker::configure_recurrent_custom(const unsigned long long &simulation_id)
{
	TRACE_LOGGER;
	retrieve_simulation(simulation_id)->post([=]()
	{
		auto processor = handle->manager->retrieve(simulation_id);
		processor->configuring();
		processor->post([=]()
		{
			
			TRN::Engine::Message<CONFIGURE_RECURRENT_CUSTOM> message;
			message.simulation_id = simulation_id;

			handle->communicator->send(message, processor->get_rank());
		});
	 });
}
void TRN::Engine::Broker::configure_feedforward_uniform(const unsigned long long &simulation_id, const float &a, const float &b, const float &sparsity)
{
	TRACE_LOGGER;
	retrieve_simulation(simulation_id)->post([=]()
	{
		auto processor = handle->manager->retrieve(simulation_id);
		processor->configuring();
		processor->post([=]()
		{
		
			TRN::Engine::Message<CONFIGURE_FEEDFORWARD_UNIFORM> message;
			message.simulation_id = simulation_id;
			message.a = a;
			message.b = b;
			message.sparsity = sparsity;
			handle->communicator->send(message, processor->get_rank());
		});
	});
}
void TRN::Engine::Broker::configure_feedforward_gaussian(const unsigned long long &simulation_id, const float &mu, const float &sigma, const float &sparsity)
{
	TRACE_LOGGER;
	retrieve_simulation(simulation_id)->post([=]()
	{
		auto processor = handle->manager->retrieve(simulation_id);
		processor->configuring();
		processor->post([=]()
		{
			
			TRN::Engine::Message<CONFIGURE_FEEDFORWARD_GAUSSIAN> message;
			message.simulation_id = simulation_id;
			message.mu = mu;
			message.sigma = sigma;
			message.sparsity = sparsity;

			handle->communicator->send(message, processor->get_rank());
		});
	});
}
void TRN::Engine::Broker::configure_feedforward_custom(const unsigned long long &simulation_id)
{
	TRACE_LOGGER;
	retrieve_simulation(simulation_id)->post([=]()
	{
		auto processor = handle->manager->retrieve(simulation_id);
		processor->configuring();
		processor->post([=]()
		{

			TRN::Engine::Message<CONFIGURE_FEEDFORWARD_CUSTOM> message;
			message.simulation_id = simulation_id;

			handle->communicator->send(message, processor->get_rank());
		});
	});
}
void TRN::Engine::Broker::notify_stimulus(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &stimulus, const std::size_t &rows, const std::size_t &cols)
{
	TRACE_LOGGER;
	/*retrieve_simulation(simulation_id)->post([=]()
	{*/
		auto processor = handle->manager->retrieve(simulation_id);
		/*processor->post([=]()
		{*/
			TRN::Engine::Message<STIMULUS> message;
			message.simulation_id = simulation_id;
			message.evaluation_id = evaluation_id;
			message.elements = stimulus;
			message.rows = rows;
			message.cols = cols;

			handle->communicator->send(message, processor->get_rank());
		/*});
	});*/
}
void TRN::Engine::Broker::notify_position(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols)
{
	TRACE_LOGGER;
	/*retrieve_simulation(simulation_id)->post([=]()
	{*/
		auto processor = handle->manager->retrieve(simulation_id);
		/*processor->post([=]()
		{*/
			TRN::Engine::Message<POSITION> message;
			message.simulation_id = simulation_id;
			message.evaluation_id = evaluation_id;
			message.elements = position;
			message.rows = rows;
			message.cols = cols;
		
			handle->communicator->send(message, processor->get_rank());
		/*});
	});*/
}
void TRN::Engine::Broker::notify_scheduler(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<int> &offsets, const std::vector<int> &durations)
{
	TRACE_LOGGER;
	/*retrieve_simulation(simulation_id)->post([=]()
	{*/
		auto processor = handle->manager->retrieve(simulation_id);
		/*processor->post([=]()
		{*/
			TRN::Engine::Message<SCHEDULING> message;
			message.simulation_id = simulation_id;
			message.evaluation_id = evaluation_id;
			message.is_from_mutator = false;
			message.offsets = offsets;
			message.durations = durations;

			handle->communicator->send(message, processor->get_rank());
		/*});
	});*/
}
void TRN::Engine::Broker::notify_mutator(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<int> &offsets, const std::vector<int> &durations)
{
	TRACE_LOGGER;
	/*retrieve_simulation(simulation_id)->post([=]()
	{*/
		auto processor = handle->manager->retrieve(simulation_id);
		/*processor->post([=]()
		{*/
			TRN::Engine::Message<SCHEDULING> message;
			message.simulation_id = simulation_id;
			message.is_from_mutator = true;
			message.offsets = offsets;
			message.durations = durations;
			message.evaluation_id = evaluation_id;

			handle->communicator->send(message, processor->get_rank());
	/*	});
	});*/
}
void TRN::Engine::Broker::notify_feedforward(const unsigned long long &simulation_id, const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)
{
	TRACE_LOGGER;
	/*retrieve_simulation(simulation_id)->post([=]()
	{*/
		auto processor = handle->manager->retrieve(simulation_id);
		/*processor->configuring();
		processor->post([=]()
		{
		*/
			TRN::Engine::Message<FEEDFORWARD_WEIGHTS> message;
			message.simulation_id = simulation_id;
			message.elements = weights;
			message.matrices = matrices;
			message.rows = rows;
			message.cols = cols;
			handle->communicator->send(message, processor->get_rank());
	/*	});
	});*/
}

void TRN::Engine::Broker::notify_readout(const unsigned long long &simulation_id, const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)
{
	TRACE_LOGGER;
	/*retrieve_simulation(simulation_id)->post([=]()
	{*/
		auto processor = handle->manager->retrieve(simulation_id);
		/*processor->configuring();
		processor->post([=]()
		{*/
	
			TRN::Engine::Message<READOUT_WEIGHTS> message;
			message.simulation_id = simulation_id;
			message.elements = weights;
			message.matrices = matrices;
			message.rows = rows;
			message.cols = cols;
			handle->communicator->send(message, processor->get_rank());
	/*	});
	});*/
}
void TRN::Engine::Broker::notify_recurrent(const unsigned long long &simulation_id, const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)
{
	TRACE_LOGGER;
	/*retrieve_simulation(simulation_id)->post([=]()
	{*/
		auto processor = handle->manager->retrieve(simulation_id);
		/*processor->configuring();
		processor->post([=]()
		{
		*/
			TRN::Engine::Message<RECURRENT_WEIGHTS> message;
			message.simulation_id = simulation_id;
			message.elements = weights;
			message.matrices = matrices;
			message.rows = rows;
			message.cols = cols;
			handle->communicator->send(message, processor->get_rank());
		/*});
	});*/
}


