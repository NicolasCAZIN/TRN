#include "stdafx.h"
#include "Node_impl.h"
#include "Helper/Logger.h"

TRN::Engine::Node::Node(const std::shared_ptr<TRN::Engine::Communicator> &communicator, const int &rank) :
	TRN::Helper::Bridge<TRN::Engine::Communicator, std::weak_ptr>(communicator),
	handle(std::make_unique<Handle>())
{
	TRACE_LOGGER;
	// INFORMATION_LOGGER <<   __FUNCTION__ ;
	handle->rank = rank;
	handle->cache = TRN::Engine::Cache::create();
	handle->synchronized = false;
}


TRN::Engine::Node::~Node()
{
	TRACE_LOGGER;

	// INFORMATION_LOGGER <<   __FUNCTION__ ;
	handle.reset();
}


void TRN::Engine::Node::synchronized()
{
	std::unique_lock<std::mutex> lock(handle->mutex);
	handle->synchronized = true;
	lock.unlock();
	handle->cond.notify_one();
}

void TRN::Engine::Node::synchronize()
{
	std::unique_lock<std::mutex> lock(handle->mutex);

	while (handle->synchronized)
		handle->cond.wait(lock);
	INFORMATION_LOGGER << __FUNCTION__ << "Node synchronized";
	lock.unlock();
}

template <TRN::Engine::Tag tag>
static TRN::Engine::Message<tag> unpack(const std::shared_ptr<TRN::Engine::Communicator> &communicator, const int &rank, unsigned long long &simulation_id, size_t &counter, unsigned short &number)
{
	// INFORMATION_LOGGER <<   __FUNCTION__ ;
	TRN::Engine::Message<tag> message = communicator->receive<tag>(rank);

	simulation_id = message.simulation_id;
//	number = message.number;

	return message;
}

void TRN::Engine::Node::erase_functors(const unsigned long long &simulation_id)
{
	TRACE_LOGGER;
	// INFORMATION_LOGGER <<   __FUNCTION__ ;
	if (handle->perceived_stimulus.find(simulation_id) != handle->perceived_stimulus.end())
		handle->perceived_stimulus.erase(simulation_id);
	if (handle->estimated_position.find(simulation_id) != handle->estimated_position.end())
		handle->estimated_position.erase(simulation_id);
	if (handle->mutator.find(simulation_id) != handle->mutator.end())
		handle->mutator.erase(simulation_id);
	if (handle->scheduler.find(simulation_id) != handle->scheduler.end())
		handle->scheduler.erase(simulation_id);
	if (handle->feedback_weights.find(simulation_id) != handle->feedback_weights.end())
		handle->feedback_weights.erase(simulation_id);
	if (handle->feedforward_weights.find(simulation_id) != handle->feedforward_weights.end())
		handle->feedforward_weights.erase(simulation_id);
	if (handle->readout.find(simulation_id) != handle->readout.end())
		handle->readout.erase(simulation_id);
	if (handle->recurrent.find(simulation_id) != handle->recurrent.end())
		handle->recurrent.erase(simulation_id);
}


void TRN::Engine::Node::body()
{
	TRACE_LOGGER;
	// INFORMATION_LOGGER <<   __FUNCTION__ ;
	std::string data;
	unsigned long long simulation_id = 0;
	size_t counter = 0;
	unsigned short number;
	std::string cause = "";
	//bool ack_required = true;

	auto locked = implementor.lock();
	if (!locked)
		throw std::runtime_error("Communicator is deleted");
	try
	{
		auto probed = locked->probe(handle->rank);
		if (!probed)
		{
			DEBUG_LOGGER << "Node " << handle->rank << " encountered an invalid probe. Terminating RX task";
			cancel();
		}
		else
		{
			auto tag = *probed;
			TRACE_LOGGER <<   "Node " << handle->rank << " received tag #" << tag ;
			switch (tag)
			{
				case TRN::Engine::START:
				{
				//	ack_required = false;
					auto message = locked->receive<TRN::Engine::START>(handle->rank);
					process(message);

					TRN::Engine::Message<TRN::Engine::Tag::CACHED> cached;
					cached.rank = TRN::Engine::Node::handle->rank;
					cached.checksums = handle->cache->cached();
					locked->send(cached, 0);
				}
				break;
				case TRN::Engine::STOP:
				{
					//ack_required = false;
			
					process(locked->receive<TRN::Engine::STOP>(handle->rank));
				}
				break;
				case TRN::Engine::QUIT:
				{
					//ack_required = false;
					process(locked->receive<TRN::Engine::QUIT>(handle->rank));
				}
				break;

				case TRN::Engine::ALLOCATE:
				{
					process(unpack<TRN::Engine::ALLOCATE>(locked, handle->rank, simulation_id, counter, number));
				}
				break;

				case TRN::Engine::DEALLOCATE:
				{
					process(unpack<TRN::Engine::DEALLOCATE>(locked, handle->rank, simulation_id, counter, number));
					erase_functors(simulation_id);
				}
				break;

				case TRN::Engine::TRAIN:
				{
					process(unpack<TRN::Engine::TRAIN>(locked, handle->rank, simulation_id, counter, number));
				}
				break;

				case TRN::Engine::TEST:
				{
					process(unpack<TRN::Engine::TEST>(locked, handle->rank, simulation_id, counter, number));
				}
				break;

				case TRN::Engine::DECLARE_SEQUENCE:
				{
					auto message = unpack<TRN::Engine::DECLARE_SEQUENCE>(locked, handle->rank, simulation_id, counter, number);
		
					if (!message.sequence.empty())
					{
						handle->cache->store(message.checksum, message.sequence);
						TRN::Engine::Message<TRN::Engine::Tag::CACHED> cached;
						cached.rank = TRN::Engine::Node::handle->rank;
						cached.checksums = {message.checksum};
						locked->send(cached, 0);
					}
					else
					{
						message.sequence = handle->cache->retrieve(message.checksum);
					}
					process(message);
				}
				break;

				case TRN::Engine::DECLARE_SET:
				{
					process(unpack<TRN::Engine::DECLARE_SET>(locked, handle->rank, simulation_id, counter, number));
				}
				break;

				case TRN::Engine::SETUP_STATES:
				{
					process(unpack<TRN::Engine::SETUP_STATES>(locked, handle->rank, simulation_id, counter, number));
				}
				break;

				case TRN::Engine::SETUP_WEIGHTS:
				{
					process(unpack<TRN::Engine::SETUP_WEIGHTS>(locked, handle->rank, simulation_id, counter, number));
				}
				break;

				case TRN::Engine::SETUP_PERFORMANCES:
				{
					process(unpack<TRN::Engine::SETUP_PERFORMANCES>(locked, handle->rank, simulation_id, counter, number));
				}
				break;

				case TRN::Engine::SETUP_SCHEDULING:
				{
					process(unpack<TRN::Engine::SETUP_SCHEDULING>(locked, handle->rank, simulation_id, counter, number));
				}
				break;

				case TRN::Engine::CONFIGURE_BEGIN:
				{

					process(unpack<TRN::Engine::CONFIGURE_BEGIN>(locked, handle->rank, simulation_id, counter, number));
				}
				break;

				case TRN::Engine::CONFIGURE_END:
				{

					process(unpack<TRN::Engine::CONFIGURE_END>(locked, handle->rank, simulation_id, counter, number));
				}
				break;

				case TRN::Engine::CONFIGURE_MEASUREMENT_READOUT_MEAN_SQUARE_ERROR:
				{
					process(unpack<TRN::Engine::CONFIGURE_MEASUREMENT_READOUT_MEAN_SQUARE_ERROR>(locked, handle->rank, simulation_id, counter, number));
				}
				break;

				case TRN::Engine::CONFIGURE_MEASUREMENT_READOUT_FRECHET_DISTANCE:
				{
					process(unpack<TRN::Engine::CONFIGURE_MEASUREMENT_READOUT_FRECHET_DISTANCE>(locked, handle->rank, simulation_id, counter, number));
				}
				break;

				case TRN::Engine::CONFIGURE_MEASUREMENT_READOUT_CUSTOM:
				{
					process(unpack<TRN::Engine::CONFIGURE_MEASUREMENT_READOUT_CUSTOM>(locked, handle->rank, simulation_id, counter, number));
				}
				break;

				case TRN::Engine::CONFIGURE_MEASUREMENT_POSITION_MEAN_SQUARE_ERROR:
				{
					process(unpack<TRN::Engine::CONFIGURE_MEASUREMENT_POSITION_MEAN_SQUARE_ERROR>(locked, handle->rank, simulation_id, counter, number));
				}
				break;

				case TRN::Engine::CONFIGURE_MEASUREMENT_POSITION_FRECHET_DISTANCE:
				{
					process(unpack<TRN::Engine::CONFIGURE_MEASUREMENT_POSITION_FRECHET_DISTANCE>(locked, handle->rank, simulation_id, counter, number));
				}
				break;

				case TRN::Engine::CONFIGURE_MEASUREMENT_POSITION_CUSTOM:
				{
					process(unpack<TRN::Engine::CONFIGURE_MEASUREMENT_POSITION_CUSTOM>(locked, handle->rank, simulation_id, counter, number));
				}
				break;

				case TRN::Engine::CONFIGURE_RESERVOIR_WIDROW_HOFF:
				{
					process(unpack<TRN::Engine::CONFIGURE_RESERVOIR_WIDROW_HOFF>(locked, handle->rank, simulation_id, counter, number));
				}
				break;
				case TRN::Engine::CONFIGURE_ENCODER_MODEL:
				{
					process(unpack<TRN::Engine::CONFIGURE_ENCODER_MODEL>(locked, handle->rank, simulation_id, counter, number));
				}
				break;
				case TRN::Engine::CONFIGURE_ENCODER_CUSTOM:
				{
					process(unpack<TRN::Engine::CONFIGURE_ENCODER_CUSTOM>(locked, handle->rank, simulation_id, counter, number));
				}
				break;
				case TRN::Engine::CONFIGURE_DECODER_LINEAR:
				{
					process(unpack<TRN::Engine::CONFIGURE_DECODER_LINEAR>(locked, handle->rank, simulation_id, counter, number));
				}
				break;
				case TRN::Engine::CONFIGURE_DECODER_KERNEL_MODEL:
				{
					process(unpack<TRN::Engine::CONFIGURE_DECODER_KERNEL_MODEL>(locked, handle->rank, simulation_id, counter, number));
				}
				break;
				case TRN::Engine::CONFIGURE_DECODER_KERNEL_MAP:
				{
					auto message = unpack<TRN::Engine::CONFIGURE_DECODER_KERNEL_MAP>(locked, handle->rank, simulation_id, counter, number);
					std::set<unsigned int> checksums;

					if (!message.response.second.empty())
					{
						handle->cache->store(message.response.first, message.response.second);
						checksums.insert(message.response.first);
					}
					else
					{
						message.response.second = handle->cache->retrieve(message.response.first);
					}
			
					if (!checksums.empty())
					{
						TRN::Engine::Message<TRN::Engine::Tag::CACHED> cached;
						cached.rank = TRN::Engine::Node::handle->rank;
						cached.checksums = checksums;
						locked->send(cached, 0);
					}
					process(message);
				}
				break;

				case TRN::Engine::CONFIGURE_LOOP_COPY:
				{
					process(unpack<TRN::Engine::CONFIGURE_LOOP_COPY>(locked, handle->rank, simulation_id, counter, number));
				}
				break;

				case TRN::Engine::CONFIGURE_LOOP_SPATIAL_FILTER:
				{
					process(unpack<TRN::Engine::CONFIGURE_LOOP_SPATIAL_FILTER>(locked, handle->rank, simulation_id, counter, number));
				}
				break;

				case TRN::Engine::CONFIGURE_LOOP_CUSTOM:
				{
					process(unpack<TRN::Engine::CONFIGURE_LOOP_CUSTOM>(locked, handle->rank, simulation_id, counter, number));
				}
				break;

				case TRN::Engine::CONFIGURE_SCHEDULER_TILED:
				{
					process(unpack<TRN::Engine::CONFIGURE_SCHEDULER_TILED>(locked, handle->rank, simulation_id, counter, number));
				}
				break;

				case TRN::Engine::CONFIGURE_SCHEDULER_SNIPPETS:
				{
					process(unpack<TRN::Engine::CONFIGURE_SCHEDULER_SNIPPETS>(locked, handle->rank, simulation_id, counter, number));
				}
				break;

				case TRN::Engine::CONFIGURE_SCHEDULER_CUSTOM:
				{
					process(unpack<TRN::Engine::CONFIGURE_SCHEDULER_CUSTOM>(locked, handle->rank, simulation_id, counter, number));
				}
				break;

				case TRN::Engine::CONFIGURE_MUTATOR_SHUFFLE:
				{
					process(unpack<TRN::Engine::CONFIGURE_MUTATOR_SHUFFLE>(locked, handle->rank, simulation_id, counter, number));
				}
				break;

				case TRN::Engine::CONFIGURE_MUTATOR_REVERSE:
				{
					process(unpack<TRN::Engine::CONFIGURE_MUTATOR_REVERSE>(locked, handle->rank, simulation_id, counter, number));
				}
				break;

				case TRN::Engine::CONFIGURE_MUTATOR_PUNCH:
				{
					process(unpack<TRN::Engine::CONFIGURE_MUTATOR_PUNCH>(locked, handle->rank, simulation_id, counter, number));
				}
				break;

				case TRN::Engine::CONFIGURE_MUTATOR_CUSTOM:
				{
					process(unpack<TRN::Engine::CONFIGURE_MUTATOR_CUSTOM>(locked, handle->rank, simulation_id, counter, number));
				}
				break;

				case TRN::Engine::CONFIGURE_FEEDFORWARD_UNIFORM:
				{
					process(unpack<TRN::Engine::CONFIGURE_FEEDFORWARD_UNIFORM>(locked, handle->rank, simulation_id, counter, number));
				}
				break;

				case TRN::Engine::CONFIGURE_FEEDFORWARD_GAUSSIAN:
				{
					process(unpack<TRN::Engine::CONFIGURE_FEEDFORWARD_GAUSSIAN>(locked, handle->rank, simulation_id, counter, number));
				}
				break;

				case TRN::Engine::CONFIGURE_FEEDFORWARD_CUSTOM:
				{

					process(unpack<TRN::Engine::CONFIGURE_FEEDFORWARD_CUSTOM>(locked, handle->rank, simulation_id, counter, number));
				}
				break;

				case TRN::Engine::CONFIGURE_RECURRENT_UNIFORM:
				{
					process(unpack<TRN::Engine::CONFIGURE_RECURRENT_UNIFORM>(locked, handle->rank, simulation_id, counter, number));
				}
				break;

				case TRN::Engine::CONFIGURE_RECURRENT_GAUSSIAN:
				{
					process(unpack<TRN::Engine::CONFIGURE_RECURRENT_GAUSSIAN>(locked, handle->rank, simulation_id, counter, number));
				}
				break;

				case TRN::Engine::CONFIGURE_RECURRENT_CUSTOM:
				{
					process(unpack<TRN::Engine::CONFIGURE_RECURRENT_CUSTOM>(locked, handle->rank, simulation_id, counter, number));
				}
				break;

				case TRN::Engine::CONFIGURE_READOUT_UNIFORM:
				{
					process(unpack<TRN::Engine::CONFIGURE_READOUT_UNIFORM>(locked, handle->rank, simulation_id, counter, number));
				}
				break;

				case TRN::Engine::CONFIGURE_READOUT_GAUSSIAN:
				{
					process(unpack<TRN::Engine::CONFIGURE_READOUT_GAUSSIAN>(locked, handle->rank, simulation_id, counter, number));
				}
				break;

				case TRN::Engine::CONFIGURE_READOUT_CUSTOM:
				{
					process(unpack<TRN::Engine::CONFIGURE_READOUT_CUSTOM>(locked, handle->rank, simulation_id, counter, number));
				}
				break;

				case TRN::Engine::POSITION:
				{
					process(unpack<TRN::Engine::POSITION>(locked, handle->rank, simulation_id, counter, number));
				}
				break;

				case TRN::Engine::STIMULUS:
				{
					process(unpack<TRN::Engine::STIMULUS>(locked, handle->rank, simulation_id, counter, number));
				}
				break;

				case TRN::Engine::SCHEDULING:
				{
					process(unpack<TRN::Engine::SCHEDULING>(locked, handle->rank, simulation_id, counter, number));
				}
				break;

				case TRN::Engine::FEEDFORWARD_WEIGHTS:
				{
					process(unpack<TRN::Engine::FEEDFORWARD_WEIGHTS>(locked, handle->rank, simulation_id, counter, number));
				}
				break;

				case TRN::Engine::RECURRENT_WEIGHTS:
				{
					process(unpack<TRN::Engine::RECURRENT_WEIGHTS>(locked, handle->rank, simulation_id, counter, number));
				}
				break;

				case TRN::Engine::READOUT_WEIGHTS:
				{
					process(unpack<TRN::Engine::READOUT_WEIGHTS>(locked, handle->rank, simulation_id, counter, number));
				}
				break;

				default:
					throw std::invalid_argument("unexpected tag " + std::to_string(tag));
			}
		}

	}
	catch (std::exception &e)
	{
		TRN::Engine::Message<TRN::Engine::LOG_ERROR> error;
		error.message = e.what();
		ERROR_LOGGER << error.message;
	
		locked->send(error, 0);
	}
}