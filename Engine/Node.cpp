#include "stdafx.h"
#include "Node_impl.h"

TRN::Engine::Node::Node(const std::shared_ptr<TRN::Engine::Communicator> &communicator, const int &rank) :
	TRN::Helper::Bridge<TRN::Engine::Communicator, std::weak_ptr>(communicator),
	handle(std::make_unique<Handle>())
{
	// std::cout << __FUNCTION__ << std::endl;
	handle->rank = rank;

	handle->disposed = false;
}


TRN::Engine::Node::~Node()
{
	auto communicator = TRN::Engine::Node::implementor.lock();
	if (communicator)
	{
		communicator->dispose();
	}
	// std::cout << __FUNCTION__ << std::endl;
	handle.reset();
}

void TRN::Engine::Node::dispose()
{
	
	join();
}

template <TRN::Engine::Tag tag>
static TRN::Engine::Message<tag> unpack(const std::shared_ptr<TRN::Engine::Communicator> &communicator, const int &rank, unsigned long long &id, size_t &counter, unsigned short &number)
{
	// std::cout << __FUNCTION__ << std::endl;
	TRN::Engine::Message<tag> message = communicator->receive<tag>(rank);

	id = message.id;
	counter = message.counter;
//	number = message.number;

	return message;
}

void TRN::Engine::Node::erase_functors(const unsigned long long &id)
{
	// std::cout << __FUNCTION__ << std::endl;
	if (handle->perceived_stimulus.find(id) != handle->perceived_stimulus.end())
		handle->perceived_stimulus.erase(id);
	if (handle->estimated_position.find(id) != handle->estimated_position.end())
		handle->estimated_position.erase(id);
	if (handle->mutator.find(id) != handle->mutator.end())
		handle->mutator.erase(id);
	if (handle->scheduler.find(id) != handle->scheduler.end())
		handle->scheduler.erase(id);
	if (handle->feedback_weights.find(id) != handle->feedback_weights.end())
		handle->feedback_weights.erase(id);
	if (handle->feedforward_weights.find(id) != handle->feedforward_weights.end())
		handle->feedforward_weights.erase(id);
	if (handle->readout.find(id) != handle->readout.end())
		handle->readout.erase(id);
	if (handle->recurrent.find(id) != handle->recurrent.end())
		handle->recurrent.erase(id);
}




void TRN::Engine::Node::body()
{
	// std::cout << __FUNCTION__ << std::endl;
	std::string data;
	unsigned long long id = 0;
	size_t counter = 0;
	unsigned short number;
	std::string cause = "";
	bool ack_required = true;

	auto locked = implementor.lock();
	if (!locked)
		throw std::runtime_error("Communicator is deleted");
	try
	{
		auto tag = locked->probe(handle->rank);
		//PrintThread{} << "node " << handle->rank << " received tag #" << tag << std::endl;
		switch (tag)
		{
			case TRN::Engine::START:
			{
				ack_required = false;
				auto message = locked->receive<TRN::Engine::START>(handle->rank);
				process(message);
		
			}
			break;
			case TRN::Engine::STOP:
			{
				ack_required = false;
				auto message = locked->receive<TRN::Engine::STOP>(handle->rank);
				process(message);

			}
			break;
			case TRN::Engine::QUIT:
			{
				ack_required = false;
				process(locked->receive<TRN::Engine::QUIT>(handle->rank));
			}
			break;

			case TRN::Engine::ALLOCATE:
			{
				process(unpack<TRN::Engine::ALLOCATE>(locked, handle->rank, id, counter, number));
			}
			break;

			case TRN::Engine::DEALLOCATE:
			{	
				process(unpack<TRN::Engine::DEALLOCATE>(locked, handle->rank, id, counter, number));
				erase_functors(id);
			}
			break;

			case TRN::Engine::TRAIN:
			{	
				process(unpack<TRN::Engine::TRAIN>(locked, handle->rank, id, counter, number));
			}
				break;

			case TRN::Engine::TEST:
			{	
				process(unpack<TRN::Engine::TEST>(locked, handle->rank, id, counter, number));
			}
				break;

			case TRN::Engine::DECLARE_SEQUENCE:
			{	
				process(unpack<TRN::Engine::DECLARE_SEQUENCE>(locked, handle->rank, id, counter, number));
			}
				break;

			case TRN::Engine::DECLARE_SET:
			{	
				process(unpack<TRN::Engine::DECLARE_SET>(locked, handle->rank, id, counter, number));
			}
				break;

			case TRN::Engine::SETUP_STATES:
			{	
				process(unpack<TRN::Engine::SETUP_STATES>(locked, handle->rank, id, counter, number));
			}
				break;

			case TRN::Engine::SETUP_WEIGHTS:
			{	
				process(unpack<TRN::Engine::SETUP_WEIGHTS>(locked, handle->rank, id, counter, number));
			}
				break;

			case TRN::Engine::SETUP_PERFORMANCES:
			{	
				process(unpack<TRN::Engine::SETUP_PERFORMANCES>(locked, handle->rank, id, counter, number));
			}
				break;

			case TRN::Engine::SETUP_SCHEDULING:
			{	
				process(unpack<TRN::Engine::SETUP_SCHEDULING>(locked, handle->rank, id, counter, number));
			}
				break;

			case TRN::Engine::CONFIGURE_BEGIN:
			{	
			
				process(unpack<TRN::Engine::CONFIGURE_BEGIN>(locked, handle->rank, id, counter, number));
			}
				break;

			case TRN::Engine::CONFIGURE_END:
			{	
		
				process(unpack<TRN::Engine::CONFIGURE_END>(locked, handle->rank, id, counter, number));
			}
				break;

			case TRN::Engine::CONFIGURE_MEASUREMENT_READOUT_MEAN_SQUARE_ERROR:
			{	
				process(unpack<TRN::Engine::CONFIGURE_MEASUREMENT_READOUT_MEAN_SQUARE_ERROR>(locked, handle->rank, id, counter, number));
			}
				break;
						
			case TRN::Engine::CONFIGURE_MEASUREMENT_READOUT_FRECHET_DISTANCE:
			{	
				process(unpack<TRN::Engine::CONFIGURE_MEASUREMENT_READOUT_FRECHET_DISTANCE>(locked, handle->rank, id, counter, number));
			}
				break;

			case TRN::Engine::CONFIGURE_MEASUREMENT_READOUT_CUSTOM:
			{	
				process(unpack<TRN::Engine::CONFIGURE_MEASUREMENT_READOUT_CUSTOM>(locked, handle->rank, id, counter, number));
			}
				break;

			case TRN::Engine::CONFIGURE_MEASUREMENT_POSITION_MEAN_SQUARE_ERROR:
			{	
				process(unpack<TRN::Engine::CONFIGURE_MEASUREMENT_POSITION_MEAN_SQUARE_ERROR>(locked, handle->rank, id, counter, number));
			}
				break;

			case TRN::Engine::CONFIGURE_MEASUREMENT_POSITION_FRECHET_DISTANCE:
			{	
				process(unpack<TRN::Engine::CONFIGURE_MEASUREMENT_POSITION_FRECHET_DISTANCE>(locked, handle->rank, id, counter, number));
			}
				break;

			case TRN::Engine::CONFIGURE_MEASUREMENT_POSITION_CUSTOM:
			{	
				process(unpack<TRN::Engine::CONFIGURE_MEASUREMENT_POSITION_CUSTOM>(locked, handle->rank, id, counter, number));
			}
				break;

			case TRN::Engine::CONFIGURE_RESERVOIR_WIDROW_HOFF:
			{	
				process(unpack<TRN::Engine::CONFIGURE_RESERVOIR_WIDROW_HOFF>(locked, handle->rank, id, counter, number));
			}
				break;

			case TRN::Engine::CONFIGURE_LOOP_COPY:
			{	
				process(unpack<TRN::Engine::CONFIGURE_LOOP_COPY>(locked, handle->rank, id, counter, number));
			}
				break;

			case TRN::Engine::CONFIGURE_LOOP_SPATIAL_FILTER:
			{	
				process(unpack<TRN::Engine::CONFIGURE_LOOP_SPATIAL_FILTER>(locked, handle->rank, id, counter, number));
			}
				break;

			case TRN::Engine::CONFIGURE_LOOP_CUSTOM:
			{	
				process(unpack<TRN::Engine::CONFIGURE_LOOP_CUSTOM>(locked, handle->rank, id, counter, number));
			}
				break;

			case TRN::Engine::CONFIGURE_SCHEDULER_TILED:
			{	
				process(unpack<TRN::Engine::CONFIGURE_SCHEDULER_TILED>(locked, handle->rank, id, counter, number));
			}
				break;

			case TRN::Engine::CONFIGURE_SCHEDULER_SNIPPETS:
			{	
				process(unpack<TRN::Engine::CONFIGURE_SCHEDULER_SNIPPETS>(locked, handle->rank, id, counter, number));
			}
				break;

			case TRN::Engine::CONFIGURE_SCHEDULER_CUSTOM:
			{	
				process(unpack<TRN::Engine::CONFIGURE_SCHEDULER_CUSTOM>(locked, handle->rank, id, counter, number));
			}
				break;

			case TRN::Engine::CONFIGURE_MUTATOR_SHUFFLE:
			{	
				process(unpack<TRN::Engine::CONFIGURE_MUTATOR_SHUFFLE>(locked, handle->rank, id, counter, number));
			}
				break;

			case TRN::Engine::CONFIGURE_MUTATOR_REVERSE:
			{	
				process(unpack<TRN::Engine::CONFIGURE_MUTATOR_REVERSE>(locked, handle->rank, id, counter, number));
			}
				break;

			case TRN::Engine::CONFIGURE_MUTATOR_PUNCH:
			{	
				process(unpack<TRN::Engine::CONFIGURE_MUTATOR_PUNCH>(locked, handle->rank, id, counter, number));
			}
				break;
				
			case TRN::Engine::CONFIGURE_MUTATOR_CUSTOM:
			{
				process(unpack<TRN::Engine::CONFIGURE_MUTATOR_CUSTOM>(locked, handle->rank, id, counter, number));
			}
				break;

			case TRN::Engine::CONFIGURE_FEEDFORWARD_UNIFORM:
			{	
				process(unpack<TRN::Engine::CONFIGURE_FEEDFORWARD_UNIFORM>(locked, handle->rank, id, counter, number));
			}
				break;
				
			case TRN::Engine::CONFIGURE_FEEDFORWARD_GAUSSIAN:
			{	
				process(unpack<TRN::Engine::CONFIGURE_FEEDFORWARD_GAUSSIAN>(locked, handle->rank, id, counter, number));
			}
				break;

			case TRN::Engine::CONFIGURE_FEEDFORWARD_CUSTOM:
			{

				process(unpack<TRN::Engine::CONFIGURE_FEEDFORWARD_CUSTOM>(locked, handle->rank, id, counter, number));
			}
				break;

			case TRN::Engine::CONFIGURE_FEEDBACK_UNIFORM:
			{
				process(unpack<TRN::Engine::CONFIGURE_FEEDBACK_UNIFORM>(locked, handle->rank, id, counter, number));
			}
				break;

			case TRN::Engine::CONFIGURE_FEEDBACK_GAUSSIAN:
			{	
				process(unpack<TRN::Engine::CONFIGURE_FEEDBACK_GAUSSIAN>(locked, handle->rank, id, counter, number));
			}
				break;

			case TRN::Engine::CONFIGURE_FEEDBACK_CUSTOM:
			{
				process(unpack<TRN::Engine::CONFIGURE_FEEDBACK_CUSTOM>(locked, handle->rank, id, counter, number));
			}
				break;

			case TRN::Engine::CONFIGURE_RECURRENT_UNIFORM:
			{
				process(unpack<TRN::Engine::CONFIGURE_RECURRENT_UNIFORM>(locked, handle->rank, id, counter, number));
			}
				break;

			case TRN::Engine::CONFIGURE_RECURRENT_GAUSSIAN:
			{	
				process(unpack<TRN::Engine::CONFIGURE_RECURRENT_GAUSSIAN>(locked, handle->rank, id, counter, number));
			}
				break;

			case TRN::Engine::CONFIGURE_RECURRENT_CUSTOM:
			{	
				process(unpack<TRN::Engine::CONFIGURE_RECURRENT_CUSTOM>(locked, handle->rank, id, counter, number));
			}
				break;
				
			case TRN::Engine::CONFIGURE_READOUT_UNIFORM:
			{	
				process(unpack<TRN::Engine::CONFIGURE_READOUT_UNIFORM>(locked, handle->rank, id, counter, number));
			}
				break;

			case TRN::Engine::CONFIGURE_READOUT_GAUSSIAN:
			{	
				process(unpack<TRN::Engine::CONFIGURE_READOUT_GAUSSIAN>(locked, handle->rank, id, counter, number));
			}
				break;

			case TRN::Engine::CONFIGURE_READOUT_CUSTOM:
			{	
				process(unpack<TRN::Engine::CONFIGURE_READOUT_CUSTOM>(locked, handle->rank, id, counter, number));
			}
				break;

			case TRN::Engine::POSITION:
			{	
				process(unpack<TRN::Engine::POSITION>(locked, handle->rank, id, counter, number));
			}
				break;

			case TRN::Engine::STIMULUS:
			{	
				process(unpack<TRN::Engine::STIMULUS>(locked, handle->rank, id, counter, number));
			}
				break;

			case TRN::Engine::SCHEDULING:
			{	
				process(unpack<TRN::Engine::SCHEDULING>(locked, handle->rank, id, counter, number));
			}
				break;

			case TRN::Engine::FEEDFORWARD_WEIGHTS:
			{	
				process(unpack<TRN::Engine::FEEDFORWARD_WEIGHTS>(locked, handle->rank, id, counter, number));
			}
				break;

			case TRN::Engine::RECURRENT_WEIGHTS:
			{	
				process(unpack<TRN::Engine::RECURRENT_WEIGHTS>(locked, handle->rank, id, counter, number));
			}
				break;

			case TRN::Engine::FEEDBACK_WEIGHTS:
			{	
				process(unpack<TRN::Engine::FEEDBACK_WEIGHTS>(locked, handle->rank, id, counter, number));
			}
				break;

			case TRN::Engine::READOUT_WEIGHTS:
			{
				process(unpack<TRN::Engine::READOUT_WEIGHTS>(locked, handle->rank, id, counter, number));
			}
				break;

			default:
				throw std::invalid_argument("unexpected tag " + std::to_string(tag));
		}
		if (ack_required)
		{
			TRN::Engine::Message<TRN::Engine::ACK> ack;

			ack.id = id;
			//ack.number = number;
			ack.counter = counter;
			ack.cause = cause;
			ack.success = cause.empty();
			//PrintThread{} << handle->name << " sent ack to broker" << std::endl;
			locked->send(ack, 0);
		}

	}
	catch (std::exception &e)
	{
		TRN::Engine::Message<TRN::Engine::LOG_ERROR> error;
		std::cout << e.what() << std::endl;
		std::cerr << e.what() << std::endl;
		error.message = e.what();
		locked->send(error, 0);
		//stop();
	}
}