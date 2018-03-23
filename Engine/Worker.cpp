#include "stdafx.h"
#include "Worker_impl.h"
#include "Simulator/Basic.h"
#include "Model/Simulator.h"
#include "Model/Reservoir.h"
#include "Model/Decoder.h"
#include "Model/Loop.h"
#include "Model/Scheduler.h"
#include "Model/Initializer.h"
#include "Model/Measurement.h"
#include "Model/Mutator.h"
#include "Node_impl.h"
#include "Helper/Logger.h"


static std::mutex cache_mutex;
static std::map<std::pair<unsigned int, unsigned int>, std::shared_ptr<TRN::Core::Matrix>> cache;

TRN::Engine::Worker::Worker(const std::shared_ptr<TRN::Engine::Communicator> &communicator, const int &rank, const std::shared_ptr<TRN::Backend::Driver> &driver) :
	TRN::Helper::Bridge<TRN::Backend::Driver>(driver),
	TRN::Engine::Node(communicator, rank),
	handle(std::make_unique<Handle>())
{
	TRACE_LOGGER;
	// INFORMATION_LOGGER <<   __FUNCTION__ ;
	//TRN::Engine::Node::handle->name = "WORKER";


}

TRN::Engine::Worker::~Worker()
{
	TRACE_LOGGER;
	// INFORMATION_LOGGER <<   __FUNCTION__ ;
	handle.reset();
}

void TRN::Engine::Worker::send_configured(const unsigned long long &simulation_id)
{
	TRACE_LOGGER;
	// INFORMATION_LOGGER <<   __FUNCTION__ ;
	if (handle->configured_required[simulation_id] == true && handle->remaining_initializations[simulation_id] == 0)
	{
		if (handle->simulators.find(simulation_id) == handle->simulators.end())
			throw std::invalid_argument("Simulator #" + std::to_string(simulation_id) + "does not exist");
		handle->simulators[simulation_id]->get_reservoir()->initialize();
		TRN::Engine::Message<TRN::Engine::CONFIGURED> configured;


		configured.simulation_id = simulation_id;
		
		auto locked = TRN::Engine::Node::get_implementor().lock();
		if (locked)
			locked->send(configured, 0);
		handle->configured_required[simulation_id] = false;
	}
}

void TRN::Engine::Worker::initialize()
{
	TRACE_LOGGER;
	// INFORMATION_LOGGER <<   __FUNCTION__ ;
	TRN::Helper::Bridge<TRN::Backend::Driver>::implementor->toggle();
}


void TRN::Engine::Worker::uninitialize()
{
	TRACE_LOGGER;
	TRN::Engine::Message<TRN::Engine::Tag::TERMINATED> terminated;

	terminated.rank = TRN::Engine::Node::handle->rank;
	auto communicator = TRN::Engine::Node::implementor.lock();
	if (communicator)
		communicator->send(terminated, 0);
	TRN::Helper::Bridge<TRN::Backend::Driver>::implementor->dispose();

}

void TRN::Engine::Worker::process(const TRN::Engine::Message<TRN::Engine::Tag::QUIT> &message)
{
	TRACE_LOGGER;
	if (handle->frontends.find(message.number) == handle->frontends.end())
		throw std::runtime_error("Frontend " + std::to_string(message.number) + " is not declared");
	if (!message.terminate)
	{
		handle->quit_not_required.insert(message.number);
	}
	TRN::Engine::Message<TRN::Engine::EXIT> exit;

	

	exit.rank = TRN::Engine::Node::handle->rank;
	exit.number = message.number;

	//INFORMATION_LOGGER <<   "SEND EXIT rank "<< exit.rank ;

	auto communicator = TRN::Engine::Node::implementor.lock();
	if (communicator)
		communicator->send(exit, 0);
}

void TRN::Engine::Worker::process(const TRN::Engine::Message<TRN::Engine::Tag::START> &message)
{
	TRACE_LOGGER;
	if (handle->frontends.find(message.number) != handle->frontends.end())
		throw std::runtime_error("Frontend " + std::to_string(message.number) + " is already declared");
	handle->frontends.insert(message.number);
	INFORMATION_LOGGER <<   "Worker #" << TRN::Engine::Node::handle->rank << " START " << message.number ;


	auto communicator = TRN::Engine::Node::implementor.lock();
	TRN::Engine::Message<TRN::Engine::Tag::WORKER> worker;
	worker.host = communicator->host();
	worker.rank = TRN::Engine::Node::handle->rank;
	worker.index = TRN::Helper::Bridge<TRN::Backend::Driver>::implementor->index();
	worker.name = TRN::Helper::Bridge<TRN::Backend::Driver>::implementor->name();


	if (communicator)
	{
		communicator->send(worker, 0);
	}



}

void TRN::Engine::Worker::process(const TRN::Engine::Message<TRN::Engine::Tag::STOP> &message)
{
	TRACE_LOGGER;
//	INFORMATION_LOGGER <<   "STOP worker " << TRN::Engine::Node::handle->rank << ", frontend " << message.number ;
	if (handle->frontends.find(message.number) == handle->frontends.end())
		throw std::runtime_error("Frontend " + std::to_string(message.number) + " is not declared");
	INFORMATION_LOGGER <<   "Worker #" << TRN::Engine::Node::handle->rank << " STOP " << message.number ;
	handle->frontends.erase(message.number);

	if (handle->quit_not_required.empty() && handle->frontends.empty())
	{
		INFORMATION_LOGGER <<   "No more frontends. Stopping worker" ;
		stop();
	}
}
void TRN::Engine::Worker::process(const TRN::Engine::Message<TRN::Engine::Tag::ALLOCATE> &message)
{
	TRACE_LOGGER;
	//INFORMATION_LOGGER <<   "allocate " << message.simulation_id << "on rank " << TRN::Engine::Node::handle->rank ;
	// INFORMATION_LOGGER <<   __FUNCTION__ ;
	if (handle->simulators.find(message.simulation_id) != handle->simulators.end())
		throw std::invalid_argument("Simulator #" + std::to_string(message.simulation_id) + "already exists");

	auto basic = TRN::Model::Simulator::Basic::create(
		[this, message]()
	{
		TRN::Engine::Message<TRN::Engine::TRAINED> trained;

		trained.simulation_id = message.simulation_id;

		 auto locked = TRN::Engine::Node::implementor.lock();
		if (locked)
			locked->send(trained, 0);
	},
	[this, message]()
	{
		TRN::Engine::Message<TRN::Engine::PRIMED> primed;

		primed.simulation_id = message.simulation_id;
		 auto locked = TRN::Engine::Node::implementor.lock();
		if (locked)
			locked->send(primed, 0);
	}
		,
	[this, message]()
	{
		TRN::Engine::Message<TRN::Engine::TESTED> tested;

		tested.simulation_id = message.simulation_id;
		 auto locked = TRN::Engine::Node::implementor.lock();
		if (locked)
			locked->send(tested, 0);
	}
	);
	handle->simulators[message.simulation_id] = basic;
	handle->configured_required[message.simulation_id] = false;
	handle->remaining_initializations[message.simulation_id] = 0;

	TRN::Engine::Message<TRN::Engine::ALLOCATED> allocated;

	allocated.simulation_id = message.simulation_id;
	auto locked = TRN::Engine::Node::implementor.lock();
	if (locked)
		locked->send(allocated, 0);
}
void TRN::Engine::Worker::process(const TRN::Engine::Message<TRN::Engine::Tag::DEALLOCATE> &message)
{
	TRACE_LOGGER;
	// 
	//INFORMATION_LOGGER <<   "deallocate " << message.simulation_id << " on rank " << TRN::Engine::Node::handle->rank ;
	if (handle->simulators.find(message.simulation_id) == handle->simulators.end())
	{
		throw std::invalid_argument("Simulator #" + std::to_string(message.simulation_id) + " does not exist");
	}
	handle->simulators[message.simulation_id]->uninitialize();
	handle->simulators.erase(message.simulation_id);
	handle->remaining_initializations.erase(message.simulation_id);
	handle->configured_required.erase(message.simulation_id);
//	INFORMATION_LOGGER <<   "deallocated " << message.simulation_id << " on rank " << TRN::Engine::Node::handle->rank ;
	TRN::Engine::Message<TRN::Engine::DEALLOCATED> deallocated;

	deallocated.simulation_id = message.simulation_id;
	auto locked = TRN::Engine::Node::implementor.lock();
	if (locked)
		locked->send(deallocated, 0);
}
void TRN::Engine::Worker::process(const TRN::Engine::Message<TRN::Engine::Tag::TRAIN> &message)
{
	TRACE_LOGGER;
	// INFORMATION_LOGGER <<   __FUNCTION__ ;
	if (handle->simulators.find(message.simulation_id) == handle->simulators.end())
		throw std::invalid_argument("Simulator #" + std::to_string(message.simulation_id) + " does not exist");
	handle->simulators[message.simulation_id]->train(message.evaluation_id, message.label, message.incoming, message.expected, message.reset_readout);
}
void TRN::Engine::Worker::process(const TRN::Engine::Message<TRN::Engine::Tag::TEST> &message)
{
	TRACE_LOGGER;
	// INFORMATION_LOGGER <<   __FUNCTION__ ;
	if (handle->simulators.find(message.simulation_id) == handle->simulators.end())
		throw std::invalid_argument("Simulator #" + std::to_string(message.simulation_id) + " does not exist");
	handle->simulators[message.simulation_id]->test(message.evaluation_id, message.label, message.incoming, message.expected, message.preamble, message.autonomous, message.supplementary_generations);

}
void TRN::Engine::Worker::process(const TRN::Engine::Message<TRN::Engine::Tag::DECLARE_SEQUENCE> &message)
{
	TRACE_LOGGER;
	// INFORMATION_LOGGER <<   __FUNCTION__ ;
	if (handle->simulators.find(message.simulation_id) == handle->simulators.end())
		throw std::invalid_argument("Simulator #" + std::to_string(message.simulation_id) + " does not exist");
	std::shared_ptr<TRN::Core::Matrix> matrix;
	auto key = std::make_pair(TRN::Helper::Bridge<TRN::Backend::Driver>::implementor->index(), message.checksum);
	std::unique_lock<std::mutex> guard(cache_mutex);
	if (cache.find(key) == cache.end())
	{
		cache[key] = TRN::Core::Matrix::create(TRN::Helper::Bridge<TRN::Backend::Driver>::implementor, message.sequence, message.observations, message.sequence.size() / message.observations);
		DEBUG_LOGGER << "Device matrix having checksum 0x" << std::hex << message.checksum << " is stored in process cache";
	}
	else
	{
		DEBUG_LOGGER << "Device matrix having checksum 0x" << std::hex << message.checksum << " had been retreived from process cache";
	}
	matrix = cache[key];
	guard.unlock();
	handle->simulators[message.simulation_id]->declare(message.label, message.tag, matrix);
}
void TRN::Engine::Worker::process(const TRN::Engine::Message<TRN::Engine::Tag::DECLARE_SET> &message)
{
	TRACE_LOGGER;
	// INFORMATION_LOGGER <<   __FUNCTION__ ;
	if (handle->simulators.find(message.simulation_id) == handle->simulators.end())
		throw std::invalid_argument("Simulator #" + std::to_string(message.simulation_id) + " does not exist");
	std::vector<std::shared_ptr<TRN::Core::Matrix>> sequences;
	for (auto sequence_label : message.labels)
	{
		sequences.push_back(handle->simulators[message.simulation_id]->retrieve_sequence(sequence_label, message.tag));
	}

	handle->simulators[message.simulation_id]->declare(message.label, message.tag, TRN::Core::Set::create(TRN::Helper::Bridge<TRN::Backend::Driver>::implementor, sequences));
}
void TRN::Engine::Worker::process(const TRN::Engine::Message<TRN::Engine::Tag::SETUP_STATES> &message) 
{
	TRACE_LOGGER;
	// INFORMATION_LOGGER <<   __FUNCTION__ ;
	if (handle->simulators.find(message.simulation_id) == handle->simulators.end())
		throw std::invalid_argument("Simulator #" + std::to_string(message.simulation_id) + " does not exist");
	if (!handle->simulators[message.simulation_id]->get_reservoir())
		throw std::logic_error("Simulator #" + std::to_string(message.simulation_id) + " does not not have a reservoir to decorate");
	auto decorator = TRN::Model::Simulator::States::create(handle->simulators[message.simulation_id], [this, message]
	(const unsigned long long &evaluation_id, const std::string &phase, const std::string &label, const std::size_t &batch, const std::vector<float> &samples, const std::size_t &rows, const std::size_t &cols)
	{
		TRN::Engine::Message<TRN::Engine::STATES> states;

		states.simulation_id = message.simulation_id;
		states.label = label;
		states.phase = phase;
		states.batch = batch;
		states.evaluation_id = evaluation_id;
		states.elements = samples;
		states.rows = rows;
		states.cols = cols;

		 auto locked = TRN::Engine::Node::implementor.lock();
		if (locked)
			locked->send(states, 0);
	}, message.train, message.prime, message.generate);
	handle->simulators[message.simulation_id]->attach(decorator);
	handle->simulators[message.simulation_id] = decorator;
}
void TRN::Engine::Worker::process(const TRN::Engine::Message<TRN::Engine::Tag::SETUP_WEIGHTS> &message)
{
	TRACE_LOGGER;
	// INFORMATION_LOGGER <<   __FUNCTION__ ;
	if (handle->simulators.find(message.simulation_id) == handle->simulators.end())
		throw std::invalid_argument("Simulator #" + std::to_string(message.simulation_id) + " does not exist");
	if (!handle->simulators[message.simulation_id]->get_reservoir())
		throw std::logic_error("Simulator #" + std::to_string(message.simulation_id) + " does not not have a reservoir to decorate");
	auto decorator = TRN::Model::Simulator::Weights::create(handle->simulators[message.simulation_id], [this, message]
	(const unsigned long long &evaluation_id, const std::string &phase, const std::string &label, const std::size_t &batch, const std::vector<float> &samples, const std::size_t &rows, const std::size_t &cols)
	{
		TRN::Engine::Message<TRN::Engine::WEIGHTS> weights;

		weights.simulation_id = message.simulation_id;
		weights.phase = phase;
		weights.label = label;
		weights.evaluation_id = evaluation_id;
		weights.batch = batch;
		weights.elements = samples;
		weights.rows = rows;
		weights.cols = cols;

		 auto locked = TRN::Engine::Node::implementor.lock();
		if (locked)
			locked->send(weights, 0);
	}, message.initialization, message.train);
	handle->simulators[message.simulation_id]->attach(decorator);
	handle->simulators[message.simulation_id] = decorator;
}
void TRN::Engine::Worker::process(const TRN::Engine::Message<TRN::Engine::Tag::SETUP_PERFORMANCES> &message)
{
	TRACE_LOGGER;
	// INFORMATION_LOGGER <<   __FUNCTION__ ;
	if (handle->simulators.find(message.simulation_id) == handle->simulators.end())
		throw std::invalid_argument("Simulator #" + std::to_string(message.simulation_id) + " does not exist");
	if (!handle->simulators[message.simulation_id]->get_reservoir())
		throw std::logic_error("Simulator #" + std::to_string(message.simulation_id) + " does not not have a reservoir to decorate");
	auto decorator = TRN::Model::Simulator::Performances::create(handle->simulators[message.simulation_id], [this, message]
	(const unsigned long long &evaluation_id, const std::string &phase, const float &cycles_per_second, const float &gflops_per_second)
	{
		TRN::Engine::Message<TRN::Engine::PERFORMANCES> performances;

		performances.simulation_id = message.simulation_id;
		performances.phase = phase;

		performances.evaluation_id = evaluation_id;
		performances.phase = phase;
		performances.cycles_per_second = cycles_per_second;
		performances.gflops_per_second = gflops_per_second;
		 auto locked = TRN::Engine::Node::implementor.lock();
		if (locked)
			locked->send(performances, 0);
	}, message.train, message.prime, message.generate);
	handle->simulators[message.simulation_id]->attach(decorator);
	handle->simulators[message.simulation_id] = decorator;
}
void TRN::Engine::Worker::process(const TRN::Engine::Message<TRN::Engine::Tag::SETUP_SCHEDULING> &message)
{
	TRACE_LOGGER;
	// INFORMATION_LOGGER <<   __FUNCTION__ ;
	if (handle->simulators.find(message.simulation_id) == handle->simulators.end())
		throw std::invalid_argument("Simulator #" + std::to_string(message.simulation_id) + " does not exist");
	if (!handle->simulators[message.simulation_id]->get_reservoir())
		throw std::logic_error("Simulator #" + std::to_string(message.simulation_id) + " does not not have a reservoir to decorate");
	auto decorator = TRN::Model::Simulator::Scheduling::create(handle->simulators[message.simulation_id], [this, message]
	(const unsigned long long &evaluation_id, const std::vector<int> &offsets, const std::vector<int> &durations)
	{
		TRN::Engine::Message<TRN::Engine::SCHEDULING> scheduling;

		scheduling.simulation_id = message.simulation_id;
		scheduling.evaluation_id = evaluation_id;
		scheduling.offsets = offsets;
		scheduling.durations = durations;
		scheduling.is_from_mutator = false;
		 auto locked = TRN::Engine::Node::implementor.lock();
		if (locked)
			locked->send(scheduling, 0);
	});
	//handle->simulators[message.simulation_id]->attach(decorator);
	handle->simulators[message.simulation_id] = decorator;
}
void TRN::Engine::Worker::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_BEGIN> &message)
{
	TRACE_LOGGER;
	// INFORMATION_LOGGER <<   __FUNCTION__ ;
	if (handle->simulators.find(message.simulation_id) == handle->simulators.end())
		throw std::invalid_argument("Simulator #" + std::to_string(message.simulation_id) + " does not exist");
	handle->simulators[message.simulation_id]->uninitialize();
}
void TRN::Engine::Worker::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_END> &message)
{
	TRACE_LOGGER;
	// INFORMATION_LOGGER <<   __FUNCTION__ ;
	if (handle->simulators.find(message.simulation_id) == handle->simulators.end())
		throw std::invalid_argument("Simulator #" + std::to_string(message.simulation_id) + " does not exist");
	handle->simulators[message.simulation_id]->initialize();

	handle->configured_required[message.simulation_id] = true;
	send_configured(message.simulation_id);
}
void TRN::Engine::Worker::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_MEASUREMENT_READOUT_MEAN_SQUARE_ERROR> &message)
{
	TRACE_LOGGER;
	// INFORMATION_LOGGER <<   __FUNCTION__ ;
	if (handle->simulators.find(message.simulation_id) == handle->simulators.end())
		throw std::invalid_argument("Simulator #" + std::to_string(message.simulation_id) + " does not exist");
	handle->simulators[message.simulation_id]->append_measurement(
		TRN::Model::Measurement::Sequence::create(
			TRN::Model::Measurement::MeanSquareError::create(TRN::Helper::Bridge<TRN::Backend::Driver>::implementor,
				[this, message](const unsigned long long &evaluation_id, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols)
	{
		TRN::Engine::Message<TRN::Engine::MEASUREMENT_READOUT_MEAN_SQUARE_ERROR> measurement;

		measurement.simulation_id = message.simulation_id;
		measurement.evaluation_id = evaluation_id;
		measurement.elements = values;

		measurement.rows = rows;
		measurement.cols = cols;

		 auto locked = TRN::Engine::Node::implementor.lock();
		if (locked)
			locked->send(measurement, 0);
	}), message.batch_size));
}
void TRN::Engine::Worker::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_MEASUREMENT_READOUT_FRECHET_DISTANCE> &message)
{
	TRACE_LOGGER;
	// INFORMATION_LOGGER <<   __FUNCTION__ ;
	if (handle->simulators.find(message.simulation_id) == handle->simulators.end())
		throw std::invalid_argument("Simulator #" + std::to_string(message.simulation_id) + " does not exist");
	handle->simulators[message.simulation_id]->append_measurement(
		TRN::Model::Measurement::Sequence::create(
			TRN::Model::Measurement::FrechetDistance::create(TRN::Helper::Bridge<TRN::Backend::Driver>::implementor, message.norm, message.aggregator,
				[this, message](const unsigned long long &evaluation_id, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols)
	{
		TRN::Engine::Message<TRN::Engine::MEASUREMENT_READOUT_FRECHET_DISTANCE> measurement;

		measurement.simulation_id = message.simulation_id;
		measurement.evaluation_id = evaluation_id;
		measurement.elements = values;
		measurement.rows = rows;
		measurement.cols = cols;

		 auto locked = TRN::Engine::Node::implementor.lock();
		if (locked)
			locked->send(measurement, 0);
	}), message.batch_size));
}
void TRN::Engine::Worker::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_MEASUREMENT_READOUT_CUSTOM> &message)
{
	TRACE_LOGGER;
	// INFORMATION_LOGGER <<   __FUNCTION__ ;
	if (handle->simulators.find(message.simulation_id) == handle->simulators.end())
		throw std::invalid_argument("Simulator #" + std::to_string(message.simulation_id) + " does not exist");
	handle->simulators[message.simulation_id]->append_measurement(
		TRN::Model::Measurement::Sequence::create(
			TRN::Model::Measurement::Custom::create(TRN::Helper::Bridge<TRN::Backend::Driver>::implementor,
				[this, message](const unsigned long long &evaluation_id, const std::vector<float> &primed, const std::vector<float> &predicted, const std::vector<float> &expected, const std::size_t &preamble, const std::size_t &pages, const std::size_t &rows, const  std::size_t &cols)
	{
		TRN::Engine::Message<TRN::Engine::MEASUREMENT_READOUT_CUSTOM> measurement;

		measurement.simulation_id = message.simulation_id;
		measurement.evaluation_id = evaluation_id;
		measurement.primed = primed;
		measurement.elements = predicted;
		measurement.expected = expected;
		measurement.preamble = preamble;
		measurement.matrices = pages;
		measurement.rows = rows;
		measurement.cols = cols;

		 auto locked = TRN::Engine::Node::implementor.lock();
		if (locked)
			locked->send(measurement, 0);
	}), message.batch_size));
}
void TRN::Engine::Worker::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_MEASUREMENT_POSITION_MEAN_SQUARE_ERROR> &message)
{
	TRACE_LOGGER;
	// INFORMATION_LOGGER <<   __FUNCTION__ ;
	if (handle->simulators.find(message.simulation_id) == handle->simulators.end())
		throw std::invalid_argument("Simulator #" + std::to_string(message.simulation_id) + " does not exist");
	handle->simulators[message.simulation_id]->append_measurement(
		TRN::Model::Measurement::Position::create(
			TRN::Model::Measurement::MeanSquareError::create(TRN::Helper::Bridge<TRN::Backend::Driver>::implementor,
				[this, message](const unsigned long long &evaluation_id, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols)
	{
		TRN::Engine::Message<TRN::Engine::MEASUREMENT_POSITION_MEAN_SQUARE_ERROR> measurement;

		measurement.simulation_id = message.simulation_id;
		measurement.evaluation_id = evaluation_id;
		measurement.elements = values;
		measurement.rows = rows;
		measurement.cols = cols;

		 auto locked = TRN::Engine::Node::implementor.lock();
		if (locked)
			locked->send(measurement, 0);
	}), message.batch_size));
}
void TRN::Engine::Worker::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_MEASUREMENT_POSITION_FRECHET_DISTANCE> &message)
{
	TRACE_LOGGER;
	// INFORMATION_LOGGER <<   __FUNCTION__ ;
	if (handle->simulators.find(message.simulation_id) == handle->simulators.end())
		throw std::invalid_argument("Simulator #" + std::to_string(message.simulation_id) + " does not exist");
	handle->simulators[message.simulation_id]->append_measurement(
		TRN::Model::Measurement::Position::create(
			TRN::Model::Measurement::FrechetDistance::create(TRN::Helper::Bridge<TRN::Backend::Driver>::implementor, message.norm, message.aggregator,
				[this, message](const unsigned long long &evaluation_id, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols)
	{
		TRN::Engine::Message<TRN::Engine::MEASUREMENT_POSITION_FRECHET_DISTANCE> measurement;

		measurement.simulation_id = message.simulation_id;
		measurement.evaluation_id = evaluation_id;
		measurement.elements = values;
		measurement.rows = rows;
		measurement.cols = cols;

		 auto locked = TRN::Engine::Node::implementor.lock();
		if (locked)
			locked->send(measurement, 0);
	}), message.batch_size));
}
void TRN::Engine::Worker::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_MEASUREMENT_POSITION_CUSTOM> &message)
{
	TRACE_LOGGER;
	// INFORMATION_LOGGER <<   __FUNCTION__ ;
	if (handle->simulators.find(message.simulation_id) == handle->simulators.end())
		throw std::invalid_argument("Simulator #" + std::to_string(message.simulation_id) + " does not exist");
	handle->simulators[message.simulation_id]->append_measurement(
		TRN::Model::Measurement::Position::create(
			TRN::Model::Measurement::Custom::create(TRN::Helper::Bridge<TRN::Backend::Driver>::implementor,
				[this, message](const unsigned long long &evaluation_id, const std::vector<float> &primed, const std::vector<float> &predicted, const std::vector<float> &expected, const std::size_t &preamble, const std::size_t &pages, const std::size_t &rows, const  std::size_t &cols)
	{
		TRN::Engine::Message<TRN::Engine::MEASUREMENT_POSITION_CUSTOM> measurement;

		measurement.simulation_id = message.simulation_id;
		measurement.evaluation_id = evaluation_id;
		measurement.elements = predicted;
		measurement.expected = expected;
		measurement.primed = primed;
		measurement.matrices = pages;
		measurement.rows = rows;
		measurement.cols = cols;
		measurement.preamble = preamble;
		 auto locked = TRN::Engine::Node::implementor.lock();
		if (locked)
			locked->send(measurement, 0);
	}), message.batch_size));
}
void TRN::Engine::Worker::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_RESERVOIR_WIDROW_HOFF> &message)
{
	TRACE_LOGGER;
	// INFORMATION_LOGGER <<   __FUNCTION__ ;
	if (handle->simulators.find(message.simulation_id) == handle->simulators.end())
		throw std::invalid_argument("Simulator #" + std::to_string(message.simulation_id) + " does not exist");
	handle->simulators[message.simulation_id]->set_reservoir(TRN::Model::Reservoir::WidrowHoff::create(TRN::Helper::Bridge<TRN::Backend::Driver>::implementor, message.stimulus_size, message.prediction_size, message.reservoir_size, message.leak_rate, message.initial_state_scale, message.learning_rate, message.seed, message.batch_size, message.mini_batch_size));
}

void TRN::Engine::Worker::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_DECODER_LINEAR> &message)
{
	TRACE_LOGGER;
	// INFORMATION_LOGGER <<   __FUNCTION__ ;
	if (handle->simulators.find(message.simulation_id) == handle->simulators.end())
		throw std::invalid_argument("Simulator #" + std::to_string(message.simulation_id) + " does not exist");
	handle->simulators[message.simulation_id]->set_decoder(TRN::Model::Decoder::Linear::create(TRN::Helper::Bridge<TRN::Backend::Driver>::implementor,
		message.batch_size, message.stimulus_size,
		TRN::Core::Matrix::create(TRN::Helper::Bridge<TRN::Backend::Driver>::implementor, message.cx, 1, message.cx.size()),
		TRN::Core::Matrix::create(TRN::Helper::Bridge<TRN::Backend::Driver>::implementor, message.cy, 1, message.cy.size())));
	
}
void TRN::Engine::Worker::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_DECODER_KERNEL_MAP> &message)
{
	TRACE_LOGGER;
	// INFORMATION_LOGGER <<   __FUNCTION__ ;
	if (handle->simulators.find(message.simulation_id) == handle->simulators.end())
		throw std::invalid_argument("Simulator #" + std::to_string(message.simulation_id) + " does not exist");

	std::shared_ptr<TRN::Core::Matrix> firing_rate_map;

	auto response_key = std::make_pair(TRN::Helper::Bridge<TRN::Backend::Driver>::implementor->index(), message.response.first);
	std::unique_lock<std::mutex> guard(cache_mutex);
	if (cache.find(response_key) == cache.end())
	{

		cache[response_key] = TRN::Core::Matrix::create(TRN::Helper::Bridge<TRN::Backend::Driver>::implementor, message.response.second, message.rows * message.cols, message.stimulus_size);

		DEBUG_LOGGER << "Device matrix having checksum 0x" << std::hex << message.response.first << " is stored in process cache";
	}
	else
	{
		DEBUG_LOGGER << "Device matrix having checksum 0x" << std::hex << message.response.first << " had been retreived from process cache";
	}

	firing_rate_map = cache[response_key];

	handle->simulators[message.simulation_id]->set_decoder(TRN::Model::Decoder::Kernel::Map::create(TRN::Helper::Bridge<TRN::Backend::Driver>::implementor,
		message.batch_size, message.stimulus_size,
		message.rows, message.cols, message.x, message.y, message.sigma, message.radius, message.angle, message.scale, message.seed, 
		firing_rate_map));
}
void TRN::Engine::Worker::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_DECODER_KERNEL_MODEL> &message)
{
	TRACE_LOGGER;
	// INFORMATION_LOGGER <<   __FUNCTION__ ;
	if (handle->simulators.find(message.simulation_id) == handle->simulators.end())
		throw std::invalid_argument("Simulator #" + std::to_string(message.simulation_id) + " does not exist");
	handle->simulators[message.simulation_id]->set_decoder(TRN::Model::Decoder::Kernel::Model::create(TRN::Helper::Bridge<TRN::Backend::Driver>::implementor,
		message.batch_size, message.stimulus_size,
		message.rows, message.cols, message.x, message.y, message.sigma, message.radius, message.angle, message.scale, message.seed,
		TRN::Core::Matrix::create(TRN::Helper::Bridge<TRN::Backend::Driver>::implementor, message.cx, 1, message.cx.size()),
		TRN::Core::Matrix::create(TRN::Helper::Bridge<TRN::Backend::Driver>::implementor, message.cy, 1, message.cy.size()),
		TRN::Core::Matrix::create(TRN::Helper::Bridge<TRN::Backend::Driver>::implementor, message.K, 1, message.K.size())
	));
}


void TRN::Engine::Worker::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_LOOP_COPY> &message)
{
	TRACE_LOGGER;
	// INFORMATION_LOGGER <<   __FUNCTION__ ;
	if (handle->simulators.find(message.simulation_id) == handle->simulators.end())
		throw std::invalid_argument("Simulator #" + std::to_string(message.simulation_id) + " does not exist");
	handle->simulators[message.simulation_id]->set_loop(TRN::Model::Loop::Copy::create(TRN::Helper::Bridge<TRN::Backend::Driver>::implementor, message.batch_size, message.stimulus_size));
}
void TRN::Engine::Worker::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_LOOP_SPATIAL_FILTER> &message)
{
	TRACE_LOGGER;
	TRN::Helper::Bridge<TRN::Backend::Driver>::implementor->toggle();
	// INFORMATION_LOGGER <<   __FUNCTION__ ;
	if (handle->simulators.find(message.simulation_id) == handle->simulators.end())
		throw std::invalid_argument("Simulator #" + std::to_string(message.simulation_id) + " does not exist");

	if (TRN::Engine::Node::handle->estimated_position.find(message.simulation_id) != TRN::Engine::Node::handle->estimated_position.end())
	{
		throw std::runtime_error("Estimated position functor is already setup for simulator #" + std::to_string(message.simulation_id));
	}
	if (TRN::Engine::Node::handle->perceived_stimulus.find(message.simulation_id) != TRN::Engine::Node::handle->perceived_stimulus.end())
	{
		throw std::runtime_error("Perceived stimulus functor is already setup for simulator #" + std::to_string(message.simulation_id));
	}

	auto decoder = handle->simulators[message.simulation_id]->get_decoder();
	if (!decoder)
		throw std::runtime_error("Decoder not configured");
	handle->simulators[message.simulation_id]->set_loop(TRN::Model::Loop::SpatialFilter::create(TRN::Helper::Bridge<TRN::Backend::Driver>::implementor, message.batch_size, message.stimulus_size, 
		[this, message]
	(const unsigned long long &evaluation_id, const std::vector<float> &values, const std::size_t &rows, const std::size_t &cols)
	{
		TRN::Engine::Message<POSITION> position;
		position.simulation_id = message.simulation_id;
		position.evaluation_id = evaluation_id;
		position.elements = values;
		position.rows = rows;
		position.cols = cols;

		 auto locked = TRN::Engine::Node::implementor.lock();
		if (locked)
			locked->send(position, 0);
	},
		TRN::Engine::Node::handle->estimated_position[message.simulation_id],
		[this, message]
	(const unsigned long long &evaluation_id, const std::vector<float> &values, const std::size_t &rows, const std::size_t &cols)
	{
		TRN::Engine::Message<STIMULUS> stimulus;
		stimulus.simulation_id = message.simulation_id;
		stimulus.evaluation_id = evaluation_id;
		stimulus.elements = values;
		stimulus.rows = rows;
		stimulus.cols = cols;
		 auto locked = TRN::Engine::Node::implementor.lock();
		if (locked)
			locked->send(stimulus, 0);
	},
		TRN::Engine::Node::handle->perceived_stimulus[message.simulation_id],
		decoder, message.tag));

}
void TRN::Engine::Worker::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_LOOP_CUSTOM> &message)
{
	TRACE_LOGGER;
	// INFORMATION_LOGGER <<   __FUNCTION__ ;
	if (handle->simulators.find(message.simulation_id) == handle->simulators.end())
		throw std::invalid_argument("Simulator #" + std::to_string(message.simulation_id) + " does not exist");
	if (TRN::Engine::Node::handle->perceived_stimulus.find(message.simulation_id) != TRN::Engine::Node::handle->perceived_stimulus.end())
	{
		throw std::runtime_error("Perceived stimulus functor is already setup for simulator #" + std::to_string(message.simulation_id));
	}
	handle->simulators[message.simulation_id]->set_loop(TRN::Model::Loop::Custom::create(TRN::Helper::Bridge<TRN::Backend::Driver>::implementor, message.batch_size, message.stimulus_size, [=]
	(const unsigned long long &evaluation_id, const std::vector<float> &values, const std::size_t &rows, const std::size_t &cols)
	{
		TRN::Engine::Message<TRN::Engine::STIMULUS> stimulus;
		stimulus.simulation_id = message.simulation_id;
		stimulus.evaluation_id = evaluation_id;
		stimulus.elements = values;
		stimulus.rows = rows;
		stimulus.cols = cols;
		 auto locked = TRN::Engine::Node::implementor.lock();
		if (locked)
			locked->send(stimulus, 0);
	},
		TRN::Engine::Node::handle->perceived_stimulus[message.simulation_id]
		));
}
void TRN::Engine::Worker::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_SCHEDULER_TILED> &message)
{
	TRACE_LOGGER;
	// INFORMATION_LOGGER <<   __FUNCTION__ ;
	if (handle->simulators.find(message.simulation_id) == handle->simulators.end())
		throw std::invalid_argument("Simulator #" + std::to_string(message.simulation_id) + " does not exist");
	handle->simulators[message.simulation_id]->set_scheduler(TRN::Model::Scheduler::Tiled::create(message.epochs));
}
void TRN::Engine::Worker::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_SCHEDULER_SNIPPETS> &message)
{
	TRACE_LOGGER;
	// INFORMATION_LOGGER <<   __FUNCTION__ ;
	if (handle->simulators.find(message.simulation_id) == handle->simulators.end())
		throw std::invalid_argument("Simulator #" + std::to_string(message.simulation_id) + " does not exist");
	handle->simulators[message.simulation_id]->set_scheduler(TRN::Model::Scheduler::Snippets::create(message.seed, message.snippets_size, message.time_budget, message.tag));

}
void TRN::Engine::Worker::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_SCHEDULER_CUSTOM> &message)
{
	TRACE_LOGGER;
	// INFORMATION_LOGGER <<   __FUNCTION__ ;
	if (handle->simulators.find(message.simulation_id) == handle->simulators.end())
		throw std::invalid_argument("Simulator #" + std::to_string(message.simulation_id) + " does not exist");

	if (TRN::Engine::Node::handle->scheduler.find(message.simulation_id) != TRN::Engine::Node::handle->scheduler.end())
	{
		throw std::runtime_error("Scheduler functor is already setup for simulator #" + std::to_string(message.simulation_id));
	}
	handle->simulators[message.simulation_id]->set_scheduler(TRN::Model::Scheduler::Custom::create(message.seed,
		[=](const unsigned long long &evaluation_id, const unsigned long &seed, const std::vector<float> &elements, const std::size_t &rows, const std::size_t &cols, const std::vector<int> &offsets, const std::vector<int> &durations)
	{
		TRN::Engine::Message<SCHEDULER_CUSTOM> scheduling_request;
		scheduling_request.evaluation_id = evaluation_id;
		scheduling_request.seed = seed;
		scheduling_request.simulation_id = message.simulation_id;
		scheduling_request.elements = elements;
		scheduling_request.rows = rows;
		scheduling_request.cols = cols;
		scheduling_request.offsets = offsets;
		scheduling_request.durations = durations;
		auto locked = TRN::Engine::Node::implementor.lock();
		if (locked)
			locked->send(scheduling_request, 0);
	},
		TRN::Engine::Node::handle->scheduler[message.simulation_id], message.tag));
}
void TRN::Engine::Worker::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_MUTATOR_SHUFFLE> &message)
{
	TRACE_LOGGER;
	// INFORMATION_LOGGER <<   __FUNCTION__ ;
	if (handle->simulators.find(message.simulation_id) == handle->simulators.end())
		throw std::invalid_argument("Simulator #" + std::to_string(message.simulation_id) + " does not exist");
	handle->simulators[message.simulation_id]->append_mutator(TRN::Model::Mutator::Shuffle::create(message.seed));
}
void TRN::Engine::Worker::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_MUTATOR_REVERSE> &message)
{
	TRACE_LOGGER;
	// INFORMATION_LOGGER <<   __FUNCTION__ ;
	if (handle->simulators.find(message.simulation_id) == handle->simulators.end())
		throw std::invalid_argument("Simulator #" + std::to_string(message.simulation_id) + " does not exist");
	handle->simulators[message.simulation_id]->append_mutator(TRN::Model::Mutator::Reverse::create(message.seed, message.rate, message.size));
}
void TRN::Engine::Worker::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_MUTATOR_PUNCH> &message)
{
	TRACE_LOGGER;
	// INFORMATION_LOGGER <<   __FUNCTION__ ;
	if (handle->simulators.find(message.simulation_id) == handle->simulators.end())
		throw std::invalid_argument("Simulator #" + std::to_string(message.simulation_id) + " does not exist");
	handle->simulators[message.simulation_id]->append_mutator(TRN::Model::Mutator::Punch::create(message.seed, message.rate, message.size, message.repetition));

}
void TRN::Engine::Worker::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_MUTATOR_CUSTOM> &message)
{
	TRACE_LOGGER;
	// INFORMATION_LOGGER <<   __FUNCTION__ ;
	if (handle->simulators.find(message.simulation_id) == handle->simulators.end())
		throw std::invalid_argument("Simulator #" + std::to_string(message.simulation_id) + " does not exist");
	
	if (TRN::Engine::Node::handle->mutator.find(message.simulation_id) != TRN::Engine::Node::handle->mutator.end())
	{
		throw std::runtime_error("Mutator functor is already setup for simulator #" + std::to_string(message.simulation_id));
	}
	handle->simulators[message.simulation_id]->append_mutator(TRN::Model::Mutator::Custom::create(message.seed, [=](const unsigned long long &evaluation_id, const unsigned long &seed, const std::vector<int> &offsets, const std::vector<int> &durations)
	{
		TRN::Engine::Message<MUTATOR_CUSTOM> scheduling;

		scheduling.evaluation_id = evaluation_id;
		scheduling.simulation_id = message.simulation_id;
		scheduling.offsets = offsets;
		scheduling.durations = durations;
		scheduling.seed = seed;
		 auto locked = TRN::Engine::Node::implementor.lock();
		if (locked)
			locked->send(scheduling, 0);
	},
		TRN::Engine::Node::handle->mutator[message.simulation_id])
	);
}
void TRN::Engine::Worker::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_FEEDFORWARD_UNIFORM> &message)
{
	TRACE_LOGGER;
	// INFORMATION_LOGGER <<   __FUNCTION__ ;
	if (handle->simulators.find(message.simulation_id) == handle->simulators.end())
		throw std::invalid_argument("Simulator #" + std::to_string(message.simulation_id) + " does not exist");
	handle->simulators[message.simulation_id]->set_feedforward(TRN::Model::Initializer::Uniform::create(TRN::Helper::Bridge<TRN::Backend::Driver>::implementor, message.a, message.b, message.sparsity));
}
void TRN::Engine::Worker::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_FEEDFORWARD_GAUSSIAN> &message)
{
	TRACE_LOGGER;
	// INFORMATION_LOGGER <<   __FUNCTION__ ;
	if (handle->simulators.find(message.simulation_id) == handle->simulators.end())
		throw std::invalid_argument("Simulator #" + std::to_string(message.simulation_id) + " does not exist");
	handle->simulators[message.simulation_id]->set_feedforward(TRN::Model::Initializer::Gaussian::create(TRN::Helper::Bridge<TRN::Backend::Driver>::implementor, message.mu, message.sigma, message.sparsity));
}
void TRN::Engine::Worker::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_FEEDFORWARD_CUSTOM> &message)
{
	TRACE_LOGGER;
	// INFORMATION_LOGGER <<   __FUNCTION__ ;
	if (handle->simulators.find(message.simulation_id) == handle->simulators.end())
		throw std::invalid_argument("Simulator #" + std::to_string(message.simulation_id) + " does not exist");
	
	if (TRN::Engine::Node::handle->feedforward_weights.find(message.simulation_id) != TRN::Engine::Node::handle->feedforward_weights.end())
	{
		throw std::runtime_error("Feedforward functor is already setup for simulator #" + std::to_string(message.simulation_id));
	}
	handle->simulators[message.simulation_id]->set_feedforward(TRN::Model::Initializer::Custom::create(TRN::Helper::Bridge<TRN::Backend::Driver>::implementor, [=](const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)
	{
		TRN::Engine::Message<FEEDFORWARD_DIMENSIONS> feedforward_dimensions;

		feedforward_dimensions.simulation_id = message.simulation_id;
		feedforward_dimensions.matrices = matrices;
		feedforward_dimensions.rows = rows;
		feedforward_dimensions.cols = cols;
		feedforward_dimensions.seed = seed;
		 auto locked = TRN::Engine::Node::implementor.lock();
		if (locked)
			locked->send(feedforward_dimensions, 0);
	}, TRN::Engine::Node::handle->feedforward_weights[message.simulation_id]));
	handle->remaining_initializations[message.simulation_id]++;
}

void TRN::Engine::Worker::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_RECURRENT_UNIFORM> &message) 
{
	TRACE_LOGGER;
	// INFORMATION_LOGGER <<   __FUNCTION__ ;
	if (handle->simulators.find(message.simulation_id) == handle->simulators.end())
		throw std::invalid_argument("Simulator #" + std::to_string(message.simulation_id) + " does not exist");
	handle->simulators[message.simulation_id]->set_recurrent(TRN::Model::Initializer::Uniform::create(TRN::Helper::Bridge<TRN::Backend::Driver>::implementor, message.a, message.b, message.sparsity));
}
void TRN::Engine::Worker::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_RECURRENT_GAUSSIAN> &message)
{
	TRACE_LOGGER;
	// INFORMATION_LOGGER <<   __FUNCTION__ ;
	if (handle->simulators.find(message.simulation_id) == handle->simulators.end())
		throw std::invalid_argument("Simulator #" + std::to_string(message.simulation_id) + " does not exist");
	handle->simulators[message.simulation_id]->set_recurrent(TRN::Model::Initializer::Gaussian::create(TRN::Helper::Bridge<TRN::Backend::Driver>::implementor, message.mu, message.sigma, message.sparsity));
}
void TRN::Engine::Worker::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_RECURRENT_CUSTOM> &message)
{
	TRACE_LOGGER;
	// INFORMATION_LOGGER <<   __FUNCTION__ ;
	if (handle->simulators.find(message.simulation_id) == handle->simulators.end())
		throw std::invalid_argument("Simulator #" + std::to_string(message.simulation_id) + " does not exist");

	if (TRN::Engine::Node::handle->recurrent.find(message.simulation_id) != TRN::Engine::Node::handle->recurrent.end())
	{
		throw std::runtime_error("Recurrent functor is already setup for simulator #" + std::to_string(message.simulation_id));
	}
	handle->simulators[message.simulation_id]->set_recurrent(TRN::Model::Initializer::Custom::create(TRN::Helper::Bridge<TRN::Backend::Driver>::implementor, [=](const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)
	{
		TRN::Engine::Message<RECURRENT_DIMENSIONS> recurrent_dimensions;

		recurrent_dimensions.simulation_id = message.simulation_id;
		recurrent_dimensions.matrices = matrices;
		recurrent_dimensions.rows = rows;
		recurrent_dimensions.cols = cols;
		recurrent_dimensions.seed = seed;
		 auto locked = TRN::Engine::Node::implementor.lock();
		if (locked)
			locked->send(recurrent_dimensions, 0);
	}, TRN::Engine::Node::handle->recurrent[message.simulation_id]));
	handle->remaining_initializations[message.simulation_id]++;
}
void TRN::Engine::Worker::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_READOUT_UNIFORM> &message)
{
	TRACE_LOGGER;
	// INFORMATION_LOGGER <<   __FUNCTION__ ;
	if (handle->simulators.find(message.simulation_id) == handle->simulators.end())
		throw std::invalid_argument("Simulator #" + std::to_string(message.simulation_id) + " does not exist");
	handle->simulators[message.simulation_id]->set_readout(TRN::Model::Initializer::Uniform::create(TRN::Helper::Bridge<TRN::Backend::Driver>::implementor, message.a, message.b, message.sparsity));
}
void TRN::Engine::Worker::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_READOUT_GAUSSIAN> &message)
{
	TRACE_LOGGER;
	// INFORMATION_LOGGER <<   __FUNCTION__ ;
	if (handle->simulators.find(message.simulation_id) == handle->simulators.end())
		throw std::invalid_argument("Simulator #" + std::to_string(message.simulation_id) + " does not exist");
	handle->simulators[message.simulation_id]->set_readout(TRN::Model::Initializer::Gaussian::create(TRN::Helper::Bridge<TRN::Backend::Driver>::implementor, message.mu, message.sigma, message.sparsity));
}
void TRN::Engine::Worker::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_READOUT_CUSTOM> &message)
{
	TRACE_LOGGER;
	// INFORMATION_LOGGER <<   __FUNCTION__ ;
	if (handle->simulators.find(message.simulation_id) == handle->simulators.end())
		throw std::invalid_argument("Simulator #" + std::to_string(message.simulation_id) + " does not exist");

	if (TRN::Engine::Node::handle->readout.find(message.simulation_id) != TRN::Engine::Node::handle->readout.end())
	{
		throw std::runtime_error("Readout functor is already setup for simulator #" + std::to_string(message.simulation_id));
	}
	handle->simulators[message.simulation_id]->set_readout(TRN::Model::Initializer::Custom::create(TRN::Helper::Bridge<TRN::Backend::Driver>::implementor, [=](const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)
	{
		TRN::Engine::Message<READOUT_DIMENSIONS> readout_dimensions;

		readout_dimensions.simulation_id = message.simulation_id;
		readout_dimensions.matrices = matrices;
		readout_dimensions.rows = rows;
		readout_dimensions.cols = cols;
		readout_dimensions.seed = seed;
		 auto locked = TRN::Engine::Node::implementor.lock();
		if (locked)
			locked->send(readout_dimensions, 0);
	}, TRN::Engine::Node::handle->readout[message.simulation_id]));
	handle->remaining_initializations[message.simulation_id]++;
}
void TRN::Engine::Worker::process(const TRN::Engine::Message<TRN::Engine::Tag::POSITION> &message)
{
	TRACE_LOGGER;
	// INFORMATION_LOGGER <<   __FUNCTION__ ;
	if (TRN::Engine::Node::handle->estimated_position.find(message.simulation_id) == TRN::Engine::Node::handle->estimated_position.end())
		throw std::runtime_error("Estimated position functor is not setup for simulator #" + std::to_string(message.simulation_id));
	TRN::Engine::Node::handle->estimated_position[message.simulation_id](message.evaluation_id, message.elements, message.rows, message.cols);
}
void TRN::Engine::Worker::process(const TRN::Engine::Message<TRN::Engine::Tag::STIMULUS> &message)
{
	TRACE_LOGGER;
	// INFORMATION_LOGGER <<   __FUNCTION__ ;
	if (TRN::Engine::Node::handle->perceived_stimulus.find(message.simulation_id) == TRN::Engine::Node::handle->perceived_stimulus.end())
		throw std::runtime_error("Perceived stimulus functor is not setup for simulator #" + std::to_string(message.simulation_id));
	TRN::Engine::Node::handle->perceived_stimulus[message.simulation_id](message.evaluation_id, message.elements, message.rows, message.cols);
}
void TRN::Engine::Worker::process(const TRN::Engine::Message<TRN::Engine::Tag::SCHEDULING> &message)
{
	TRACE_LOGGER;
	// INFORMATION_LOGGER <<   __FUNCTION__ ;
	if (message.is_from_mutator)
	{
		if (TRN::Engine::Node::handle->mutator.find(message.simulation_id) == TRN::Engine::Node::handle->mutator.end())
			throw std::runtime_error("Mutator functor is not setup for simulator #" + std::to_string(message.simulation_id));
		TRN::Engine::Node::handle->mutator[message.simulation_id](message.evaluation_id, message.offsets, message.durations);
	}
	else
	{
		if (TRN::Engine::Node::handle->scheduler.find(message.simulation_id) == TRN::Engine::Node::handle->scheduler.end())
			throw std::runtime_error("Scheduling functor is not setup for simulator #" + std::to_string(message.simulation_id));
		TRN::Engine::Node::handle->scheduler[message.simulation_id](message.evaluation_id, message.offsets, message.durations);
	}
}
void TRN::Engine::Worker::process(const TRN::Engine::Message<TRN::Engine::Tag::FEEDFORWARD_WEIGHTS> &message)
{
	TRACE_LOGGER;
	// INFORMATION_LOGGER <<   __FUNCTION__ ;
	if (TRN::Engine::Node::handle->feedforward_weights.find(message.simulation_id) == TRN::Engine::Node::handle->feedforward_weights.end())
		throw std::runtime_error("Perceived stimulus functor is not setup for simulator #" + std::to_string(message.simulation_id));
	TRN::Engine::Node::handle->feedforward_weights[message.simulation_id](message.elements, message.matrices, message.rows, message.cols);
	handle->remaining_initializations[message.simulation_id]--;
	send_configured(message.simulation_id);
}
void TRN::Engine::Worker::process(const TRN::Engine::Message<TRN::Engine::Tag::RECURRENT_WEIGHTS> &message)
{
	TRACE_LOGGER;
	// INFORMATION_LOGGER <<   __FUNCTION__ ;
	if (TRN::Engine::Node::handle->recurrent.find(message.simulation_id) == TRN::Engine::Node::handle->recurrent.end())
		throw std::runtime_error("Recurrent stimulus functor is not setup for simulator #" + std::to_string(message.simulation_id));
	TRN::Engine::Node::handle->recurrent[message.simulation_id](message.elements, message.matrices, message.rows, message.cols);
	handle->remaining_initializations[message.simulation_id]--;
	send_configured(message.simulation_id);
}
void TRN::Engine::Worker::process(const TRN::Engine::Message<TRN::Engine::Tag::READOUT_WEIGHTS> &message) 
{
	TRACE_LOGGER;
	// INFORMATION_LOGGER <<   __FUNCTION__ ;
	if (TRN::Engine::Node::handle->readout.find(message.simulation_id) == TRN::Engine::Node::handle->readout.end())
		throw std::runtime_error("Readout stimulus functor is not setup for simulator #" + std::to_string(message.simulation_id));
	TRN::Engine::Node::handle->readout[message.simulation_id](message.elements, message.matrices, message.rows, message.cols);
	handle->remaining_initializations[message.simulation_id]--;
	send_configured(message.simulation_id);
}




std::shared_ptr<TRN::Engine::Worker> TRN::Engine::Worker::create( const std::shared_ptr<TRN::Engine::Communicator> &communicator, const int &rank, const std::shared_ptr<TRN::Backend::Driver> &driver)
{
	TRACE_LOGGER;
	// INFORMATION_LOGGER <<   __FUNCTION__ ;
	return std::make_shared<TRN::Engine::Worker>(communicator, rank, driver);
}
