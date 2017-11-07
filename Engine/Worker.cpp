#include "stdafx.h"
#include "Worker_impl.h"
#include "Simulator/Basic.h"
#include "Model/Simulator.h"
#include "Model/Reservoir.h"
#include "Model/Loop.h"
#include "Model/Scheduler.h"
#include "Model/Initializer.h"
#include "Model/Measurement.h"
#include "Model/Mutator.h"
#include "Node_impl.h"

TRN::Engine::Worker::Worker(const std::shared_ptr<TRN::Engine::Communicator> &communicator, const int &rank, const std::shared_ptr<TRN::Backend::Driver> &driver) :
	TRN::Helper::Bridge<TRN::Backend::Driver>(driver),
	TRN::Engine::Node(communicator, rank),
	handle(std::make_unique<Handle>())
{
	// std::cout << __FUNCTION__ << std::endl;
	//TRN::Engine::Node::handle->name = "WORKER";
	TRN::Engine::Message<TRN::Engine::Tag::WORKER> message;

	message.host = communicator->host();
	message.rank = rank;
	message.index = driver->index();
	message.name = driver->name();
	//std::cout << "WORKER " << message.rank << " on " << message.host << " device " << message.name << " #" << message.index << std::endl;
	communicator->send(message, 0);

}

TRN::Engine::Worker::~Worker()
{

	// std::cout << __FUNCTION__ << std::endl;
	handle.reset();
}

void TRN::Engine::Worker::send_configured(const unsigned long long &id)
{
	// std::cout << __FUNCTION__ << std::endl;
	if (handle->configured_required[id] == true && handle->remaining_initializations[id] == 0)
	{
		if (handle->simulators.find(id) == handle->simulators.end())
			throw std::invalid_argument("Simulator #" + std::to_string(id) + "does not exist");
		handle->simulators[id]->get_reservoir()->initialize();
		TRN::Engine::Message<TRN::Engine::CONFIGURED> configured;


		configured.id = id;
		
		auto locked = TRN::Engine::Node::get_implementor().lock();
		if (locked)
			locked->send(configured, 0);
		handle->configured_required[id] = false;
	}
}

void TRN::Engine::Worker::initialize()
{
	// std::cout << __FUNCTION__ << std::endl;
	TRN::Helper::Bridge<TRN::Backend::Driver>::implementor->toggle();
}
/*void TRN::Engine::Worker::process(const TRN::Engine::Message<TRN::Engine::Tag::READY> &message)
{


}*/
void TRN::Engine::Worker::process(const TRN::Engine::Message<TRN::Engine::Tag::COMPLETED> &message)
{
	// std::cout << __FUNCTION__ << std::endl;
	TRN::Engine::Message<TRN::Engine::QUIT> quit;
	quit.rank = TRN::Engine::Node::handle->rank;
	auto locked = TRN::Engine::Node::implementor.lock();
	if (locked)
		locked->send(quit, 0);
}
void TRN::Engine::Worker::process(const TRN::Engine::Message<TRN::Engine::Tag::ALLOCATE> &message)
{
	//std::cout << "allocate " << message.id << "on rank " << TRN::Engine::Node::handle->rank << std::endl;
	// std::cout << __FUNCTION__ << std::endl;
	if (handle->simulators.find(message.id) != handle->simulators.end())
		throw std::invalid_argument("Simulator #" + std::to_string(message.id) + "already exists");

	auto basic = TRN::Model::Simulator::Basic::create(
		[this, message]()
	{
		TRN::Engine::Message<TRN::Engine::TRAINED> trained;

		trained.id = message.id;

		 auto locked = TRN::Engine::Node::implementor.lock();
		if (locked)
			locked->send(trained, 0);
	},
	[this, message]()
	{
		TRN::Engine::Message<TRN::Engine::PRIMED> primed;

		primed.id = message.id;
		 auto locked = TRN::Engine::Node::implementor.lock();
		if (locked)
			locked->send(primed, 0);
	}
		,
	[this, message]()
	{
		TRN::Engine::Message<TRN::Engine::TESTED> tested;

		tested.id = message.id;
		 auto locked = TRN::Engine::Node::implementor.lock();
		if (locked)
			locked->send(tested, 0);
	}
	);
	handle->simulators[message.id] = basic;
	handle->configured_required[message.id] = false;
	handle->remaining_initializations[message.id] = 0;

	TRN::Engine::Message<TRN::Engine::ALLOCATED> allocated;

	allocated.id = message.id;
	auto locked = TRN::Engine::Node::implementor.lock();
	if (locked)
		locked->send(allocated, 0);
}
void TRN::Engine::Worker::process(const TRN::Engine::Message<TRN::Engine::Tag::DEALLOCATE> &message)
{
	// 
	//std::cout << "deallocate " << message.id << " on rank " << TRN::Engine::Node::handle->rank << std::endl;
	if (handle->simulators.find(message.id) == handle->simulators.end())
	{
		throw std::invalid_argument("Simulator #" + std::to_string(message.id) + " does not exist");
	}
	handle->simulators[message.id]->uninitialize();
	handle->simulators.erase(message.id);
	handle->remaining_initializations.erase(message.id);
	handle->configured_required.erase(message.id);
//	std::cout << "deallocated " << message.id << " on rank " << TRN::Engine::Node::handle->rank << std::endl;
	TRN::Engine::Message<TRN::Engine::DEALLOCATED> deallocated;

	deallocated.id = message.id;
	auto locked = TRN::Engine::Node::implementor.lock();
	if (locked)
		locked->send(deallocated, 0);
}
void TRN::Engine::Worker::process(const TRN::Engine::Message<TRN::Engine::Tag::TRAIN> &message)
{
	// std::cout << __FUNCTION__ << std::endl;
	if (handle->simulators.find(message.id) == handle->simulators.end())
		throw std::invalid_argument("Simulator #" + std::to_string(message.id) + " does not exist");
	handle->simulators[message.id]->train(message.label, message.incoming, message.expected);
}
void TRN::Engine::Worker::process(const TRN::Engine::Message<TRN::Engine::Tag::TEST> &message)
{
	// std::cout << __FUNCTION__ << std::endl;
	if (handle->simulators.find(message.id) == handle->simulators.end())
		throw std::invalid_argument("Simulator #" + std::to_string(message.id) + " does not exist");
	handle->simulators[message.id]->test(message.label, message.incoming, message.expected, message.preamble, message.autonomous, message.supplementary_generations);

}
void TRN::Engine::Worker::process(const TRN::Engine::Message<TRN::Engine::Tag::DECLARE_SEQUENCE> &message)
{
	// std::cout << __FUNCTION__ << std::endl;
	if (handle->simulators.find(message.id) == handle->simulators.end())
		throw std::invalid_argument("Simulator #" + std::to_string(message.id) + " does not exist");
	handle->simulators[message.id]->declare(message.label, message.tag, TRN::Core::Matrix::create(TRN::Helper::Bridge<TRN::Backend::Driver>::implementor, message.sequence, message.observations, message.sequence.size() / message.observations));
}
void TRN::Engine::Worker::process(const TRN::Engine::Message<TRN::Engine::Tag::DECLARE_SET> &message)
{
	// std::cout << __FUNCTION__ << std::endl;
	if (handle->simulators.find(message.id) == handle->simulators.end())
		throw std::invalid_argument("Simulator #" + std::to_string(message.id) + " does not exist");
	std::vector<std::shared_ptr<TRN::Core::Matrix>> sequences;
	for (auto sequence_label : message.labels)
	{
		sequences.push_back(handle->simulators[message.id]->retrieve_sequence(sequence_label, message.tag));
	}

	handle->simulators[message.id]->declare(message.label, message.tag, TRN::Core::Set::create(TRN::Helper::Bridge<TRN::Backend::Driver>::implementor, sequences));
}
void TRN::Engine::Worker::process(const TRN::Engine::Message<TRN::Engine::Tag::SETUP_STATES> &message) 
{
	// std::cout << __FUNCTION__ << std::endl;
	if (handle->simulators.find(message.id) == handle->simulators.end())
		throw std::invalid_argument("Simulator #" + std::to_string(message.id) + " does not exist");
	if (!handle->simulators[message.id]->get_reservoir())
		throw std::logic_error("Simulator #" + std::to_string(message.id) + " does not not have a reservoir to decorate");
	auto decorator = TRN::Model::Simulator::States::create(handle->simulators[message.id], [this, message]
	(const std::string &phase, const std::string &label, const std::size_t &batch, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &samples, const std::size_t &rows, const std::size_t &cols)
	{
		TRN::Engine::Message<TRN::Engine::STATES> states;

		states.id = message.id;
		states.label = label;
		states.phase = phase;
		states.batch = batch;
		states.trial = trial;
		states.evaluation = evaluation;
		states.elements = samples;
		states.rows = rows;
		states.cols = cols;

		 auto locked = TRN::Engine::Node::implementor.lock();
		if (locked)
			locked->send(states, 0);
	}, message.train, message.prime, message.generate);
	handle->simulators[message.id]->attach(decorator);
	handle->simulators[message.id] = decorator;
}
void TRN::Engine::Worker::process(const TRN::Engine::Message<TRN::Engine::Tag::SETUP_WEIGHTS> &message)
{
	// std::cout << __FUNCTION__ << std::endl;
	if (handle->simulators.find(message.id) == handle->simulators.end())
		throw std::invalid_argument("Simulator #" + std::to_string(message.id) + " does not exist");
	if (!handle->simulators[message.id]->get_reservoir())
		throw std::logic_error("Simulator #" + std::to_string(message.id) + " does not not have a reservoir to decorate");
	auto decorator = TRN::Model::Simulator::Weights::create(handle->simulators[message.id], [this, message]
	(const std::string &phase, const std::string &label, const std::size_t &batch, const std::size_t &trial, const std::vector<float> &samples, const std::size_t &rows, const std::size_t &cols)
	{
		TRN::Engine::Message<TRN::Engine::WEIGHTS> weights;

		weights.id = message.id;
		weights.phase = phase;
		weights.label = label;
		weights.trial = trial;
		weights.batch = batch;
		weights.elements = samples;
		weights.rows = rows;
		weights.cols = cols;

		 auto locked = TRN::Engine::Node::implementor.lock();
		if (locked)
			locked->send(weights, 0);
	}, message.initialization, message.train);
	handle->simulators[message.id]->attach(decorator);
	handle->simulators[message.id] = decorator;
}
void TRN::Engine::Worker::process(const TRN::Engine::Message<TRN::Engine::Tag::SETUP_PERFORMANCES> &message)
{
	// std::cout << __FUNCTION__ << std::endl;
	if (handle->simulators.find(message.id) == handle->simulators.end())
		throw std::invalid_argument("Simulator #" + std::to_string(message.id) + " does not exist");
	if (!handle->simulators[message.id]->get_reservoir())
		throw std::logic_error("Simulator #" + std::to_string(message.id) + " does not not have a reservoir to decorate");
	auto decorator = TRN::Model::Simulator::Performances::create(handle->simulators[message.id], [this, message]
	(const std::size_t &trial, const std::size_t &evaluation, const std::string &phase, const float &cycles_per_second, const float &gflops_per_second)
	{
		TRN::Engine::Message<TRN::Engine::PERFORMANCES> performances;

		performances.id = message.id;
		performances.phase = phase;

		performances.trial = trial;
		performances.evaluation = evaluation;
		performances.phase = phase;
		performances.cycles_per_second = cycles_per_second;
		performances.gflops_per_second = gflops_per_second;
		 auto locked = TRN::Engine::Node::implementor.lock();
		if (locked)
			locked->send(performances, 0);
	}, message.train, message.prime, message.generate);
	handle->simulators[message.id]->attach(decorator);
	handle->simulators[message.id] = decorator;
}
void TRN::Engine::Worker::process(const TRN::Engine::Message<TRN::Engine::Tag::SETUP_SCHEDULING> &message)
{
	// std::cout << __FUNCTION__ << std::endl;
	if (handle->simulators.find(message.id) == handle->simulators.end())
		throw std::invalid_argument("Simulator #" + std::to_string(message.id) + " does not exist");
	if (!handle->simulators[message.id]->get_reservoir())
		throw std::logic_error("Simulator #" + std::to_string(message.id) + " does not not have a reservoir to decorate");
	auto decorator = TRN::Model::Simulator::Scheduling::create(handle->simulators[message.id], [this, message]
	(const std::size_t &trial, const std::vector<int> &offsets, const std::vector<int> &durations)
	{
		TRN::Engine::Message<TRN::Engine::SCHEDULING> scheduling;

		scheduling.id = message.id;
		scheduling.trial = trial;
		scheduling.offsets = offsets;
		scheduling.durations = durations;
		scheduling.is_from_mutator = false;
		 auto locked = TRN::Engine::Node::implementor.lock();
		if (locked)
			locked->send(scheduling, 0);
	});
	//handle->simulators[message.id]->attach(decorator);
	handle->simulators[message.id] = decorator;
}
void TRN::Engine::Worker::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_BEGIN> &message)
{
	// std::cout << __FUNCTION__ << std::endl;
	if (handle->simulators.find(message.id) == handle->simulators.end())
		throw std::invalid_argument("Simulator #" + std::to_string(message.id) + " does not exist");
	handle->simulators[message.id]->uninitialize();
}
void TRN::Engine::Worker::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_END> &message)
{
	// std::cout << __FUNCTION__ << std::endl;
	if (handle->simulators.find(message.id) == handle->simulators.end())
		throw std::invalid_argument("Simulator #" + std::to_string(message.id) + " does not exist");
	handle->simulators[message.id]->initialize();

	handle->configured_required[message.id] = true;
	send_configured(message.id);
}
void TRN::Engine::Worker::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_MEASUREMENT_READOUT_MEAN_SQUARE_ERROR> &message)
{
	// std::cout << __FUNCTION__ << std::endl;
	if (handle->simulators.find(message.id) == handle->simulators.end())
		throw std::invalid_argument("Simulator #" + std::to_string(message.id) + " does not exist");
	handle->simulators[message.id]->append_measurement(
		TRN::Model::Measurement::Sequence::create(
			TRN::Model::Measurement::MeanSquareError::create(TRN::Helper::Bridge<TRN::Backend::Driver>::implementor,
				[this, message](const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols)
	{
		TRN::Engine::Message<TRN::Engine::MEASUREMENT_READOUT_MEAN_SQUARE_ERROR> measurement;

		measurement.id = message.id;
		measurement.trial = trial;
		measurement.evaluation = evaluation;
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
	// std::cout << __FUNCTION__ << std::endl;
	if (handle->simulators.find(message.id) == handle->simulators.end())
		throw std::invalid_argument("Simulator #" + std::to_string(message.id) + " does not exist");
	handle->simulators[message.id]->append_measurement(
		TRN::Model::Measurement::Sequence::create(
			TRN::Model::Measurement::FrechetDistance::create(TRN::Helper::Bridge<TRN::Backend::Driver>::implementor,
				[this, message](const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols)
	{
		TRN::Engine::Message<TRN::Engine::MEASUREMENT_READOUT_FRECHET_DISTANCE> measurement;

		measurement.id = message.id;
		measurement.trial = trial;
		measurement.evaluation = evaluation;
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
	// std::cout << __FUNCTION__ << std::endl;
	if (handle->simulators.find(message.id) == handle->simulators.end())
		throw std::invalid_argument("Simulator #" + std::to_string(message.id) + " does not exist");
	handle->simulators[message.id]->append_measurement(
		TRN::Model::Measurement::Sequence::create(
			TRN::Model::Measurement::Custom::create(TRN::Helper::Bridge<TRN::Backend::Driver>::implementor,
				[this, message](const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &primed, const std::vector<float> &predicted, const std::vector<float> &expected, const std::size_t &preamble, const std::size_t &pages, const std::size_t &rows, const  std::size_t &cols)
	{
		TRN::Engine::Message<TRN::Engine::MEASUREMENT_READOUT_CUSTOM> measurement;

		measurement.id = message.id;
		measurement.trial = trial;
		measurement.evaluation = evaluation;
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
	// std::cout << __FUNCTION__ << std::endl;
	if (handle->simulators.find(message.id) == handle->simulators.end())
		throw std::invalid_argument("Simulator #" + std::to_string(message.id) + " does not exist");
	handle->simulators[message.id]->append_measurement(
		TRN::Model::Measurement::Position::create(
			TRN::Model::Measurement::MeanSquareError::create(TRN::Helper::Bridge<TRN::Backend::Driver>::implementor,
				[this, message](const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols)
	{
		TRN::Engine::Message<TRN::Engine::MEASUREMENT_POSITION_MEAN_SQUARE_ERROR> measurement;

		measurement.id = message.id;
		measurement.trial = trial;
		measurement.evaluation = evaluation;
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
	// std::cout << __FUNCTION__ << std::endl;
	if (handle->simulators.find(message.id) == handle->simulators.end())
		throw std::invalid_argument("Simulator #" + std::to_string(message.id) + " does not exist");
	handle->simulators[message.id]->append_measurement(
		TRN::Model::Measurement::Position::create(
			TRN::Model::Measurement::MeanSquareError::create(TRN::Helper::Bridge<TRN::Backend::Driver>::implementor,
				[this, message](const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols)
	{
		TRN::Engine::Message<TRN::Engine::MEASUREMENT_POSITION_FRECHET_DISTANCE> measurement;

		measurement.id = message.id;
		measurement.trial = trial;
		measurement.evaluation = evaluation;
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
	// std::cout << __FUNCTION__ << std::endl;
	if (handle->simulators.find(message.id) == handle->simulators.end())
		throw std::invalid_argument("Simulator #" + std::to_string(message.id) + " does not exist");
	handle->simulators[message.id]->append_measurement(
		TRN::Model::Measurement::Position::create(
			TRN::Model::Measurement::Custom::create(TRN::Helper::Bridge<TRN::Backend::Driver>::implementor,
				[this, message](const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &primed, const std::vector<float> &predicted, const std::vector<float> &expected, const std::size_t &preamble, const std::size_t &pages, const std::size_t &rows, const  std::size_t &cols)
	{
		TRN::Engine::Message<TRN::Engine::MEASUREMENT_POSITION_CUSTOM> measurement;

		measurement.id = message.id;
		measurement.trial = trial;
		measurement.evaluation = evaluation;
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
	// std::cout << __FUNCTION__ << std::endl;
	if (handle->simulators.find(message.id) == handle->simulators.end())
		throw std::invalid_argument("Simulator #" + std::to_string(message.id) + " does not exist");
	handle->simulators[message.id]->set_reservoir(TRN::Model::Reservoir::WidrowHoff::create(TRN::Helper::Bridge<TRN::Backend::Driver>::implementor, message.stimulus_size, message.prediction_size, message.reservoir_size, message.leak_rate, message.initial_state_scale, message.learning_rate, message.seed, message.batch_size));
}
void TRN::Engine::Worker::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_LOOP_COPY> &message)
{
	// std::cout << __FUNCTION__ << std::endl;
	if (handle->simulators.find(message.id) == handle->simulators.end())
		throw std::invalid_argument("Simulator #" + std::to_string(message.id) + " does not exist");
	handle->simulators[message.id]->set_loop(TRN::Model::Loop::Copy::create(TRN::Helper::Bridge<TRN::Backend::Driver>::implementor, message.batch_size, message.stimulus_size));
}
void TRN::Engine::Worker::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_LOOP_SPATIAL_FILTER> &message)
{
	// std::cout << __FUNCTION__ << std::endl;
	if (handle->simulators.find(message.id) == handle->simulators.end())
		throw std::invalid_argument("Simulator #" + std::to_string(message.id) + " does not exist");

	if (TRN::Engine::Node::handle->estimated_position.find(message.id) != TRN::Engine::Node::handle->estimated_position.end())
	{
		throw std::runtime_error("Estimated position functor is already setup for simulator #" + std::to_string(message.id));
	}
	if (TRN::Engine::Node::handle->perceived_stimulus.find(message.id) != TRN::Engine::Node::handle->perceived_stimulus.end())
	{
		throw std::runtime_error("Perceived stimulus functor is already setup for simulator #" + std::to_string(message.id));
	}
	handle->simulators[message.id]->set_loop(TRN::Model::Loop::SpatialFilter::create(TRN::Helper::Bridge<TRN::Backend::Driver>::implementor, message.batch_size, message.stimulus_size, message.seed,
		[this, message]
	(const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const std::size_t &cols)
	{
		TRN::Engine::Message<POSITION> position;
		position.id = message.id;
		position.trial = trial;
		position.evaluation = evaluation;
		position.elements = values;
		position.rows = rows;
		position.cols = cols;
		 auto locked = TRN::Engine::Node::implementor.lock();
		if (locked)
			locked->send(position, 0);
	},
		TRN::Engine::Node::handle->estimated_position[message.id],
		[this, message]
	(const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const std::size_t &cols)
	{
		TRN::Engine::Message<STIMULUS> stimulus;
		stimulus.id = message.id;
		stimulus.trial = trial;
		stimulus.evaluation = evaluation;
		stimulus.elements = values;
		stimulus.rows = rows;
		stimulus.cols = cols;
		 auto locked = TRN::Engine::Node::implementor.lock();
		if (locked)
			locked->send(stimulus, 0);
	},
		TRN::Engine::Node::handle->perceived_stimulus[message.id],
		message.rows, message.cols, message.x, message.y, message.response, message.sigma, message.radius, message.scale, message.tag));

}
void TRN::Engine::Worker::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_LOOP_CUSTOM> &message)
{
	// std::cout << __FUNCTION__ << std::endl;
	if (handle->simulators.find(message.id) == handle->simulators.end())
		throw std::invalid_argument("Simulator #" + std::to_string(message.id) + " does not exist");
	if (TRN::Engine::Node::handle->perceived_stimulus.find(message.id) != TRN::Engine::Node::handle->perceived_stimulus.end())
	{
		throw std::runtime_error("Perceived stimulus functor is already setup for simulator #" + std::to_string(message.id));
	}
	handle->simulators[message.id]->set_loop(TRN::Model::Loop::Custom::create(TRN::Helper::Bridge<TRN::Backend::Driver>::implementor, message.batch_size, message.stimulus_size, [=]
	(const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const std::size_t &cols)
	{
		TRN::Engine::Message<TRN::Engine::STIMULUS> stimulus;
		stimulus.id = message.id;
		stimulus.trial = trial;
		stimulus.evaluation = evaluation;
		stimulus.elements = values;
		stimulus.rows = rows;
		stimulus.cols = cols;
		 auto locked = TRN::Engine::Node::implementor.lock();
		if (locked)
			locked->send(stimulus, 0);
	},
		TRN::Engine::Node::handle->perceived_stimulus[message.id]
		));
}
void TRN::Engine::Worker::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_SCHEDULER_TILED> &message)
{
	// std::cout << __FUNCTION__ << std::endl;
	if (handle->simulators.find(message.id) == handle->simulators.end())
		throw std::invalid_argument("Simulator #" + std::to_string(message.id) + " does not exist");
	handle->simulators[message.id]->set_scheduler(TRN::Model::Scheduler::Tiled::create(message.epochs));
}
void TRN::Engine::Worker::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_SCHEDULER_SNIPPETS> &message)
{
	// std::cout << __FUNCTION__ << std::endl;
	if (handle->simulators.find(message.id) == handle->simulators.end())
		throw std::invalid_argument("Simulator #" + std::to_string(message.id) + " does not exist");
	handle->simulators[message.id]->set_scheduler(TRN::Model::Scheduler::Snippets::create(message.seed, message.snippets_size, message.time_budget, message.tag));

}
void TRN::Engine::Worker::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_SCHEDULER_CUSTOM> &message)
{
	// std::cout << __FUNCTION__ << std::endl;
	if (handle->simulators.find(message.id) == handle->simulators.end())
		throw std::invalid_argument("Simulator #" + std::to_string(message.id) + " does not exist");

	if (TRN::Engine::Node::handle->scheduler.find(message.id) != TRN::Engine::Node::handle->scheduler.end())
	{
		throw std::runtime_error("Scheduler functor is already setup for simulator #" + std::to_string(message.id));
	}
	handle->simulators[message.id]->set_scheduler(TRN::Model::Scheduler::Custom::create(message.seed,
		[=](const unsigned long &seed, const std::size_t &trial, const std::vector<float> &elements, const std::size_t &rows, const std::size_t &cols, const std::vector<int> &offsets, const std::vector<int> &durations)
	{
		TRN::Engine::Message<SCHEDULER_CUSTOM> scheduling_request;
		scheduling_request.trial = trial;
		scheduling_request.seed = seed;
		scheduling_request.id = message.id;
		scheduling_request.elements = elements;
		scheduling_request.rows = rows;
		scheduling_request.cols = cols;
		scheduling_request.offsets = offsets;
		scheduling_request.durations = durations;
		 auto locked = TRN::Engine::Node::implementor.lock();
		if (locked)
			locked->send(scheduling_request, 0);
	},
		TRN::Engine::Node::handle->scheduler[message.id], message.tag));
}
void TRN::Engine::Worker::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_MUTATOR_SHUFFLE> &message)
{
	// std::cout << __FUNCTION__ << std::endl;
	if (handle->simulators.find(message.id) == handle->simulators.end())
		throw std::invalid_argument("Simulator #" + std::to_string(message.id) + " does not exist");
	handle->simulators[message.id]->append_mutator(TRN::Model::Mutator::Shuffle::create(message.seed));
}
void TRN::Engine::Worker::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_MUTATOR_REVERSE> &message)
{
	// std::cout << __FUNCTION__ << std::endl;
	if (handle->simulators.find(message.id) == handle->simulators.end())
		throw std::invalid_argument("Simulator #" + std::to_string(message.id) + " does not exist");
	handle->simulators[message.id]->append_mutator(TRN::Model::Mutator::Reverse::create(message.seed, message.rate, message.size));
}
void TRN::Engine::Worker::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_MUTATOR_PUNCH> &message)
{
	// std::cout << __FUNCTION__ << std::endl;
	if (handle->simulators.find(message.id) == handle->simulators.end())
		throw std::invalid_argument("Simulator #" + std::to_string(message.id) + " does not exist");
	handle->simulators[message.id]->append_mutator(TRN::Model::Mutator::Punch::create(message.seed, message.rate, message.size, message.repetition));

}
void TRN::Engine::Worker::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_MUTATOR_CUSTOM> &message)
{
	// std::cout << __FUNCTION__ << std::endl;
	if (handle->simulators.find(message.id) == handle->simulators.end())
		throw std::invalid_argument("Simulator #" + std::to_string(message.id) + " does not exist");
	
	if (TRN::Engine::Node::handle->mutator.find(message.id) != TRN::Engine::Node::handle->mutator.end())
	{
		throw std::runtime_error("Mutator functor is already setup for simulator #" + std::to_string(message.id));
	}
	handle->simulators[message.id]->append_mutator(TRN::Model::Mutator::Custom::create(message.seed, [=](const unsigned long &seed, const std::size_t &trial, const std::vector<int> &offsets, const std::vector<int> &durations)
	{
		TRN::Engine::Message<MUTATOR_CUSTOM> scheduling;

		scheduling.trial = trial;
		scheduling.id = message.id;
		scheduling.offsets = offsets;
		scheduling.durations = durations;
		scheduling.seed = seed;
		 auto locked = TRN::Engine::Node::implementor.lock();
		if (locked)
			locked->send(scheduling, 0);
	},
		TRN::Engine::Node::handle->mutator[message.id])
	);
}
void TRN::Engine::Worker::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_FEEDFORWARD_UNIFORM> &message)
{
	// std::cout << __FUNCTION__ << std::endl;
	if (handle->simulators.find(message.id) == handle->simulators.end())
		throw std::invalid_argument("Simulator #" + std::to_string(message.id) + " does not exist");
	handle->simulators[message.id]->set_feedforward(TRN::Model::Initializer::Uniform::create(TRN::Helper::Bridge<TRN::Backend::Driver>::implementor, message.a, message.b, message.sparsity));
}
void TRN::Engine::Worker::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_FEEDFORWARD_GAUSSIAN> &message)
{
	// std::cout << __FUNCTION__ << std::endl;
	if (handle->simulators.find(message.id) == handle->simulators.end())
		throw std::invalid_argument("Simulator #" + std::to_string(message.id) + " does not exist");
	handle->simulators[message.id]->set_feedforward(TRN::Model::Initializer::Gaussian::create(TRN::Helper::Bridge<TRN::Backend::Driver>::implementor, message.mu, message.sigma));
}
void TRN::Engine::Worker::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_FEEDFORWARD_CUSTOM> &message)
{
	// std::cout << __FUNCTION__ << std::endl;
	if (handle->simulators.find(message.id) == handle->simulators.end())
		throw std::invalid_argument("Simulator #" + std::to_string(message.id) + " does not exist");
	
	if (TRN::Engine::Node::handle->feedforward_weights.find(message.id) != TRN::Engine::Node::handle->feedforward_weights.end())
	{
		throw std::runtime_error("Feedforward functor is already setup for simulator #" + std::to_string(message.id));
	}
	handle->simulators[message.id]->set_feedforward(TRN::Model::Initializer::Custom::create(TRN::Helper::Bridge<TRN::Backend::Driver>::implementor, [=](const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)
	{
		TRN::Engine::Message<FEEDFORWARD_DIMENSIONS> feedforward_dimensions;

		feedforward_dimensions.id = message.id;
		feedforward_dimensions.matrices = matrices;
		feedforward_dimensions.rows = rows;
		feedforward_dimensions.cols = cols;
		feedforward_dimensions.seed = seed;
		 auto locked = TRN::Engine::Node::implementor.lock();
		if (locked)
			locked->send(feedforward_dimensions, 0);
	}, TRN::Engine::Node::handle->feedforward_weights[message.id]));
	handle->remaining_initializations[message.id]++;
}
void TRN::Engine::Worker::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_FEEDBACK_UNIFORM> &message)
{
	// std::cout << __FUNCTION__ << std::endl;
	if (handle->simulators.find(message.id) == handle->simulators.end())
		throw std::invalid_argument("Simulator #" + std::to_string(message.id) + " does not exist");
	handle->simulators[message.id]->set_feedback(TRN::Model::Initializer::Uniform::create(TRN::Helper::Bridge<TRN::Backend::Driver>::implementor, message.a, message.b, message.sparsity));

}
void TRN::Engine::Worker::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_FEEDBACK_GAUSSIAN> &message) 
{
	// std::cout << __FUNCTION__ << std::endl;
	if (handle->simulators.find(message.id) == handle->simulators.end())
		throw std::invalid_argument("Simulator #" + std::to_string(message.id) + " does not exist");
	handle->simulators[message.id]->set_feedback(TRN::Model::Initializer::Gaussian::create(TRN::Helper::Bridge<TRN::Backend::Driver>::implementor, message.mu, message.sigma));
}
void TRN::Engine::Worker::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_FEEDBACK_CUSTOM> &message)
{
	// std::cout << __FUNCTION__ << std::endl;
	if (handle->simulators.find(message.id) == handle->simulators.end())
		throw std::invalid_argument("Simulator #" + std::to_string(message.id) + " does not exist");

	if (TRN::Engine::Node::handle->feedback_weights.find(message.id) != TRN::Engine::Node::handle->feedback_weights.end())
	{
		throw std::runtime_error("Feedback functor is already setup for simulator #" + std::to_string(message.id));
	}
	handle->simulators[message.id]->set_feedback(TRN::Model::Initializer::Custom::create(TRN::Helper::Bridge<TRN::Backend::Driver>::implementor, [=](const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)
	{
		TRN::Engine::Message<FEEDBACK_DIMENSIONS> feedback_dimensions;

		feedback_dimensions.id = message.id;
		feedback_dimensions.matrices = matrices;
		feedback_dimensions.rows = rows;
		feedback_dimensions.cols = cols;
		feedback_dimensions.seed = seed;
		 auto locked = TRN::Engine::Node::implementor.lock();
		if (locked)
			locked->send(feedback_dimensions, 0);
	}, TRN::Engine::Node::handle->feedback_weights[message.id]));
	handle->remaining_initializations[message.id]++;
}
void TRN::Engine::Worker::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_RECURRENT_UNIFORM> &message) 
{
	// std::cout << __FUNCTION__ << std::endl;
	if (handle->simulators.find(message.id) == handle->simulators.end())
		throw std::invalid_argument("Simulator #" + std::to_string(message.id) + " does not exist");
	handle->simulators[message.id]->set_recurrent(TRN::Model::Initializer::Uniform::create(TRN::Helper::Bridge<TRN::Backend::Driver>::implementor, message.a, message.b, message.sparsity));
}
void TRN::Engine::Worker::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_RECURRENT_GAUSSIAN> &message)
{
	// std::cout << __FUNCTION__ << std::endl;
	if (handle->simulators.find(message.id) == handle->simulators.end())
		throw std::invalid_argument("Simulator #" + std::to_string(message.id) + " does not exist");
	handle->simulators[message.id]->set_recurrent(TRN::Model::Initializer::Gaussian::create(TRN::Helper::Bridge<TRN::Backend::Driver>::implementor, message.mu, message.sigma));
}
void TRN::Engine::Worker::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_RECURRENT_CUSTOM> &message)
{
	// std::cout << __FUNCTION__ << std::endl;
	if (handle->simulators.find(message.id) == handle->simulators.end())
		throw std::invalid_argument("Simulator #" + std::to_string(message.id) + " does not exist");

	if (TRN::Engine::Node::handle->recurrent.find(message.id) != TRN::Engine::Node::handle->recurrent.end())
	{
		throw std::runtime_error("Recurrent functor is already setup for simulator #" + std::to_string(message.id));
	}
	handle->simulators[message.id]->set_recurrent(TRN::Model::Initializer::Custom::create(TRN::Helper::Bridge<TRN::Backend::Driver>::implementor, [=](const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)
	{
		TRN::Engine::Message<RECURRENT_DIMENSIONS> recurrent_dimensions;

		recurrent_dimensions.id = message.id;
		recurrent_dimensions.matrices = matrices;
		recurrent_dimensions.rows = rows;
		recurrent_dimensions.cols = cols;
		recurrent_dimensions.seed = seed;
		 auto locked = TRN::Engine::Node::implementor.lock();
		if (locked)
			locked->send(recurrent_dimensions, 0);
	}, TRN::Engine::Node::handle->recurrent[message.id]));
	handle->remaining_initializations[message.id]++;
}
void TRN::Engine::Worker::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_READOUT_UNIFORM> &message)
{
	// std::cout << __FUNCTION__ << std::endl;
	if (handle->simulators.find(message.id) == handle->simulators.end())
		throw std::invalid_argument("Simulator #" + std::to_string(message.id) + " does not exist");
	handle->simulators[message.id]->set_readout(TRN::Model::Initializer::Uniform::create(TRN::Helper::Bridge<TRN::Backend::Driver>::implementor, message.a, message.b, message.sparsity));
}
void TRN::Engine::Worker::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_READOUT_GAUSSIAN> &message)
{
	// std::cout << __FUNCTION__ << std::endl;
	if (handle->simulators.find(message.id) == handle->simulators.end())
		throw std::invalid_argument("Simulator #" + std::to_string(message.id) + " does not exist");
	handle->simulators[message.id]->set_readout(TRN::Model::Initializer::Gaussian::create(TRN::Helper::Bridge<TRN::Backend::Driver>::implementor, message.mu, message.sigma));
}
void TRN::Engine::Worker::process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_READOUT_CUSTOM> &message)
{
	// std::cout << __FUNCTION__ << std::endl;
	if (handle->simulators.find(message.id) == handle->simulators.end())
		throw std::invalid_argument("Simulator #" + std::to_string(message.id) + " does not exist");

	if (TRN::Engine::Node::handle->readout.find(message.id) != TRN::Engine::Node::handle->readout.end())
	{
		throw std::runtime_error("Readout functor is already setup for simulator #" + std::to_string(message.id));
	}
	handle->simulators[message.id]->set_readout(TRN::Model::Initializer::Custom::create(TRN::Helper::Bridge<TRN::Backend::Driver>::implementor, [=](const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)
	{
		TRN::Engine::Message<READOUT_DIMENSIONS> readout_dimensions;

		readout_dimensions.id = message.id;
		readout_dimensions.matrices = matrices;
		readout_dimensions.rows = rows;
		readout_dimensions.cols = cols;
		readout_dimensions.seed = seed;
		 auto locked = TRN::Engine::Node::implementor.lock();
		if (locked)
			locked->send(readout_dimensions, 0);
	}, TRN::Engine::Node::handle->readout[message.id]));
	handle->remaining_initializations[message.id]++;
}
void TRN::Engine::Worker::process(const TRN::Engine::Message<TRN::Engine::Tag::POSITION> &message)
{
	// std::cout << __FUNCTION__ << std::endl;
	if (TRN::Engine::Node::handle->estimated_position.find(message.id) == TRN::Engine::Node::handle->estimated_position.end())
		throw std::runtime_error("Estimated position functor is not setup for simulator #" + std::to_string(message.id));
	TRN::Engine::Node::handle->estimated_position[message.id](message.trial, message.evaluation, message.elements, message.rows, message.cols);
}
void TRN::Engine::Worker::process(const TRN::Engine::Message<TRN::Engine::Tag::STIMULUS> &message)
{
	// std::cout << __FUNCTION__ << std::endl;
	if (TRN::Engine::Node::handle->perceived_stimulus.find(message.id) == TRN::Engine::Node::handle->perceived_stimulus.end())
		throw std::runtime_error("Perceived stimulus functor is not setup for simulator #" + std::to_string(message.id));
	TRN::Engine::Node::handle->perceived_stimulus[message.id](message.trial, message.evaluation, message.elements, message.rows, message.cols);
}
void TRN::Engine::Worker::process(const TRN::Engine::Message<TRN::Engine::Tag::SCHEDULING> &message)
{
	// std::cout << __FUNCTION__ << std::endl;
	if (message.is_from_mutator)
	{
		if (TRN::Engine::Node::handle->mutator.find(message.id) == TRN::Engine::Node::handle->mutator.end())
			throw std::runtime_error("Mutator functor is not setup for simulator #" + std::to_string(message.id));
		TRN::Engine::Node::handle->mutator[message.id](message.trial, message.offsets, message.durations);
	}
	else
	{
		if (TRN::Engine::Node::handle->scheduler.find(message.id) == TRN::Engine::Node::handle->scheduler.end())
			throw std::runtime_error("Scheduling functor is not setup for simulator #" + std::to_string(message.id));
		TRN::Engine::Node::handle->scheduler[message.id](message.trial, message.offsets, message.durations);
	}
}
void TRN::Engine::Worker::process(const TRN::Engine::Message<TRN::Engine::Tag::FEEDFORWARD_WEIGHTS> &message)
{
	// std::cout << __FUNCTION__ << std::endl;
	if (TRN::Engine::Node::handle->feedforward_weights.find(message.id) == TRN::Engine::Node::handle->feedforward_weights.end())
		throw std::runtime_error("Perceived stimulus functor is not setup for simulator #" + std::to_string(message.id));
	TRN::Engine::Node::handle->feedforward_weights[message.id](message.elements, message.matrices, message.rows, message.cols);
	handle->remaining_initializations[message.id]--;
	send_configured(message.id);
}
void TRN::Engine::Worker::process(const TRN::Engine::Message<TRN::Engine::Tag::RECURRENT_WEIGHTS> &message)
{
	// std::cout << __FUNCTION__ << std::endl;
	if (TRN::Engine::Node::handle->recurrent.find(message.id) == TRN::Engine::Node::handle->recurrent.end())
		throw std::runtime_error("Recurrent stimulus functor is not setup for simulator #" + std::to_string(message.id));
	TRN::Engine::Node::handle->recurrent[message.id](message.elements, message.matrices, message.rows, message.cols);
	handle->remaining_initializations[message.id]--;
	send_configured(message.id);
}
void TRN::Engine::Worker::process(const TRN::Engine::Message<TRN::Engine::Tag::FEEDBACK_WEIGHTS> &message)
{
	// std::cout << __FUNCTION__ << std::endl;
	if (TRN::Engine::Node::handle->feedback_weights.find(message.id) == TRN::Engine::Node::handle->feedback_weights.end())
		throw std::runtime_error("Feedback stimulus functor is not setup for simulator #" + std::to_string(message.id));
	TRN::Engine::Node::handle->feedback_weights[message.id](message.elements, message.matrices, message.rows, message.cols);
	handle->remaining_initializations[message.id]--;
	send_configured(message.id);
}
void TRN::Engine::Worker::process(const TRN::Engine::Message<TRN::Engine::Tag::READOUT_WEIGHTS> &message) 
{
	// std::cout << __FUNCTION__ << std::endl;
	if (TRN::Engine::Node::handle->readout.find(message.id) == TRN::Engine::Node::handle->readout.end())
		throw std::runtime_error("Readout stimulus functor is not setup for simulator #" + std::to_string(message.id));
	TRN::Engine::Node::handle->readout[message.id](message.elements, message.matrices, message.rows, message.cols);
	handle->remaining_initializations[message.id]--;
	send_configured(message.id);
}




std::shared_ptr<TRN::Engine::Worker> TRN::Engine::Worker::create( const std::shared_ptr<TRN::Engine::Communicator> &communicator, const int &rank, const std::shared_ptr<TRN::Backend::Driver> &driver)
{
	// std::cout << __FUNCTION__ << std::endl;
	return std::make_shared<TRN::Engine::Worker>(communicator, rank, driver);
}
