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

TRN::Engine::Worker::Worker(const std::shared_ptr<TRN::Backend::Driver> &driver, const std::shared_ptr<TRN::Engine::Communicator> &communicator) :
	handle(std::make_unique<Handle>())
{
	handle->rank = communicator->rank();
	handle->driver = driver;
	handle->communicator = communicator;
	handle->receiver = std::thread(&TRN::Engine::Worker::receive, this);

	TRN::Engine::Message<TRN::Engine::Tag::WORKER> message;

	message.rank = handle->rank;
	message.name = driver->name();
	message.index = driver->index();
	message.host = communicator->host();
	communicator->send(message, 0);
}

TRN::Engine::Worker::~Worker()
{
	if (handle->receiver.joinable())
		handle->receiver.join();


	handle.reset();
}



template <TRN::Engine::Tag tag>
static TRN::Engine::Message<tag> unpack(const std::shared_ptr<TRN::Engine::Communicator> &communicator, const int &rank, unsigned int &id, size_t &number)
{
	TRN::Engine::Message<tag> message = communicator->receive<tag>(rank);

	id = message.id;
	number = message.number;

	return message;
}
void TRN::Engine::Worker::receive()
{
	handle->driver->toggle();
	bool stop = false;
	while (!stop)
	{
		try
		{
			std::string data;
			unsigned int id = 0;
			size_t number = 0;
			std::string cause = "";

			auto locked = handle->communicator.lock();
			if (!locked)
				throw std::runtime_error("Communicator is deleted");
			try
			{
				auto tag = locked->probe(handle->rank);
				//PrintThread{} << "worker received tag#" << tag << std::endl;
				switch (tag)
				{
					case TRN::Engine::QUIT:
					{
						auto message = unpack<TRN::Engine::QUIT>(locked, handle->rank, id, number);
					
						stop = true;
					}
					break;
					case TRN::Engine::ALLOCATE:
					{
						auto message = unpack<TRN::Engine::ALLOCATE>(locked,  handle->rank, id, number);
					
						if (handle->simulators.find(message.id) != handle->simulators.end())
							throw std::invalid_argument("Simulator #" + std::to_string(message.id) + "already exists");
						
						auto basic = TRN::Model::Simulator::Basic::create(
						[this, message]()
						{
							TRN::Engine::Message<TRN::Engine::TRAINED> trained;

							trained.id = message.id;

							auto locked = handle->communicator.lock();
							if (locked)
								locked->send(trained, 0);
						},
						[this, message]()
						{
							TRN::Engine::Message<TRN::Engine::PRIMED> primed;

							primed.id = message.id;
							auto locked = handle->communicator.lock();
							if (locked)
								locked->send(primed, 0);
						}
							,
						[this, message]()
						{
							TRN::Engine::Message<TRN::Engine::TESTED> tested;

							tested.id = message.id;
							auto locked = handle->communicator.lock();
							if (locked)
								locked->send(tested, 0);
						}
						);
						handle->simulators[message.id] = basic;
					
					
					}
					break;

					case TRN::Engine::DEALLOCATE:
					{
						auto message = unpack<TRN::Engine::DEALLOCATE>(locked,  handle->rank, id, number);

						if (handle->simulators.find(message.id) == handle->simulators.end())
							throw std::invalid_argument("Simulator #" + std::to_string(message.id) + " does not exist");
						handle->simulators.erase(message.id);
						//std::cout << "worker deallocated " << message.id << std::endl;
					}
					break;

					case TRN::Engine::TRAIN:
					{
						auto message = unpack<TRN::Engine::TRAIN>(locked,  handle->rank, id, number);
						if (handle->simulators.find(message.id) == handle->simulators.end())
							throw std::invalid_argument("Simulator #" + std::to_string(message.id) + " does not exist");
						handle->simulators[message.id]->train(message.label, message.incoming, message.expected);
					}
					break;

					case TRN::Engine::TEST:
					{
						auto message = unpack<TRN::Engine::TEST>(locked,  handle->rank, id, number);
						if (handle->simulators.find(message.id) == handle->simulators.end())
							throw std::invalid_argument("Simulator #" + std::to_string(message.id) + " does not exist");
						handle->simulators[message.id]->test(message.label, message.incoming, message.expected, message.preamble, message.supplementary_generations);
					}
					break;

					case TRN::Engine::DECLARE_SEQUENCE:
					{
						auto message = unpack<TRN::Engine::DECLARE_SEQUENCE>(locked,  handle->rank, id, number);
						if (handle->simulators.find(message.id) == handle->simulators.end())
							throw std::invalid_argument("Simulator #" + std::to_string(message.id) + " does not exist");
						handle->simulators[message.id]->declare(message.label, message.tag, TRN::Core::Matrix::create(handle->driver, message.sequence, message.observations, message.sequence.size() / message.observations));
					}
					break;

					case TRN::Engine::DECLARE_SET:
					{
						auto message = unpack<TRN::Engine::DECLARE_SET>(locked,  handle->rank, id, number);
						if (handle->simulators.find(message.id) == handle->simulators.end())
							throw std::invalid_argument("Simulator #" + std::to_string(message.id) + " does not exist");
						std::vector<std::shared_ptr<TRN::Core::Matrix>> sequences;
						for (auto sequence_label : message.labels)
						{
							sequences.push_back(handle->simulators[message.id]->retrieve_sequence(sequence_label, message.tag));
						}

						handle->simulators[message.id]->declare(message.label, message.tag, TRN::Core::Set::create(handle->driver, sequences));
					}
					break;

					case TRN::Engine::SETUP_STATES:
					{
						auto message = unpack<TRN::Engine::SETUP_STATES>(locked,  handle->rank, id, number);
						if (handle->simulators.find(message.id) == handle->simulators.end())
							throw std::invalid_argument("Simulator #" + std::to_string(message.id) + " does not exist");
						if (!handle->simulators[message.id]->get_reservoir())
							throw std::logic_error("Simulator #" + std::to_string(message.id) + " does not not have a reservoir to decorate");
						auto decorator = TRN::Model::Simulator::States::create(handle->simulators[message.id], [this, message]
						(const std::string &phase, const std::string &label, const std::vector<float> &samples, const std::size_t &rows, const std::size_t &cols)
						{
							TRN::Engine::Message<TRN::Engine::STATES> states;

							states.id = message.id;
							states.label = label;
							states.phase = phase;
							states.elements = samples;
							states.rows = rows;
							states.cols = cols;

							auto locked = handle->communicator.lock();
							if (locked)
								locked->send(states, 0);
						}, message.train, message.prime, message.generate);
						handle->simulators[message.id]->attach(decorator);
						handle->simulators[message.id] = decorator;

		
		
					}
					break;

					case TRN::Engine::SETUP_WEIGHTS:
					{
						auto message = unpack<TRN::Engine::SETUP_WEIGHTS>(locked,  handle->rank, id, number);
						if (handle->simulators.find(message.id) == handle->simulators.end())
							throw std::invalid_argument("Simulator #" + std::to_string(message.id) + " does not exist");
						if (!handle->simulators[message.id]->get_reservoir())
							throw std::logic_error("Simulator #" + std::to_string(message.id) + " does not not have a reservoir to decorate");
						auto decorator = TRN::Model::Simulator::Weights::create(handle->simulators[message.id], [this, message]
						(const std::string &phase, const std::string &label, const std::vector<float> &samples, const std::size_t &rows, const std::size_t &cols)
						{
							TRN::Engine::Message<TRN::Engine::WEIGHTS> weights;

							weights.id = message.id;
							weights.phase = phase;
							weights.label = label;
							weights.elements = samples;
							weights.rows = rows;
							weights.cols = cols;

							auto locked = handle->communicator.lock();
							if (locked)
								locked->send(weights, 0);
						}, message.initialization, message.train);
						handle->simulators[message.id]->attach(decorator);
						handle->simulators[message.id] = decorator;
					}
					break;

					case TRN::Engine::SETUP_PERFORMANCES:
					{
						auto message = unpack<TRN::Engine::SETUP_PERFORMANCES>(locked,  handle->rank, id, number);
						if (handle->simulators.find(message.id) == handle->simulators.end())
							throw std::invalid_argument("Simulator #" + std::to_string(message.id) + " does not exist");
						if (!handle->simulators[message.id]->get_reservoir())
							throw std::logic_error("Simulator #" + std::to_string(message.id) + " does not not have a reservoir to decorate");
						auto decorator = TRN::Model::Simulator::Performances::create(handle->simulators[message.id], [this, message]
						(const std::string &phase, const size_t &batch_size, const size_t &cycles, const float &gflops, const float &seconds)
						{
							TRN::Engine::Message<TRN::Engine::PERFORMANCES> performances;

							performances.id = message.id;
							performances.phase = phase;
					
							performances.cycles = cycles;
							performances.gflops = gflops;
							performances.seconds = seconds;
							performances.batch_size = batch_size;
							auto locked = handle->communicator.lock();
							if (locked)
								locked->send(performances, 0);
						}, message.train, message.prime, message.generate);
						handle->simulators[message.id]->attach(decorator);
						handle->simulators[message.id] = decorator;
					}
					break;
					case TRN::Engine::SETUP_SCHEDULING:
					{
						auto message = unpack<TRN::Engine::SETUP_SCHEDULING>(locked, handle->rank, id, number);
						if (handle->simulators.find(message.id) == handle->simulators.end())
							throw std::invalid_argument("Simulator #" + std::to_string(message.id) + " does not exist");
						if (!handle->simulators[message.id]->get_reservoir())
							throw std::logic_error("Simulator #" + std::to_string(message.id) + " does not not have a reservoir to decorate");
						auto decorator = TRN::Model::Simulator::Scheduling::create(handle->simulators[message.id], [this, message]
						(const std::vector<int> &offsets, const std::vector<int> &durations)
						{
							TRN::Engine::Message<TRN::Engine::SCHEDULING> performances;

							performances.id = message.id;
							performances.offsets = offsets;
							performances.durations = durations;
					
							auto locked = handle->communicator.lock();
							if (locked)
								locked->send(performances, 0);
						});
						//handle->simulators[message.id]->attach(decorator);
						handle->simulators[message.id] = decorator;
					}
					break;
					case TRN::Engine::CONFIGURE_BEGIN:
					{
						auto message = unpack<TRN::Engine::CONFIGURE_BEGIN>(locked,  handle->rank, id, number);
						if (handle->simulators.find(message.id) == handle->simulators.end())
							throw std::invalid_argument("Simulator #" + std::to_string(message.id) + " does not exist");
						handle->simulators[message.id]->uninitialize();
					}
					break;

					case TRN::Engine::CONFIGURE_END:
					{
						auto message = unpack<TRN::Engine::CONFIGURE_END>(locked,  handle->rank, id, number);
						if (handle->simulators.find(message.id) == handle->simulators.end())
							throw std::invalid_argument("Simulator #" + std::to_string(message.id) + " does not exist");
						handle->simulators[message.id]->initialize();
					}
					break;
					case TRN::Engine::CONFIGURE_MEASUREMENT_READOUT_MEAN_SQUARE_ERROR:
					{
						auto message = unpack<TRN::Engine::CONFIGURE_MEASUREMENT_READOUT_MEAN_SQUARE_ERROR>(locked, handle->rank, id, number);
						if (handle->simulators.find(message.id) == handle->simulators.end())
							throw std::invalid_argument("Simulator #" + std::to_string(message.id) + " does not exist");
						handle->simulators[message.id]->append_measurement(
							TRN::Model::Measurement::Sequence::create(
								TRN::Model::Measurement::MeanSquareError::create(handle->driver, 
								[this, message](const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols)
								{
									TRN::Engine::Message<TRN::Engine::MEASUREMENT_READOUT_MEAN_SQUARE_ERROR> measurement;

									measurement.id = message.id;
									measurement.elements = values;
								
									measurement.rows = rows;
									measurement.cols = cols;
						
									auto locked = handle->communicator.lock();
									if (locked)
										locked->send(measurement, 0);
						}), message.batch_size));
					}
					break;
					case TRN::Engine::CONFIGURE_MEASUREMENT_READOUT_FRECHET_DISTANCE:
					{
						auto message = unpack<TRN::Engine::CONFIGURE_MEASUREMENT_READOUT_FRECHET_DISTANCE>(locked, handle->rank, id, number);
						if (handle->simulators.find(message.id) == handle->simulators.end())
							throw std::invalid_argument("Simulator #" + std::to_string(message.id) + " does not exist");
						handle->simulators[message.id]->append_measurement(
							TRN::Model::Measurement::Sequence::create(
								TRN::Model::Measurement::FrechetDistance::create(handle->driver,
								[this, message](const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols)
								{
									TRN::Engine::Message<TRN::Engine::MEASUREMENT_READOUT_FRECHET_DISTANCE> measurement;

									measurement.id = message.id;
									measurement.elements = values;
									measurement.rows = rows;
									measurement.cols = cols;

									auto locked = handle->communicator.lock();
									if (locked)
										locked->send(measurement, 0);
								}), message.batch_size));
					}
					break;
					case TRN::Engine::CONFIGURE_MEASUREMENT_READOUT_CUSTOM:
					{
						auto message = unpack<TRN::Engine::CONFIGURE_MEASUREMENT_READOUT_CUSTOM>(locked, handle->rank, id, number);
						if (handle->simulators.find(message.id) == handle->simulators.end())
							throw std::invalid_argument("Simulator #" + std::to_string(message.id) + " does not exist");
						handle->simulators[message.id]->append_measurement(
							TRN::Model::Measurement::Sequence::create(
								TRN::Model::Measurement::Custom::create(handle->driver,
								[this, message](const std::vector<float> &predicted, const std::vector<float> &expected, const std::size_t &pages,  const std::size_t &rows, const  std::size_t &cols)
								{
									TRN::Engine::Message<TRN::Engine::MEASUREMENT_READOUT_CUSTOM> measurement;

									measurement.id = message.id;
									measurement.elements = predicted;
									measurement.expected = expected;
									measurement.matrices = pages;
									measurement.rows = rows;
									measurement.cols = cols;

									auto locked = handle->communicator.lock();
									if (locked)
										locked->send(measurement, 0);
								}), message.batch_size));
					}
					break;
					case TRN::Engine::CONFIGURE_MEASUREMENT_POSITION_MEAN_SQUARE_ERROR:
					{
						auto message = unpack<TRN::Engine::CONFIGURE_MEASUREMENT_POSITION_MEAN_SQUARE_ERROR>(locked, handle->rank, id, number);
						if (handle->simulators.find(message.id) == handle->simulators.end())
							throw std::invalid_argument("Simulator #" + std::to_string(message.id) + " does not exist");
						handle->simulators[message.id]->append_measurement(
							TRN::Model::Measurement::Position::create(
								TRN::Model::Measurement::MeanSquareError::create(handle->driver,
								[this, message](const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols)
								{
									TRN::Engine::Message<TRN::Engine::MEASUREMENT_POSITION_MEAN_SQUARE_ERROR> measurement;

									measurement.id = message.id;
									measurement.elements = values;
									measurement.rows = rows;
									measurement.cols = cols;

									auto locked = handle->communicator.lock();
									if (locked)
										locked->send(measurement, 0);
								}), message.batch_size));
					}
					break;
					case TRN::Engine::CONFIGURE_MEASUREMENT_POSITION_FRECHET_DISTANCE:
					{
						auto message = unpack<TRN::Engine::CONFIGURE_MEASUREMENT_POSITION_FRECHET_DISTANCE>(locked, handle->rank, id, number);
						if (handle->simulators.find(message.id) == handle->simulators.end())
							throw std::invalid_argument("Simulator #" + std::to_string(message.id) + " does not exist");
						handle->simulators[message.id]->append_measurement(
							TRN::Model::Measurement::Position::create(
								TRN::Model::Measurement::MeanSquareError::create(handle->driver,
								[this, message](const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols)
								{
									TRN::Engine::Message<TRN::Engine::MEASUREMENT_POSITION_FRECHET_DISTANCE> measurement;

									measurement.id = message.id;
									measurement.elements = values;
									measurement.rows = rows;
									measurement.cols = cols;

									auto locked = handle->communicator.lock();
									if (locked)
										locked->send(measurement, 0);
								}), message.batch_size));
					}
					break;
					case TRN::Engine::CONFIGURE_MEASUREMENT_POSITION_CUSTOM:
					{
						auto message = unpack<TRN::Engine::CONFIGURE_MEASUREMENT_POSITION_CUSTOM>(locked, handle->rank, id, number);
						if (handle->simulators.find(message.id) == handle->simulators.end())
							throw std::invalid_argument("Simulator #" + std::to_string(message.id) + " does not exist");
						handle->simulators[message.id]->append_measurement(
							TRN::Model::Measurement::Position::create(
								TRN::Model::Measurement::Custom::create(handle->driver,
								[this, message](const std::vector<float> &predicted, const std::vector<float> &expected, const std::size_t &pages,  const std::size_t &rows, const  std::size_t &cols)
								{
									TRN::Engine::Message<TRN::Engine::MEASUREMENT_POSITION_CUSTOM> measurement;

									measurement.id = message.id;
									measurement.elements = predicted;
									measurement.expected = expected;
									measurement.matrices = pages;
									measurement.rows = rows;
									measurement.cols = cols;

									auto locked = handle->communicator.lock();
									if (locked)
										locked->send(measurement, 0);
								}), message.batch_size));
					}
					break;


					case TRN::Engine::CONFIGURE_RESERVOIR_WIDROW_HOFF:
					{
						auto message = unpack<TRN::Engine::CONFIGURE_RESERVOIR_WIDROW_HOFF>(locked,  handle->rank, id, number);
						if (handle->simulators.find(message.id) == handle->simulators.end())
							throw std::invalid_argument("Simulator #" + std::to_string(message.id) + " does not exist");
						handle->simulators[message.id]->set_reservoir(TRN::Model::Reservoir::WidrowHoff::create(handle->driver, message.stimulus_size, message.prediction_size, message.reservoir_size, message.leak_rate, message.initial_state_scale, message.learning_rate, message.seed, message.batch_size));
					}
					break;

					case TRN::Engine::CONFIGURE_LOOP_COPY:
					{
						auto message = unpack<TRN::Engine::CONFIGURE_LOOP_COPY>(locked,  handle->rank, id, number);
						if (handle->simulators.find(message.id) == handle->simulators.end())
							throw std::invalid_argument("Simulator #" + std::to_string(message.id) + " does not exist");
						handle->simulators[message.id]->set_loop(TRN::Model::Loop::Copy::create(handle->driver, message.batch_size, message.stimulus_size));
					}
					break;

					case TRN::Engine::CONFIGURE_LOOP_SPATIAL_FILTER:
					{
						auto message = unpack<TRN::Engine::CONFIGURE_LOOP_SPATIAL_FILTER>(locked,  handle->rank, id, number);
						if (handle->simulators.find(message.id) == handle->simulators.end())
							throw std::invalid_argument("Simulator #" + std::to_string(message.id) + " does not exist");

						std::unique_lock<std::mutex> lock(handle->functors);
						if (handle->estimated_position.find(message.id) != handle->estimated_position.end())
						{
							throw std::runtime_error("Estimated position functor is already setup for simulator #" + std::to_string(message.id));
						}
						if (handle->perceived_stimulus.find(message.id) != handle->perceived_stimulus.end())
						{
							throw std::runtime_error("Perceived stimulus functor is already setup for simulator #" + std::to_string(message.id));
						}
						handle->simulators[message.id]->set_loop(TRN::Model::Loop::SpatialFilter::create(handle->driver, message.batch_size, message.stimulus_size,
							[this, message]
						(const std::vector<float> &values, const std::size_t &rows, const std::size_t &cols)
						{
							TRN::Engine::Message<POSITION> position;
							position.id = message.id;
							position.elements = values;
							position.rows = rows;
							position.cols = cols;
							auto locked = handle->communicator.lock();
							if (locked)
								locked->send(position, 0);
						},
							handle->estimated_position[message.id],
							[this, message]
						(const std::vector<float> &values, const std::size_t &rows, const std::size_t &cols)
						{
							TRN::Engine::Message<STIMULUS> stimulus;
							stimulus.id = message.id;
							stimulus.elements = values;
							stimulus.rows = rows;
							stimulus.cols = cols;
							auto locked = handle->communicator.lock();
							if (locked)
								locked->send(stimulus, 0);
						},
							handle->perceived_stimulus[message.id],
							message.rows, message.cols, message.x, message.y, message.response, message.sigma, message.radius, message.scale, message.tag));
					}
					break;

					case TRN::Engine::CONFIGURE_LOOP_CUSTOM:
					{
						auto message = unpack<TRN::Engine::CONFIGURE_LOOP_CUSTOM>(locked,  handle->rank, id, number);
						if (handle->simulators.find(message.id) == handle->simulators.end())
							throw std::invalid_argument("Simulator #" + std::to_string(message.id) + " does not exist");
						std::unique_lock<std::mutex> lock(handle->functors);
						if (handle->perceived_stimulus.find(message.id) != handle->perceived_stimulus.end())
						{
							throw std::runtime_error("Perceived stimulus functor is already setup for simulator #" + std::to_string(message.id));
						}
						handle->simulators[message.id]->set_loop(TRN::Model::Loop::Custom::create(handle->driver, message.batch_size, message.stimulus_size, [=]
						(const std::vector<float> &values, const std::size_t &rows, const std::size_t &cols)
						{
							TRN::Engine::Message<TRN::Engine::STIMULUS> stimulus;
							stimulus.id = message.id;
							stimulus.elements = values;
							stimulus.rows = rows;
							stimulus.cols = cols;
							auto locked = handle->communicator.lock();
							if (locked)
								locked->send(stimulus, 0);
						},
							handle->perceived_stimulus[message.id]
							));
					}
					break;

					case TRN::Engine::CONFIGURE_SCHEDULER_TILED:
					{
						auto message = unpack<TRN::Engine::CONFIGURE_SCHEDULER_TILED>(locked,  handle->rank, id, number);
						if (handle->simulators.find(message.id) == handle->simulators.end())
							throw std::invalid_argument("Simulator #" + std::to_string(message.id) + " does not exist");
						handle->simulators[message.id]->set_scheduler(TRN::Model::Scheduler::Tiled::create(handle->driver, message.epochs));
					}
					break;

					case TRN::Engine::CONFIGURE_SCHEDULER_SNIPPETS:
					{
						auto message = unpack<TRN::Engine::CONFIGURE_SCHEDULER_SNIPPETS>(locked,  handle->rank, id, number);
						if (handle->simulators.find(message.id) == handle->simulators.end())
							throw std::invalid_argument("Simulator #" + std::to_string(message.id) + " does not exist");
						handle->simulators[message.id]->set_scheduler(TRN::Model::Scheduler::Snippets::create(handle->driver, message.snippets_size, message.time_budget, message.tag));
					}
					break;

					case TRN::Engine::CONFIGURE_SCHEDULER_CUSTOM:
					{
						auto message = unpack<TRN::Engine::CONFIGURE_SCHEDULER_CUSTOM>(locked,  handle->rank, id, number);
						if (handle->simulators.find(message.id) == handle->simulators.end())
							throw std::invalid_argument("Simulator #" + std::to_string(message.id) + " does not exist");
						std::unique_lock<std::mutex> lock(handle->functors);
						if (handle->scheduler.find(message.id) != handle->scheduler.end())
						{
							throw std::runtime_error("Scheduler functor is already setup for simulator #" + std::to_string(message.id));
						}
						handle->simulators[message.id]->set_scheduler(TRN::Model::Scheduler::Custom::create(handle->driver,
							[=](const std::vector<float> &elements, const std::size_t &rows, const std::size_t &cols, const std::vector<int> &offsets, const std::vector<int> &durations)
						{
							TRN::Engine::Message<SCHEDULING_REQUEST> scheduling_request;
							scheduling_request.id = message.id;
							scheduling_request.elements = elements;
							scheduling_request.rows = rows;
							scheduling_request.cols = cols;
							scheduling_request.offsets = offsets;
							scheduling_request.durations = durations;
							auto locked = handle->communicator.lock();
							if (locked)
								locked->send(scheduling_request, 0);
						},
							handle->scheduler[message.id], message.tag));
					}
					break;

					case TRN::Engine::CONFIGURE_MUTATOR_SHUFFLE:
					{
						auto message = unpack<TRN::Engine::CONFIGURE_MUTATOR_SHUFFLE>(locked, handle->rank, id, number);
						if (handle->simulators.find(message.id) == handle->simulators.end())
							throw std::invalid_argument("Simulator #" + std::to_string(message.id) + " does not exist");
						handle->simulators[message.id]->append_mutator(TRN::Model::Mutator::Shuffle::create());
					}
					break;
					case TRN::Engine::CONFIGURE_MUTATOR_REVERSE:
					{
						auto message = unpack<TRN::Engine::CONFIGURE_MUTATOR_REVERSE>(locked, handle->rank, id, number);
						if (handle->simulators.find(message.id) == handle->simulators.end())
							throw std::invalid_argument("Simulator #" + std::to_string(message.id) + " does not exist");
						handle->simulators[message.id]->append_mutator(TRN::Model::Mutator::Reverse::create(message.rate, message.size));
					}
					break;
					case TRN::Engine::CONFIGURE_MUTATOR_PUNCH:
					{
						auto message = unpack<TRN::Engine::CONFIGURE_MUTATOR_PUNCH>(locked, handle->rank, id, number);
						if (handle->simulators.find(message.id) == handle->simulators.end())
							throw std::invalid_argument("Simulator #" + std::to_string(message.id) + " does not exist");
						handle->simulators[message.id]->append_mutator(TRN::Model::Mutator::Punch::create(message.rate, message.size, message.number));
					}
					break;
					case TRN::Engine::CONFIGURE_MUTATOR_CUSTOM:
					{
						auto message = unpack<TRN::Engine::CONFIGURE_SCHEDULER_CUSTOM>(locked, handle->rank, id, number);
						if (handle->simulators.find(message.id) == handle->simulators.end())
							throw std::invalid_argument("Simulator #" + std::to_string(message.id) + " does not exist");
						std::unique_lock<std::mutex> lock(handle->functors);
						if (handle->mutator.find(message.id) != handle->mutator.end())
						{
							throw std::runtime_error("Mutator functor is already setup for simulator #" + std::to_string(message.id));
						}
						handle->simulators[message.id]->append_mutator(TRN::Model::Mutator::Custom::create([=](const std::vector<int> &offsets, const std::vector<int> &durations)
						{
							TRN::Engine::Message<SCHEDULING> scheduling;

							scheduling.id = message.id;
							scheduling.offsets = offsets;
							scheduling.durations = durations;
							auto locked = handle->communicator.lock();
							if (locked)
								locked->send(scheduling, 0);
						},
							handle->mutator[message.id])
						);
					}
					break;
					case TRN::Engine::CONFIGURE_FEEDFORWARD_UNIFORM:
					{
						auto message = unpack<TRN::Engine::CONFIGURE_FEEDFORWARD_UNIFORM>(locked,  handle->rank, id, number);
						if (handle->simulators.find(message.id) == handle->simulators.end())
							throw std::invalid_argument("Simulator #" + std::to_string(message.id) + " does not exist");
						handle->simulators[message.id]->set_feedforward(TRN::Model::Initializer::Uniform::create(handle->driver, message.a, message.b, message.sparsity));
					}
					break;

					case TRN::Engine::CONFIGURE_FEEDFORWARD_GAUSSIAN:
					{
						auto message = unpack<TRN::Engine::CONFIGURE_FEEDFORWARD_GAUSSIAN>(locked,  handle->rank, id, number);
						if (handle->simulators.find(message.id) == handle->simulators.end())
							throw std::invalid_argument("Simulator #" + std::to_string(message.id) + " does not exist");
						handle->simulators[message.id]->set_feedforward(TRN::Model::Initializer::Gaussian::create(handle->driver, message.mu, message.sigma));
					}
					break;

					case TRN::Engine::CONFIGURE_FEEDFORWARD_CUSTOM:
					{
						auto message = unpack<TRN::Engine::CONFIGURE_FEEDFORWARD_CUSTOM>(locked,  handle->rank, id, number);
						if (handle->simulators.find(message.id) == handle->simulators.end())
							throw std::invalid_argument("Simulator #" + std::to_string(message.id) + " does not exist");
						std::unique_lock<std::mutex> lock(handle->functors);
						if (handle->feedforward_weights.find(message.id) != handle->feedforward_weights.end())
						{
							throw std::runtime_error("Feedforward functor is already setup for simulator #" + std::to_string(message.id));
						}
						handle->simulators[message.id]->set_feedforward(TRN::Model::Initializer::Custom::create(handle->driver, [=](const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)
						{
							TRN::Engine::Message<FEEDFORWARD_DIMENSIONS> feedforward_dimensions;

							feedforward_dimensions.id = message.id;
							feedforward_dimensions.matrices = matrices;
							feedforward_dimensions.rows = rows;
							feedforward_dimensions.cols = cols;
							feedforward_dimensions.seed = seed;
							auto locked = handle->communicator.lock();
							if (locked)
								locked->send(feedforward_dimensions, 0);
						}, handle->feedforward_weights[message.id]));
					}
					break;

					case TRN::Engine::CONFIGURE_FEEDBACK_UNIFORM:
					{
						auto message = unpack<TRN::Engine::CONFIGURE_FEEDBACK_UNIFORM>(locked,  handle->rank, id, number);
						if (handle->simulators.find(message.id) == handle->simulators.end())
							throw std::invalid_argument("Simulator #" + std::to_string(message.id) + " does not exist");
						handle->simulators[message.id]->set_feedback(TRN::Model::Initializer::Uniform::create(handle->driver, message.a, message.b, message.sparsity));
					}
					break;

					case TRN::Engine::CONFIGURE_FEEDBACK_GAUSSIAN:
					{
						auto message = unpack<TRN::Engine::CONFIGURE_FEEDBACK_GAUSSIAN>(locked,  handle->rank, id, number);
						if (handle->simulators.find(message.id) == handle->simulators.end())
							throw std::invalid_argument("Simulator #" + std::to_string(message.id) + " does not exist");
						handle->simulators[message.id]->set_feedback(TRN::Model::Initializer::Gaussian::create(handle->driver, message.mu, message.sigma));
					}
					break;

					case TRN::Engine::CONFIGURE_FEEDBACK_CUSTOM:
					{
						auto message = unpack<TRN::Engine::CONFIGURE_FEEDBACK_CUSTOM>(locked,  handle->rank, id, number);
						if (handle->simulators.find(message.id) == handle->simulators.end())
							throw std::invalid_argument("Simulator #" + std::to_string(message.id) + " does not exist");
						std::unique_lock<std::mutex> lock(handle->functors);
						if (handle->feedback_weights.find(message.id) != handle->feedback_weights.end())
						{
							throw std::runtime_error("Feedback functor is already setup for simulator #" + std::to_string(message.id));
						}
						handle->simulators[message.id]->set_feedback(TRN::Model::Initializer::Custom::create(handle->driver, [=](const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)
						{
							TRN::Engine::Message<FEEDBACK_DIMENSIONS> feedback_dimensions;

							feedback_dimensions.id = message.id;
							feedback_dimensions.matrices = matrices;
							feedback_dimensions.rows = rows;
							feedback_dimensions.cols = cols;
							feedback_dimensions.seed = seed;
							auto locked = handle->communicator.lock();
							if (locked)
								locked->send(feedback_dimensions, 0);
						}, handle->feedback_weights[message.id]));
					}
					break;

					case TRN::Engine::CONFIGURE_RECURRENT_UNIFORM:
					{
						auto message = unpack<TRN::Engine::CONFIGURE_RECURRENT_UNIFORM>(locked,  handle->rank, id, number);
						if (handle->simulators.find(message.id) == handle->simulators.end())
							throw std::invalid_argument("Simulator #" + std::to_string(message.id) + " does not exist");
						handle->simulators[message.id]->set_recurrent(TRN::Model::Initializer::Uniform::create(handle->driver, message.a, message.b, message.sparsity));
					}
					break;

					case TRN::Engine::CONFIGURE_RECURRENT_GAUSSIAN:
					{
						auto message = unpack<TRN::Engine::CONFIGURE_RECURRENT_GAUSSIAN>(locked,  handle->rank, id, number);
						if (handle->simulators.find(message.id) == handle->simulators.end())
							throw std::invalid_argument("Simulator #" + std::to_string(message.id) + " does not exist");
						handle->simulators[message.id]->set_recurrent(TRN::Model::Initializer::Gaussian::create(handle->driver, message.mu, message.sigma));
					}
					break;

					case TRN::Engine::CONFIGURE_RECURRENT_CUSTOM:
					{
						auto message = unpack<TRN::Engine::CONFIGURE_RECURRENT_CUSTOM>(locked,  handle->rank, id, number);
						if (handle->simulators.find(message.id) == handle->simulators.end())
							throw std::invalid_argument("Simulator #" + std::to_string(message.id) + " does not exist");
						std::unique_lock<std::mutex> lock(handle->functors);
						if (handle->recurrent.find(message.id) != handle->recurrent.end())
						{
							throw std::runtime_error("Recurrent functor is already setup for simulator #" + std::to_string(message.id));
						}
						handle->simulators[message.id]->set_recurrent(TRN::Model::Initializer::Custom::create(handle->driver, [=](const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)
						{
							TRN::Engine::Message<RECURRENT_DIMENSIONS> recurrent_dimensions;

							recurrent_dimensions.id = message.id;
							recurrent_dimensions.matrices = matrices;
							recurrent_dimensions.rows = rows;
							recurrent_dimensions.cols = cols;
							recurrent_dimensions.seed = seed;
							auto locked = handle->communicator.lock();
							if (locked)
								locked->send(recurrent_dimensions, 0);
						}, handle->recurrent[message.id]));
					}
					break;

					case TRN::Engine::CONFIGURE_READOUT_UNIFORM:
					{
						auto message = unpack<TRN::Engine::CONFIGURE_READOUT_UNIFORM>(locked,  handle->rank, id, number);
						if (handle->simulators.find(message.id) == handle->simulators.end())
							throw std::invalid_argument("Simulator #" + std::to_string(message.id) + " does not exist");
						handle->simulators[message.id]->set_readout(TRN::Model::Initializer::Uniform::create(handle->driver, message.a, message.b, message.sparsity));
					}
					break;

					case TRN::Engine::CONFIGURE_READOUT_GAUSSIAN:
					{
						auto message = unpack<TRN::Engine::CONFIGURE_READOUT_GAUSSIAN>(locked,  handle->rank, id, number);
						if (handle->simulators.find(message.id) == handle->simulators.end())
							throw std::invalid_argument("Simulator #" + std::to_string(message.id) + " does not exist");
						handle->simulators[message.id]->set_readout(TRN::Model::Initializer::Gaussian::create(handle->driver, message.mu, message.sigma));
					}
					break;

					case TRN::Engine::CONFIGURE_READOUT_CUSTOM:
					{
						auto message = unpack<TRN::Engine::CONFIGURE_READOUT_CUSTOM>(locked,  handle->rank, id, number);
						if (handle->simulators.find(message.id) == handle->simulators.end())
							throw std::invalid_argument("Simulator #" + std::to_string(message.id) + " does not exist");
						std::unique_lock<std::mutex> lock(handle->functors);
						if (handle->readout.find(message.id) != handle->readout.end())
						{
							throw std::runtime_error("Readout functor is already setup for simulator #" + std::to_string(message.id));
						}
						handle->simulators[message.id]->set_readout(TRN::Model::Initializer::Custom::create(handle->driver, [=](const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)
						{
							TRN::Engine::Message<READOUT_DIMENSIONS> readout_dimensions;

							readout_dimensions.id = message.id;
							readout_dimensions.matrices = matrices;
							readout_dimensions.rows = rows;
							readout_dimensions.cols = cols;
							readout_dimensions.seed = seed;
							auto locked = handle->communicator.lock();
							if (locked)
								locked->send(readout_dimensions, 0);
						}, handle->readout[message.id]));
					}
					break;

					case TRN::Engine::POSITION:
					{
						auto message = unpack<TRN::Engine::POSITION>(locked,  handle->rank, id, number);
						if (handle->estimated_position.find(message.id) == handle->estimated_position.end())
							throw std::runtime_error("Estimated position functor is not setup for simulator #" + std::to_string(message.id));
						handle->estimated_position[message.id](message.elements, message.rows, message.cols);
					}
					break;
					case TRN::Engine::STIMULUS:
					{
						auto message = unpack<TRN::Engine::STIMULUS>(locked,  handle->rank, id, number);
						if (handle->perceived_stimulus.find(message.id) == handle->perceived_stimulus.end())
							throw std::runtime_error("Perceived stimulus functor is not setup for simulator #" + std::to_string(message.id));
						handle->perceived_stimulus[message.id]( message.elements, message.rows, message.cols);
					}
					break;
					case TRN::Engine::SCHEDULING:
					{
						auto message = unpack<TRN::Engine::SCHEDULING>(locked,  handle->rank, id, number);
						if (message.is_from_mutator)
						{
							if (handle->mutator.find(message.id) == handle->mutator.end())
								throw std::runtime_error("Mutator functor is not setup for simulator #" + std::to_string(message.id));
							handle->mutator[message.id](message.offsets, message.durations);
						}
						else
						{
							if (handle->scheduler.find(message.id) == handle->scheduler.end())
								throw std::runtime_error("Scheduling functor is not setup for simulator #" + std::to_string(message.id));
							handle->scheduler[message.id](message.offsets, message.durations);
						}
					}
					break;
					case TRN::Engine::FEEDFORWARD_WEIGHTS:
					{
						auto message = unpack<TRN::Engine::FEEDFORWARD_WEIGHTS>(locked,  handle->rank, id, number);
						if (handle->feedforward_weights.find(message.id) == handle->feedforward_weights.end())
							throw std::runtime_error("Perceived stimulus functor is not setup for simulator #" + std::to_string(message.id));
						handle->feedforward_weights[message.id]( message.elements, message.matrices, message.rows, message.cols);
					}
					break;
					case TRN::Engine::RECURRENT_WEIGHTS:
					{
						auto message = unpack<TRN::Engine::RECURRENT_WEIGHTS>(locked,  handle->rank, id, number);
						if (handle->recurrent.find(message.id) == handle->recurrent.end())
							throw std::runtime_error("Recurrent stimulus functor is not setup for simulator #" + std::to_string(message.id));
						handle->recurrent[message.id]( message.elements, message.matrices, message.rows, message.cols);
					}
					break;
					case TRN::Engine::FEEDBACK_WEIGHTS:
					{
						auto message = unpack<TRN::Engine::FEEDBACK_WEIGHTS>(locked,  handle->rank, id, number);
						if (handle->feedback_weights.find(message.id) == handle->feedback_weights.end())
							throw std::runtime_error("Feedback stimulus functor is not setup for simulator #" + std::to_string(message.id));
						handle->feedback_weights[message.id]( message.elements, message.matrices, message.rows, message.cols);
					}
					break;
					case TRN::Engine::READOUT_WEIGHTS:
					{
						auto message = unpack<TRN::Engine::READOUT_WEIGHTS>(locked,  handle->rank, id, number);
						if (handle->readout.find(message.id) == handle->readout.end())
							throw std::runtime_error("Readout stimulus functor is not setup for simulator #" + std::to_string(message.id));
						handle->readout[message.id]( message.elements, message.matrices, message.rows, message.cols);
					}
					break;
					default:
						throw std::invalid_argument("unexpected tag " + std::to_string(tag));
				}
			}
			catch (std::exception &e)
			{
				stop = true;
				cause = e.what();
				std::cerr << "during processing message"<< cause << std::endl;
			}
			if (!stop)
			{
				TRN::Engine::Message<TRN::Engine::ACK> ack;

				ack.id = id;
				ack.number = number;
				ack.cause = cause;
				ack.success = cause.empty();

				locked->send(ack, 0);
			}
			else
			{
				TRN::Engine::Message<TRN::Engine::QUIT> quit;

				//PrintThread{} << "sending quit to broker" << std::endl;
				locked->send(quit, 0);
				//PrintThread{} << "sent quit to broker" << std::endl;
			}
			
		}
		catch (std::exception &e)
		{
			stop = true;
			std::cerr << "after processing message" << e.what() << std::endl;
		}
	}
	std::cout << "Worker #" << handle->rank << " quitted" << std::endl;
}



std::shared_ptr<TRN::Engine::Worker> TRN::Engine::Worker::create(const std::shared_ptr<TRN::Backend::Driver> &driver, const std::shared_ptr<TRN::Engine::Communicator> &communicator)
{
	return std::make_shared<TRN::Engine::Worker>(driver, communicator);
}
