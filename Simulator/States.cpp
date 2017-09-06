#include "stdafx.h"
#include "States_impl.h"

TRN::Simulator::States::States(const std::shared_ptr<TRN::Core::Simulator> &decorated,
	const std::function<void(const std::string &phase, const std::string &label, const std::vector<float> &elements, const std::size_t &rows, const std::size_t &cols)> &functor,
	const bool &train, const bool &prime, const bool &generate) :
	TRN::Helper::Decorator<TRN::Core::Simulator>(decorated),
	handle(std::make_unique<Handle>())
{
	handle->train = train;
	handle->prime = prime;
	handle->generate = generate;
	handle->states = TRN::Core::Message::Payload<TRN::Core::Message::STATES>::create();
	handle->functor = functor;
}
TRN::Simulator::States::~States()
{
	handle.reset();
}

const std::vector<std::shared_ptr<TRN::Core::Mutator>> TRN::Simulator::States::get_mutators()
{
	return decorated->get_mutators();
}


const std::shared_ptr<TRN::Core::Reservoir> TRN::Simulator::States::get_reservoir()
{
	return decorated->get_reservoir();
}
const std::shared_ptr<TRN::Core::Loop> TRN::Simulator::States::get_loop()
{
	return decorated->get_loop();
}
const std::shared_ptr<TRN::Core::Scheduler> TRN::Simulator::States::get_scheduler()
{
	return decorated->get_scheduler();
}


void TRN::Simulator::States::set_feedforward(const std::shared_ptr<TRN::Core::Initializer> &feedforward)
{
	decorated->set_feedforward(feedforward);
}
void TRN::Simulator::States::set_recurrent(const std::shared_ptr<TRN::Core::Initializer> &recurrent)
{
	decorated->set_recurrent(recurrent);
}
void TRN::Simulator::States::set_feedback(const std::shared_ptr<TRN::Core::Initializer> &feedback)
{
	decorated->set_feedback(feedback);
}
void TRN::Simulator::States::set_readout(const std::shared_ptr<TRN::Core::Initializer> &readout)
{
	decorated->set_readout(readout);
}
void TRN::Simulator::States::set_reservoir(const std::shared_ptr<TRN::Core::Reservoir> &reservoir)
{
	decorated->set_reservoir(reservoir);
}
void TRN::Simulator::States::set_scheduler(const std::shared_ptr<TRN::Core::Scheduler> &scheduler)
{
	decorated->set_scheduler(scheduler);
}
void TRN::Simulator::States::set_loop(const std::shared_ptr<TRN::Core::Loop> &loop)
{
	decorated->set_loop(loop);
}

void TRN::Simulator::States::append_measurement(const std::shared_ptr<TRN::Core::Measurement::Abstraction> &measurement)
{
	decorated->append_measurement(measurement);
}

void TRN::Simulator::States::append_mutator(const std::shared_ptr<TRN::Core::Mutator> &mutator)
{
	decorated->append_mutator(mutator);
}

const std::shared_ptr<TRN::Core::Matrix> TRN::Simulator::States::retrieve_sequence(const std::string &label, const std::string &tag)
{
	return decorated->retrieve_sequence(label, tag);
}
const std::shared_ptr<TRN::Core::Set> TRN::Simulator::States::retrieve_set(const std::string &label, const std::string &tag)
{
	return decorated->retrieve_set(label, tag);
}
void TRN::Simulator::States::declare(const std::string &label, const std::string &tag, const std::shared_ptr<TRN::Core::Matrix> &sequence)
{
	decorated->declare(label, tag, sequence);
}

void TRN::Simulator::States::declare(const std::string &label, const std::string &tag, const std::shared_ptr<TRN::Core::Set> &batch)
{
	decorated->declare(label, tag, batch);
}

void TRN::Simulator::States::train(const std::string &sequence, const std::string &incoming, const std::string &expected)
{
	decorated->train(sequence, incoming, expected);
}
void TRN::Simulator::States::test(const std::string &sequence, const std::string &incoming, const std::string &expected, const std::size_t &preamble, const std::size_t &supplementary_generations)
{
	decorated->test(sequence, incoming, expected, preamble, supplementary_generations);
}
void TRN::Simulator::States::initialize()
{
	TRN::Core::Simulator::initialize();
	decorated->initialize();
}
void TRN::Simulator::States::uninitialize()
{
	decorated->uninitialize();
}

void TRN::Simulator::States::to_host(const std::string &phase)
{
	std::vector<float> stimulus_data;
	std::size_t stimulus_rows;
	std::size_t stimulus_cols;
	handle->states->get_stimulus()->to(stimulus_data, stimulus_rows, stimulus_cols);


	std::vector<float> desired_data;
	std::size_t desired_rows;
	std::size_t desired_cols;
	handle->states->get_desired()->to(desired_data, desired_rows, desired_cols);

	std::vector<float> prediction_data;
	std::size_t prediction_rows;
	std::size_t prediction_cols;
	handle->states->get_prediction()->to(prediction_data, prediction_rows, prediction_cols);

	std::vector<float> reservoir_data;
	std::size_t reservoir_rows;
	std::size_t reservoir_cols;
	handle->states->get_reservoir()->to(reservoir_data, reservoir_rows, reservoir_cols);

	decorated->get_reservoir()->synchronize();

	handle->functor(phase, "stimulus", stimulus_data, stimulus_rows, stimulus_cols);
	handle->functor(phase, "desired", desired_data, desired_rows, desired_cols);
	handle->functor(phase, "prediction", prediction_data, prediction_rows, prediction_cols);
	handle->functor(phase , "reservoir", reservoir_data, reservoir_rows, reservoir_cols);
}

void  TRN::Simulator::States::update(const TRN::Core::Message::Payload<TRN::Core::Message::SCHEDULING> &payload)
{
}

void TRN::Simulator::States::update(const TRN::Core::Message::Payload<TRN::Core::Message::CYCLES> &payload)
{
	handle->states->set_rows(payload.get_cycles());
	TRN::Helper::Visitable<TRN::Core::Message::Payload<TRN::Core::Message::STATES>>(handle->states).accept(get_reservoir());
	notify(payload);
}
void  TRN::Simulator::States::update(const TRN::Core::Message::Payload<TRN::Core::Message::TESTED> &payload)
{
	if (handle->generate)
		to_host("GENERATE");
}
void  TRN::Simulator::States::update(const TRN::Core::Message::Payload<TRN::Core::Message::PRIMED> &payload)
{
	if (handle->prime)
		to_host("PRIME");
}
void  TRN::Simulator::States::update(const TRN::Core::Message::Payload<TRN::Core::Message::TRAINED> &payload)
{
	if (handle->train)
		to_host("TRAIN");
}


std::shared_ptr<TRN::Simulator::States> TRN::Simulator::States::create(const std::shared_ptr<TRN::Core::Simulator> decorated,
	const std::function<void(const std::string &phase, const std::string &label, const std::vector<float> &data, const std::size_t &rows, const std::size_t &cols)> &functor,
	const bool &train, const bool &prime, const bool &generate)
{
	return std::make_shared<TRN::Simulator::States>(decorated, functor, train, prime, generate);
}

