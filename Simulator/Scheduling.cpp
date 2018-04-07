#include "stdafx.h"
#include "Scheduling_impl.h"

TRN::Simulator::Scheduling::Scheduling(const std::shared_ptr<TRN::Core::Simulator> &decorated,
	const std::function<void(const std::size_t &trial, const std::vector<int> &offsets, const std::vector<int> &durations)> &functor) :
	TRN::Helper::Decorator<TRN::Core::Simulator>(decorated),
	handle(std::make_unique<Handle>())
{
	handle->functor = functor;
}
TRN::Simulator::Scheduling::~Scheduling()
{
	handle.reset();
}

const std::vector<std::shared_ptr<TRN::Core::Mutator>> TRN::Simulator::Scheduling::get_mutators()
{
	return decorated->get_mutators();
}

const std::shared_ptr<TRN::Core::Reservoir> TRN::Simulator::Scheduling::get_reservoir()
{
	return decorated->get_reservoir();
}
const std::shared_ptr<TRN::Core::Loop> TRN::Simulator::Scheduling::get_loop()
{
	return decorated->get_loop();
}
const std::shared_ptr<TRN::Core::Decoder> TRN::Simulator::Scheduling::get_decoder()
{
	return decorated->get_decoder();
}
const std::shared_ptr<TRN::Core::Encoder> TRN::Simulator::Scheduling::get_encoder()
{
	return decorated->get_encoder();
}
const std::shared_ptr<TRN::Core::Scheduler> TRN::Simulator::Scheduling::get_scheduler()
{
	return decorated->get_scheduler();
}


const std::shared_ptr<TRN::Core::Matrix> TRN::Simulator::Scheduling::retrieve_sequence(const std::string &label, const std::string &tag)
{
	return decorated->retrieve_sequence(label, tag);
}
const std::shared_ptr<TRN::Core::Set> TRN::Simulator::Scheduling::retrieve_set(const std::string &label, const std::string &tag)
{
	return decorated->retrieve_set(label, tag);
}
void TRN::Simulator::Scheduling::set_feedforward(const std::shared_ptr<TRN::Core::Initializer> &feedforward)
{
	decorated->set_feedforward(feedforward);
}
void TRN::Simulator::Scheduling::set_recurrent(const std::shared_ptr<TRN::Core::Initializer> &recurrent)
{
	decorated->set_recurrent(recurrent);
}

void TRN::Simulator::Scheduling::set_readout(const std::shared_ptr<TRN::Core::Initializer> &readout)
{
	decorated->set_readout(readout);
}
void TRN::Simulator::Scheduling::set_reservoir(const std::shared_ptr<TRN::Core::Reservoir> &reservoir)
{

	decorated->set_reservoir(reservoir);
}
void TRN::Simulator::Scheduling::set_scheduler(const std::shared_ptr<TRN::Core::Scheduler> &scheduler)
{
	decorated->set_scheduler(scheduler);
}
void TRN::Simulator::Scheduling::set_loop(const std::shared_ptr<TRN::Core::Loop> &loop)
{
	decorated->set_loop(loop);
}

void TRN::Simulator::Scheduling::set_decoder(const std::shared_ptr<TRN::Core::Decoder> &decoder)
{
	decorated->set_decoder(decoder);
}
void TRN::Simulator::Scheduling::set_encoder(const std::shared_ptr<TRN::Core::Encoder> &encoder)
{
	decorated->set_encoder(encoder);
}
void TRN::Simulator::Scheduling::append_measurement(const std::shared_ptr<TRN::Core::Measurement::Abstraction> &measurement)
{
	decorated->append_measurement(measurement);
}

void TRN::Simulator::Scheduling::append_mutator(const std::shared_ptr<TRN::Core::Mutator> &mutator)
{
	decorated->append_mutator(mutator);
}
void TRN::Simulator::Scheduling::declare(const std::string &label, const std::string &tag, const std::shared_ptr<TRN::Core::Matrix> &sequence)
{
	decorated->declare(label, tag, sequence);
}

void TRN::Simulator::Scheduling::declare(const std::string &label, const std::string &tag, const std::shared_ptr<TRN::Core::Set> &set)
{
	decorated->declare(label, tag, set);
}
void TRN::Simulator::Scheduling::train(const unsigned long long &evaluation_id, const std::string &label, const std::string &incoming, const std::string &expected, const bool &reset_readout)
{

	decorated->train(evaluation_id, label, incoming, expected, reset_readout);
}
void TRN::Simulator::Scheduling::test(const unsigned long long &evaluation_id, const std::string &label, const std::string &incoming, const std::string &expected, const std::size_t &preamble, const bool &autonomous, const std::size_t &supplementary_generations)
{
	decorated->test(evaluation_id, label, incoming, expected, preamble, autonomous, supplementary_generations);
}


void TRN::Simulator::Scheduling::initialize()
{
	TRN::Core::Simulator::initialize();
	decorated->initialize();
}
void TRN::Simulator::Scheduling::uninitialize()
{
	TRN::Core::Simulator::uninitialize();
	decorated->uninitialize();
}

void  TRN::Simulator::Scheduling::update(const TRN::Core::Message::Payload<TRN::Core::Message::SCHEDULING> &payload)
{
	std::vector<int> offsets, durations;

	payload.get_scheduling()->to(offsets, durations);
	handle->functor(payload.get_evaluation_id(), offsets, durations);
}

void  TRN::Simulator::Scheduling::update(const TRN::Core::Message::Payload<TRN::Core::Message::CONFIGURED> &payload)
{

}
void  TRN::Simulator::Scheduling::update(const TRN::Core::Message::Payload<TRN::Core::Message::TESTED> &payload)
{

}
void  TRN::Simulator::Scheduling::update(const TRN::Core::Message::Payload<TRN::Core::Message::PRIMED> &payload)
{
	
}
void  TRN::Simulator::Scheduling::update(const TRN::Core::Message::Payload<TRN::Core::Message::TRAINED> &payload)
{

}



std::shared_ptr<TRN::Simulator::Scheduling> TRN::Simulator::Scheduling::create(const std::shared_ptr<TRN::Core::Simulator> decorated,
	const std::function<void(const std::size_t &trial, const std::vector<int> &offsets, const std::vector<int> &durations)> &functor)
{
	return std::make_shared<TRN::Simulator::Scheduling>(decorated, functor);
}

