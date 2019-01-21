#include "stdafx.h"
#include "Basic_impl.h"
#include "Helper/Logger.h"


TRN::Simulator::Basic::Basic(const std::function<void()> &trained, const std::function<void()> &primed, const std::function<void()> &tested) :
	handle(std::make_unique<Handle>())
{
	handle->trained = trained;
	handle->tested = tested;
	handle->primed = primed;
	handle->initialized = false;
}
TRN::Simulator::Basic::~Basic()
{
	handle.reset();
}


const std::vector<std::shared_ptr<TRN::Core::Mutator>> TRN::Simulator::Basic::get_mutators()
{
	return handle->mutators;
}

const std::shared_ptr<TRN::Core::Reservoir> TRN::Simulator::Basic::get_reservoir()
{
	return handle->reservoir;
}
const std::shared_ptr<TRN::Core::Loop> TRN::Simulator::Basic::get_loop()
{
	return handle->loop;
}
const std::shared_ptr<TRN::Core::Decoder> TRN::Simulator::Basic::get_decoder()
{
	return handle->decoder;
}
const std::shared_ptr<TRN::Core::Encoder> TRN::Simulator::Basic::get_encoder()
{
	return handle->encoder;
}
const std::shared_ptr<TRN::Core::Scheduler> TRN::Simulator::Basic::get_scheduler()
{
	return handle->scheduler;
}


void TRN::Simulator::Basic::set_feedforward(const std::shared_ptr<TRN::Core::Initializer> &feedforward)
{
	handle->feedforward = feedforward;
}
void TRN::Simulator::Basic::set_recurrent(const std::shared_ptr<TRN::Core::Initializer> &recurrent)
{
	handle->recurrent = recurrent;
}

void TRN::Simulator::Basic::set_readout(const std::shared_ptr<TRN::Core::Initializer> &readout)
{
	handle->readout = readout;
}

void TRN::Simulator::Basic::set_reservoir(const std::shared_ptr<TRN::Core::Reservoir> &reservoir)
{
	handle->reservoir = reservoir;
}
void TRN::Simulator::Basic::set_scheduler(const std::shared_ptr<TRN::Core::Scheduler> &scheduler)
{
	handle->scheduler = scheduler;
}

void TRN::Simulator::Basic::set_loop(const std::shared_ptr<TRN::Core::Loop> &loop)
{
	handle->loop = loop;
}
void TRN::Simulator::Basic::set_decoder(const std::shared_ptr<TRN::Core::Decoder> &decoder)
{
	handle->decoder = decoder;
}
void TRN::Simulator::Basic::set_encoder(const std::shared_ptr<TRN::Core::Encoder> &encoder)
{
	handle->encoder = encoder;
}
void TRN::Simulator::Basic::append_measurement(const std::shared_ptr<TRN::Core::Measurement::Abstraction> &measurement)
{
	handle->measurements.push_back(measurement);
}
void TRN::Simulator::Basic::append_mutator(const std::shared_ptr<TRN::Core::Mutator> &mutator)
{
	handle->mutators.push_back(mutator);
}
void TRN::Simulator::Basic::declare(const std::string &label, const std::string &tag, const std::shared_ptr<TRN::Core::Matrix> &sequence)
{
	

	if (label.empty())
		throw std::invalid_argument("Sequence label cannot be empty");
	if (tag.empty())
		throw std::invalid_argument("Sequence label cannot be empty");

	handle->sequences[key(label, tag)] = sequence;

	DEBUG_LOGGER << "Declared sequence " << label << " (" << sequence->get_rows() << "x" << sequence->get_cols() << ") with tag " << tag << " on device #" << sequence->get_implementor()->index() << " having name " << sequence->get_implementor()->name();
}

void TRN::Simulator::Basic::declare(const std::string &label, const std::string &tag, const std::shared_ptr<TRN::Core::Set> &set)
{
	if (label.empty())
		throw std::invalid_argument("Batch label cannot be empty");
	if (tag.empty())
		throw std::invalid_argument("Batch label cannot be empty");
	handle->sets[key(label, tag)] = set;
	DEBUG_LOGGER << "Declared set " << label << " with tag " << tag;
}
const std::shared_ptr<TRN::Core::Matrix>  TRN::Simulator::Basic::retrieve_sequence(const std::string &label, const std::string &tag)
{
	if (handle->sequences.find(key(label, tag)) == handle->sequences.end())
		throw std::invalid_argument("sequence label/tag is invalid : " + tag);
	return handle->sequences[key(label, tag)];
}
const std::shared_ptr<TRN::Core::Set>  TRN::Simulator::Basic::retrieve_set(const std::string &label, const std::string &tag)
{
	if (handle->sets.find(key(label, tag)) == handle->sets.end())
		throw std::invalid_argument("batch label/tag is invalid : " + tag);
	return handle->sets[key(label, tag)];
}

void TRN::Simulator::Basic::train(const unsigned long long &evaluation_id, const std::string &label, const std::string &incoming, const std::string &expected, const bool &reset_readout)
{
	if (!handle->initialized)
		throw std::logic_error("Simulator is not initialized");

	if (handle->sets.find(key(label, incoming)) == handle->sets.end())
		throw std::invalid_argument("Batch " + key(label, incoming) + " does not exist");
	if (handle->sets.find(key(label, expected)) == handle->sets.end())
		throw std::invalid_argument("Batch " + key(label, expected) + " does not exist");
	
	DEBUG_LOGGER << "Training simulator with set " << label << " and tags (" << incoming << ", " << expected << ")";
	if (reset_readout)
	{
		handle->reservoir->reset_readout();
		handle->scheduler->reset();
	}
	TRN::Core::Message::Payload<TRN::Core::Message::SET> sequence(label, incoming, expected, evaluation_id);
	handle->pending.push(sequence);
	TRN::Helper::Observable<TRN::Core::Message::Payload<TRN::Core::Message::SET>>::notify(sequence);
}
void TRN::Simulator::Basic::test(const unsigned long long &evaluation_id, const std::string &label, const std::string &incoming, const std::string &expected, const std::size_t &preamble, const bool &autonomous, const std::size_t &supplementary_generations)
{
	if (!handle->initialized)
		throw std::logic_error("Simulator is not initialized");
	if (handle->sequences.find(key(label, incoming)) == handle->sequences.end())
		throw std::invalid_argument("Sequence " + key(label, incoming) + " does not exist");
	if (handle->sequences.find(key(label, expected)) == handle->sequences.end())
		throw std::invalid_argument("Sequence " + key(label, expected) + " does not exist");

	DEBUG_LOGGER << "Testing simulator with sequence " << label << " and tags (" << incoming << ", " << expected << "), preamble " << preamble << ", autonomous generation " << autonomous;

	auto incoming_sequence = retrieve_sequence(label, incoming);
	auto expected_sequence = retrieve_sequence(label, expected);

	TRN::Helper::Observable<TRN::Core::Message::Payload<TRN::Core::Message::TEST>>::notify(TRN::Core::Message::Payload<TRN::Core::Message::TEST>(label, autonomous, preamble, supplementary_generations));
	auto total_cycles = expected_sequence->get_rows() + supplementary_generations;
	auto observations = expected_sequence->get_rows() + supplementary_generations;
	auto batch_size = handle->reservoir->get_batch_size();
	TRN::Core::Message::Payload<TRN::Core::Message::CYCLES> cycles(batch_size, total_cycles, observations);
	TRN::Core::Message::Payload<TRN::Core::Message::TARGET_SEQUENCE> target_sequence(expected_sequence);

	TRN::Helper::Observable<TRN::Core::Message::Payload<TRN::Core::Message::CYCLES>>::notify(cycles);
	TRN::Helper::Observable<TRN::Core::Message::Payload<TRN::Core::Message::TARGET_SEQUENCE>>::notify(target_sequence);
	handle->reservoir->test(evaluation_id, incoming_sequence, expected_sequence, preamble, autonomous, supplementary_generations);
}
void TRN::Simulator::Basic::initialize()
{
	TRN::Core::Simulator::initialize();
	if (handle->initialized)
		throw std::logic_error("Simulator is already initialized");
	if (!handle->loop)
		throw std::logic_error("No Loop object have been configured");
	if (!handle->scheduler)
		throw std::logic_error("No Scheduler object have been configured");
	if (!handle->reservoir)
		throw std::logic_error("No Reservoir object have been configured");
	if (!handle->feedforward)
		throw std::logic_error("No Initializer object have been configured for the feedforward weights");
	if (!handle->recurrent)
		throw std::logic_error("No Initializer object have been configured for the recurrent weights");
	if (!handle->readout)
		throw std::logic_error("No Initializer object have been configured for the readout weights");

	handle->reservoir->TRN::Helper::Observable<TRN::Core::Message::Payload<TRN::Core::Message::PREDICTION>>::attach(handle->loop);
	for (auto measurement : handle->measurements)
	{
		handle->reservoir->TRN::Helper::Observable<TRN::Core::Message::Payload<TRN::Core::Message::PREDICTION>>::attach(measurement);
		handle->loop->TRN::Helper::Observable<TRN::Core::Message::Payload<TRN::Core::Message::POSITION>>::attach(measurement);
		handle->loop->TRN::Helper::Observable<TRN::Core::Message::Payload<TRN::Core::Message::TARGET_TRAJECTORY>>::attach(measurement);
		TRN::Helper::Observable<TRN::Core::Message::Payload<TRN::Core::Message::TARGET_SEQUENCE>>::attach(measurement);
		TRN::Helper::Observable<TRN::Core::Message::Payload<TRN::Core::Message::TEST>>::attach(measurement);
	}
	handle->loop->TRN::Helper::Observable<TRN::Core::Message::Payload<TRN::Core::Message::STIMULUS>>::attach(handle->reservoir);

	handle->loop->set_delegate(shared_from_this());
	TRN::Helper::Observable<TRN::Core::Message::Payload<TRN::Core::Message::TEST>>::attach(handle->loop);

	TRN::Helper::Observable<TRN::Core::Message::Payload<TRN::Core::Message::SET>>::attach(handle->scheduler);

	//
	
	handle->scheduler->set_delegate(shared_from_this());

	if (!handle->mutators.empty())
	{
		handle->scheduler->attach(handle->mutators[0]);
		auto mutator_iterator = handle->mutators.begin();

		for (std::size_t k = 1; k < handle->mutators.size(); k++)
		{
			handle->mutators[k - 1]->attach(handle->mutators[k]);
		}
	}

	handle->reservoir->initialize(handle->feedforward, handle->recurrent, handle->readout);
	handle->reservoir->start();
	handle->initialized = true;
}
void TRN::Simulator::Basic::uninitialize()
{
	if (handle->initialized)
	{

		if (!handle->loop)
			throw std::logic_error("No Loop object have been configured");
		if (!handle->reservoir)
			throw std::logic_error("No Reservoir object have been configured");
		if (!handle->feedforward)
			throw std::logic_error("No Initializer object have been configured for the feedforward weights");
		if (!handle->recurrent)
			throw std::logic_error("No Initializer object have been configured for the recurrent weights");
		if (!handle->readout)
			throw std::logic_error("No Initializer object have been configured for the readout weights");
		if (!handle->scheduler)
			throw std::logic_error("No Scheduler object have been configured for the readout weights");
		
		handle->reservoir->stop();
//		handle->loop->detach(handle->reservoir);
		//handle->reservoir->detach(handle->loop);
		//handle->loop->reset_delegate();
		
		//TRN::Helper::Observable<TRN::Core::Message::Payload<TRN::Core::Message::TEST>>::detach(handle->loop);
		//TRN::Helper::Observable<TRN::Core::Message::Payload<TRN::Core::Message::BATCH>>::detach(handle->scheduler);
		
		//
		//handle->scheduler->detach(shared_from_this());
		handle->loop.reset();
		handle->reservoir.reset();
		handle->feedforward.reset();
		handle->recurrent.reset();
		handle->readout.reset();
		handle->scheduler.reset();

		handle->initialized = false;
		handle->sets.clear();
		handle->sequences.clear();
		//handle->loop->set_delegate(shared_from_this());
		/*
		
			
	
		*/
	}
}

void  TRN::Simulator::Basic::update(const TRN::Core::Message::Payload<TRN::Core::Message::SCHEDULING> &payload)
{
	if (handle->pending.empty())
		throw std::logic_error("train sequence queue is empty");

	auto scheduling = payload.get_scheduling();
	
	auto total_cycles = scheduling->get_total_duration();
	auto observations = scheduling->get_total_duration() / scheduling->get_durations().size();
	auto batch_size = handle->reservoir->get_batch_size();
	TRN::Core::Message::Payload<TRN::Core::Message::CYCLES> cycles(batch_size, total_cycles, observations);
	TRN::Helper::Observable<TRN::Core::Message::Payload<TRN::Core::Message::CYCLES>>::notify(cycles);


	auto sequence = handle->pending.front();
	auto incoming = retrieve_set(sequence.get_label(), sequence.get_incoming());
	auto expected = retrieve_set(sequence.get_label(), sequence.get_expected());
	handle->reservoir->train(payload.get_evaluation_id(), incoming->get_sequence(), expected->get_sequence(), payload.get_scheduling());
	handle->pending.pop();
}

void  TRN::Simulator::Basic::update(const TRN::Core::Message::Payload<TRN::Core::Message::TESTED> &payload)
{

	handle->tested();
}

void  TRN::Simulator::Basic::update(const TRN::Core::Message::Payload<TRN::Core::Message::CONFIGURED> &payload)
{

}

void  TRN::Simulator::Basic::update(const TRN::Core::Message::Payload<TRN::Core::Message::PRIMED> &payload)
{

	handle->primed();
}
void  TRN::Simulator::Basic::update(const TRN::Core::Message::Payload<TRN::Core::Message::TRAINED> &payload)
{
	handle->trained();
}


std::shared_ptr<TRN::Simulator::Basic> TRN::Simulator::Basic::create( const std::function<void()> &trained, const std::function<void()> &primed, const std::function<void()> &tested)
{
	return std::make_shared<TRN::Simulator::Basic>(trained, primed, tested);
}

const std::string TRN::Simulator::Basic::key(const std::string &label, const std::string &tag)
{
	return label + "/" + tag;
}