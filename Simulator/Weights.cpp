#include "stdafx.h"
#include "Weights_impl.h"

TRN::Simulator::Weights::Weights(const std::shared_ptr<TRN::Core::Simulator> &decorated,
	const std::function<void(const std::string &phase, const std::string &label, const std::size_t &batch, const std::size_t &trial, const std::vector<float> &elements, const std::size_t &rows, const std::size_t &cols)> &functor, const bool &initialization, const bool &train) :
	TRN::Helper::Decorator<TRN::Core::Simulator>(decorated),
	handle(std::make_unique<Handle>())
{
	handle->train = train;
	handle->initialization = initialization;
	handle->weights = TRN::Core::Message::Payload<TRN::Core::Message::WEIGHTS>::create();
	handle->functor = functor;
	TRN::Helper::Visitable<TRN::Core::Message::Payload<TRN::Core::Message::WEIGHTS>>(handle->weights).accept(get_reservoir());
}
TRN::Simulator::Weights::~Weights()
{
	handle.reset();
}

const std::vector<std::shared_ptr<TRN::Core::Mutator>> TRN::Simulator::Weights::get_mutators()
{
	return decorated->get_mutators();
}

const std::shared_ptr<TRN::Core::Reservoir> TRN::Simulator::Weights::get_reservoir()
{
	return decorated->get_reservoir();
}
const std::shared_ptr<TRN::Core::Loop> TRN::Simulator::Weights::get_loop()
{
	return decorated->get_loop();
}
const std::shared_ptr<TRN::Core::Scheduler> TRN::Simulator::Weights::get_scheduler()
{
	return decorated->get_scheduler();
}

void TRN::Simulator::Weights::set_feedforward(const std::shared_ptr<TRN::Core::Initializer> &feedforward)
{
	decorated->set_feedforward(feedforward);
}
void TRN::Simulator::Weights::set_recurrent(const std::shared_ptr<TRN::Core::Initializer> &recurrent)
{
	decorated->set_recurrent(recurrent);
}
void TRN::Simulator::Weights::set_feedback(const std::shared_ptr<TRN::Core::Initializer> &feedback)
{
	decorated->set_feedback(feedback);
}
void TRN::Simulator::Weights::set_readout(const std::shared_ptr<TRN::Core::Initializer> &readout)
{
	decorated->set_readout(readout);
}
void TRN::Simulator::Weights::set_reservoir(const std::shared_ptr<TRN::Core::Reservoir> &reservoir)
{
	decorated->set_reservoir(reservoir);
}
void TRN::Simulator::Weights::set_scheduler(const std::shared_ptr<TRN::Core::Scheduler> &scheduler)
{
	decorated->set_scheduler(scheduler);
}
void TRN::Simulator::Weights::set_loop(const std::shared_ptr<TRN::Core::Loop> &loop)
{
	decorated->set_loop(loop);
}

void TRN::Simulator::Weights::append_measurement(const std::shared_ptr<TRN::Core::Measurement::Abstraction> &measurement)
{
	decorated->append_measurement(measurement);
}

void TRN::Simulator::Weights::append_mutator(const std::shared_ptr<TRN::Core::Mutator> &mutator)
{
	decorated->append_mutator(mutator);
}
const std::shared_ptr<TRN::Core::Matrix> TRN::Simulator::Weights::retrieve_sequence(const std::string &label, const std::string &tag)
{
	return decorated->retrieve_sequence(label, tag);
}
const std::shared_ptr<TRN::Core::Set> TRN::Simulator::Weights::retrieve_set(const std::string &label, const std::string &tag)
{
	return decorated->retrieve_set(label, tag);
}
void TRN::Simulator::Weights::declare(const std::string &label, const std::string &tag, const std::shared_ptr<TRN::Core::Matrix> &sequence)
{
	decorated->declare(label, tag, sequence);
}

void TRN::Simulator::Weights::declare(const std::string &label, const std::string &tag, const std::shared_ptr<TRN::Core::Set> &batch)
{
	decorated->declare(label, tag, batch);
}
void TRN::Simulator::Weights::train(const std::string &label, const std::string &incoming, const std::string &expected)
{
	decorated->train(label, incoming, expected);
}
void TRN::Simulator::Weights::test(const std::string &sequence, const std::string &incoming, const std::string &expected, const std::size_t &preamble, const bool &autonomous, const std::size_t &supplementary_generations)
{
	decorated->test(sequence, incoming, expected, preamble, autonomous, supplementary_generations);
}
void TRN::Simulator::Weights::initialize()
{
	TRN::Core::Simulator::initialize();
	decorated->initialize();

	
}
void TRN::Simulator::Weights::uninitialize()
{
	TRN::Core::Simulator::uninitialize();
	decorated->uninitialize();
}

void  TRN::Simulator::Weights::update(const TRN::Core::Message::Payload<TRN::Core::Message::SCHEDULING> &payload)
{

}

void  TRN::Simulator::Weights::update(const TRN::Core::Message::Payload<TRN::Core::Message::TESTED> &payload)
{
}
void  TRN::Simulator::Weights::update(const TRN::Core::Message::Payload<TRN::Core::Message::PRIMED> &payload)
{

}
void  TRN::Simulator::Weights::update(const TRN::Core::Message::Payload<TRN::Core::Message::CONFIGURED> &payload)
{
	if (handle->initialization)
	{
		

		for (std::size_t batch = 0; batch < decorated->get_reservoir()->get_batch_size(); batch++)
		{
			std::vector<float> feedforward_data;
			std::size_t feedforward_rows;
			std::size_t feedforward_cols;
			handle->weights->get_feedforward()->get_matrices(batch)->to(feedforward_data, feedforward_rows, feedforward_cols);
			handle->functor("INITIALIZATION", "feedforward", batch, get_reservoir()->get_trial(), feedforward_data, feedforward_rows, feedforward_cols);

			std::vector<float> recurrent_data;
			std::size_t recurrent_rows;
			std::size_t recurrent_cols;
			handle->weights->get_recurrent()->get_matrices(batch)->to(recurrent_data, recurrent_rows, recurrent_cols);
			handle->functor("INITIALIZATION", "recurrent", batch, get_reservoir()->get_trial(), recurrent_data, recurrent_rows, recurrent_cols);

			std::vector<float> feedback_data;
			std::size_t feedback_rows;
			std::size_t feedback_cols;
			handle->weights->get_feedback()->get_matrices(batch)->to(feedback_data, feedback_rows, feedback_cols);
			handle->functor("INITIALIZATION", "feedback", batch, get_reservoir()->get_trial(), feedback_data, feedback_rows, feedback_cols);

			std::vector<float> readout_data;
			std::size_t readout_rows;
			std::size_t readout_cols;
			handle->weights->get_readout()->get_matrices(batch)->to(readout_data, readout_rows, readout_cols);
			handle->functor("INITIALIZATION", "readout", batch, get_reservoir()->get_trial(), readout_data, readout_rows, readout_cols);
		}

	}

}

void  TRN::Simulator::Weights::update(const TRN::Core::Message::Payload<TRN::Core::Message::TRAINED> &payload)
{

	if (handle->train)
	{
		for (std::size_t batch = 0; batch < decorated->get_reservoir()->get_batch_size(); batch++)
		{
			std::vector<float> readout_data;
			std::size_t readout_rows;
			std::size_t readout_cols;
			handle->weights->get_readout()->get_matrices(batch)->to(readout_data,  readout_rows, readout_cols);
			handle->functor("TRAIN", "readout", batch, get_reservoir()->get_trial(), readout_data, readout_rows, readout_cols);
		}
	}
}


void TRN::Simulator::Weights::update(const TRN::Core::Message::Payload<TRN::Core::Message::CYCLES> &payload)
{
	notify(payload);
}

std::shared_ptr<TRN::Simulator::Weights> TRN::Simulator::Weights::create(const std::shared_ptr<TRN::Core::Simulator> decorated,
	const std::function<void(const std::string &phase, const std::string &label, const std::size_t &batch, const std::size_t &trial, const std::vector<float> &data, const std::size_t &rows, const std::size_t &cols)> &functor,const bool &initialization, const bool &train)
{
	return std::make_shared<TRN::Simulator::Weights>(decorated, functor, initialization, train);
}

