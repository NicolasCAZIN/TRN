#include "stdafx.h"
#include "Performances_impl.h"

TRN::Simulator::Performances::Performances(const std::shared_ptr<TRN::Core::Simulator> &decorated,
										   const std::function<void(const std::string &phase, const size_t &batch_size, const size_t &cycles, const float &gflops, const float &seconds)> &functor,
	const bool &train, const bool &prime, const bool &generate) :
	TRN::Helper::Decorator<TRN::Core::Simulator>(decorated),
	handle(std::make_unique<Handle>())
{
	handle->train = train;
	handle->prime = prime;
	handle->generate = generate;
	handle->functor = functor;
	handle->cycles = 0;
	handle->batch_size = 0;
	handle->observations = 0;

}
TRN::Simulator::Performances::~Performances()
{
	handle.reset();
}
const std::shared_ptr<TRN::Core::Reservoir> TRN::Simulator::Performances::get_reservoir()
{
	return decorated->get_reservoir();
}
const std::shared_ptr<TRN::Core::Loop> TRN::Simulator::Performances::get_loop()
{
	return decorated->get_loop();
}

const std::shared_ptr<TRN::Core::Scheduler> TRN::Simulator::Performances::get_scheduler()
{
	return decorated->get_scheduler();
}


const std::shared_ptr<TRN::Core::Matrix> TRN::Simulator::Performances::retrieve_sequence(const std::string &label, const std::string &tag)
{
	return decorated->retrieve_sequence(label, tag);
}
const std::shared_ptr<TRN::Core::Set> TRN::Simulator::Performances::retrieve_set(const std::string &label, const std::string &tag)
{
	return decorated->retrieve_set(label, tag);
}
void TRN::Simulator::Performances::set_feedforward(const std::shared_ptr<TRN::Core::Initializer> &feedforward)
{
	decorated->set_feedforward(feedforward);
}
void TRN::Simulator::Performances::set_recurrent(const std::shared_ptr<TRN::Core::Initializer> &recurrent)
{
	decorated->set_recurrent(recurrent);
}
void TRN::Simulator::Performances::set_feedback(const std::shared_ptr<TRN::Core::Initializer> &feedback)
{
	decorated->set_feedback(feedback);
}
void TRN::Simulator::Performances::set_readout(const std::shared_ptr<TRN::Core::Initializer> &readout)
{
	decorated->set_readout(readout);
}
void TRN::Simulator::Performances::set_reservoir(const std::shared_ptr<TRN::Core::Reservoir> &reservoir)
{

	decorated->set_reservoir(reservoir);
}
void TRN::Simulator::Performances::set_scheduler(const std::shared_ptr<TRN::Core::Scheduler> &scheduler)
{
	decorated->set_scheduler(scheduler);
}
void TRN::Simulator::Performances::set_loop(const std::shared_ptr<TRN::Core::Loop> &loop)
{
	decorated->set_loop(loop);
}

void TRN::Simulator::Performances::append_measurement(const std::shared_ptr<TRN::Core::Measurement::Abstraction> &measurement)
{
	decorated->append_measurement(measurement);
}


void TRN::Simulator::Performances::declare(const std::string &label, const std::string &tag, const std::shared_ptr<TRN::Core::Matrix> &sequence)
{
	decorated->declare(label, tag, sequence);
}

void TRN::Simulator::Performances::declare(const std::string &label, const std::string &tag, const std::shared_ptr<TRN::Core::Set> &set)
{
	decorated->declare(label, tag, set);
}
void TRN::Simulator::Performances::train(const std::string &label, const std::string &incoming, const std::string &expected)
{
	handle->start = std::chrono::high_resolution_clock::now();
	decorated->train(label, incoming, expected);
}
void TRN::Simulator::Performances::test(const std::string &label, const std::string &incoming, const std::string &expected, const std::size_t &preamble)
{
	handle->start = std::chrono::high_resolution_clock::now();
	handle->preamble = preamble;
	decorated->test(label, incoming, expected, preamble);
}


void TRN::Simulator::Performances::initialize()
{
	decorated->initialize();

	get_reservoir()->TRN::Helper::Observable<TRN::Core::Message::Payload<TRN::Core::Message::TRAINED>>::attach(shared_from_this());
	get_reservoir()->TRN::Helper::Observable<TRN::Core::Message::Payload<TRN::Core::Message::TESTED>>::attach(shared_from_this());
	get_reservoir()->TRN::Helper::Observable<TRN::Core::Message::Payload<TRN::Core::Message::PRIMED>>::attach(shared_from_this());

	get_scheduler()->attach(shared_from_this());
}
void TRN::Simulator::Performances::uninitialize()
{
	decorated->uninitialize();
}

void  TRN::Simulator::Performances::update(const TRN::Core::Message::Payload<TRN::Core::Message::SCHEDULING> &payload)
{
}
void TRN::Simulator::Performances::update(const TRN::Core::Message::Payload<TRN::Core::Message::CYCLES> &payload)
{
	handle->cycles = payload.get_cycles();
	handle->batch_size = payload.get_batch_size();
	handle->observations = payload.get_observations();
	notify(payload);
}
void  TRN::Simulator::Performances::update(const TRN::Core::Message::Payload<TRN::Core::Message::TESTED> &payload)
{
	if (handle->generate)
	{
		std::chrono::duration<float> seconds = std::chrono::duration_cast<std::chrono::duration<float>>(std::chrono::high_resolution_clock::now() - handle->start);

		auto flops = std::make_shared<TRN::Core::Message::Payload<TRN::Core::Message::FLOPS>>(0, 0);
		decorated->get_reservoir()->visit(flops);
		auto remaining = handle->observations - handle->preamble;
		auto reservoir_gflops = compute_gflops(remaining, flops->get_flops_per_epoch_factor(), remaining, flops->get_flops_per_cycle());
		decorated->get_loop()->visit(flops);
		auto loop_gflops = compute_gflops(remaining, flops->get_flops_per_epoch_factor(), remaining, flops->get_flops_per_cycle());
		handle->functor("GENERATE", handle->batch_size, handle->cycles, reservoir_gflops + loop_gflops, seconds.count());
	}
}
void  TRN::Simulator::Performances::update(const TRN::Core::Message::Payload<TRN::Core::Message::PRIMED> &payload)
{
	if (handle->prime)
	{
		std::chrono::duration<float> seconds = std::chrono::duration_cast<std::chrono::duration<float>>(std::chrono::high_resolution_clock::now() - handle->start);

		auto flops = std::make_shared<TRN::Core::Message::Payload<TRN::Core::Message::FLOPS>>(0, 0);
		decorated->get_reservoir()->visit(flops);
		auto gflops = compute_gflops(flops->get_flops_per_epoch_factor(), handle->preamble, flops->get_flops_per_cycle(), handle->preamble);
		handle->functor("PRIME", handle->batch_size, handle->preamble, gflops, seconds.count());
	}
}
void  TRN::Simulator::Performances::update(const TRN::Core::Message::Payload<TRN::Core::Message::TRAINED> &payload)
{
	if (handle->train)
	{
		std::chrono::duration<float> seconds = std::chrono::duration_cast<std::chrono::duration<float>>(std::chrono::high_resolution_clock::now() - handle->start);

		auto flops = std::make_shared<TRN::Core::Message::Payload<TRN::Core::Message::FLOPS>>(0, 0);
		decorated->get_reservoir()->visit(flops);

		auto observations = handle->observations;
		auto gflops = compute_gflops(observations, flops->get_flops_per_epoch_factor(), handle->cycles, flops->get_flops_per_cycle());
		handle->functor("TRAIN", handle->batch_size, handle->cycles, gflops, seconds.count());
	}
}

float TRN::Simulator::Performances::compute_gflops(const std::size_t &flops_per_epoch_factor, const std::size_t &observations, const std::size_t &flops_per_cycle, const std::size_t &cycles)
{
	return (observations * flops_per_epoch_factor + cycles * flops_per_cycle) * 10e-9f;
}

std::shared_ptr<TRN::Simulator::Performances> TRN::Simulator::Performances::create(const std::shared_ptr<TRN::Core::Simulator> decorated,
	const std::function<void(const std::string &phase, const size_t &batch_size, const size_t &cycles, const float &gflops, const float &seconds)> &functor,
	const bool &train, const bool &prime, const bool &test)
{
	return std::make_shared<TRN::Simulator::Performances>(decorated, functor, train, prime, test);
}

