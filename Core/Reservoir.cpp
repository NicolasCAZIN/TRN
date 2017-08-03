#include "stdafx.h"
#include "Reservoir_impl.h"


TRN::Core::Reservoir::Reservoir(
	const std::shared_ptr<TRN::Backend::Driver> &driver,
	const std::size_t &stimulus, const std::size_t &prediction,
	const std::size_t &reservoir,
	const float &leak_rate,
	const float &initial_state_scale,
	const unsigned long &seed,
	const std::size_t &batch_size
	) :
	TRN::Helper::Bridge<TRN::Backend::Driver>(driver),
	handle(std::make_unique<Handle>())
{
	handle->stimulus_size = stimulus;
	handle->reservoir_size = reservoir;
	handle->prediction_size = prediction;
	handle->batch_size = batch_size;
	
	driver->get_memory()->align(stimulus, handle->stimulus_stride);
	driver->get_memory()->align(reservoir, handle->reservoir_stride);
	driver->get_memory()->align(prediction, handle->prediction_stride);

	auto x_in_cols = handle->reservoir_stride + prediction;

	handle->leak_rate = leak_rate;
	handle->initial_state_scale = initial_state_scale;


	handle->batched_p = TRN::Core::Batch::create(driver, batch_size);
	handle->batched_u = TRN::Core::Batch::create(driver, batch_size);
	handle->batched_error = TRN::Core::Batch::create(driver, batch_size);
	handle->batched_incoming = TRN::Core::Batch::create(driver, batch_size);
	handle->batched_expected = TRN::Core::Batch::create(driver, batch_size);
	handle->batched_u_ffwd = TRN::Core::Batch::create(driver, batch_size);
	handle->batched_W_ffwd = TRN::Core::Batch::create(driver, batch_size);
	handle->batched_X_in = TRN::Core::Batch::create(driver, batch_size);
	handle->batched_W_in = TRN::Core::Batch::create(driver, batch_size);
	handle->batched_X_ro = TRN::Core::Batch::create(driver, batch_size);
	handle->batched_W_ro = TRN::Core::Batch::create(driver, batch_size);
	handle->batched_X_res = TRN::Core::Batch::create(driver, batch_size);
	handle->batched_W_rec = TRN::Core::Batch::create(driver, batch_size);
	handle->batched_W_fbck = TRN::Core::Batch::create(driver, batch_size);
	for (std::size_t batch = 0; batch < batch_size; batch++)
	{
		auto p = TRN::Core::Matrix::create(driver, reservoir, 1);
		auto u = TRN::Core::Matrix::create(driver, reservoir, 1);
		auto error = TRN::Core::Matrix::create(driver, prediction, 1);

		auto W_ffwd = TRN::Core::Matrix::create(driver,  reservoir, stimulus, false);
		auto W_in = TRN::Core::Matrix::create(driver, x_in_cols, reservoir, true);
		auto W_rec = TRN::Core::Matrix::create(driver, W_in, 0, 0, reservoir, reservoir);
		auto W_fbck = TRN::Core::Matrix::create(driver, W_in, handle->reservoir_stride, 0,  prediction, reservoir);


		auto W_ro = TRN::Core::Matrix::create(driver, reservoir, prediction, true);
		auto X_in = TRN::Core::Matrix::create(driver, x_in_cols, 1);
		auto X_res = TRN::Core::Matrix::create(driver, X_in, 0 ,0, reservoir, 1);
		auto X_ro = TRN::Core::Matrix::create(driver, X_in,   handle->reservoir_stride, 0,  prediction, 1);
	

		/*handle->_W_ffwd.push_back(W_ffwd);
		handle->_W_in.push_back(W_in);
		handle->_W_rec.push_back(W_rec);
		handle->_W_fbck.push_back(W_fbck);
		handle->_W_ro.push_back(W_ro);*/
		handle->batched_error->update(batch, error);
		handle->batched_u->update(batch, u);
		handle->batched_p->update(batch, p);

		handle->batched_W_ffwd->update(batch, W_ffwd);
		handle->batched_X_in->update(batch, X_in);
		handle->batched_W_in->update(batch, W_in);
		handle->batched_X_res->update(batch, X_res);
		handle->batched_X_ro->update(batch, X_ro);
		handle->batched_W_ro->update(batch, W_ro);
		handle->batched_W_rec->update(batch, W_rec);
		handle->batched_W_fbck->update(batch, W_fbck);
	}
		
	handle->states = TRN::Core::Message::Payload<TRN::Core::Message::STATES>::create();
	handle->states->set_global(TRN::Core::Matrix::create(driver, 0, 0, false));
	
	handle->gather_states = false;

	handle->thread = std::thread([&]()
	{
		get_implementor()->toggle();
		std::shared_ptr<TRN::Core::Batch> prediction;
		while (handle->prediction.dequeue(prediction))
		{
			TRN::Helper::Observable<TRN::Core::Message::Payload<TRN::Core::Message::PREDICTION>>::notify(prediction);
		}
	}
	);

	driver->get_algorithm()->preallocate(stimulus, reservoir, prediction, batch_size);
}

TRN::Core::Reservoir::~Reservoir()
{
	synchronize();
	handle->prediction.invalidate();
	if (handle->thread.joinable())
	{
		handle->thread.join();
	}

	handle.reset();
}


std::size_t TRN::Core::Reservoir::get_batch_size()
{
	return handle->batch_size;
}

void TRN::Core::Reservoir::synchronize()
{
	implementor->synchronize();
}
void TRN::Core::Reservoir::initialize(const std::shared_ptr<TRN::Core::Initializer> &feedforward,
										const std::shared_ptr<TRN::Core::Initializer> &recurrent,
										const std::shared_ptr<TRN::Core::Initializer> &feedback,
										const std::shared_ptr<TRN::Core::Initializer> &readout)
{
	implementor->toggle();
	if (!feedforward)
		throw std::invalid_argument("Feedforward initializer is not initialized");
	if (!recurrent)
		throw std::invalid_argument("Recurrent initializer is not initialized");
	if (!feedback)
		throw std::invalid_argument("Feedback initializer is not initialized");
	if (!readout)
		throw std::invalid_argument("Readout initializer is not initialized");


	feedforward->initialize(handle->seed, handle->batched_W_ffwd);
	recurrent->initialize(handle->seed, handle->batched_W_rec);

		/*std::vector<float> elements;
		std::size_t rows, cols;
		handle->batched_W_ffwd->get_matrices(batch)->to(elements, rows, cols);
		cv::Mat mat(rows, cols, CV_32F, elements.data());*/
		
	feedback->initialize(handle->seed, handle->batched_W_fbck);
	readout->initialize(handle->seed, handle->batched_W_ro);
}

void TRN::Core::Reservoir::test(const std::shared_ptr<TRN::Core::Matrix> &incoming, const std::shared_ptr<TRN::Core::Matrix> &expected, const std::size_t &preamble)
{
	if (!expected)
		throw std::invalid_argument("target expected is empty");
	if (!incoming)
		throw std::invalid_argument("target incoming is empty");
	if (preamble > expected->get_rows())
		throw std::invalid_argument("preamble (" + std::to_string(preamble) + ") is longer than the target sequence duration");



	auto sub_states = TRN::Core::Matrix::create(implementor, handle->states->get_global(), 0, 0, preamble, handle->states->get_global()->get_cols());
	auto sub_scheduling = TRN::Core::Scheduling::create(implementor, { (unsigned int)0 }, { (unsigned int)preamble });
	auto sub_incoming = TRN::Core::Matrix::create(implementor, incoming, 0, 0, preamble, incoming->get_cols());
	auto sub_expected = TRN::Core::Matrix::create(implementor, expected, 0, 0, preamble, expected->get_cols());

	for (std::size_t batch = 0; batch < handle->batch_size; batch++)
	{
		auto sub_u_ffwd = TRN::Core::Matrix::create(implementor,  preamble, 1);
		handle->batched_incoming->update(batch, sub_incoming);
		handle->batched_expected->update(batch, sub_expected);
		handle->batched_u_ffwd->update(batch, sub_u_ffwd);
	}

	implementor->get_algorithm()->prime(
			handle->batch_size,
			handle->seed,
			handle->stimulus_stride, handle->reservoir_stride, handle->prediction_stride,
			handle->stimulus_size, handle->reservoir_size, handle->prediction_size,
			handle->leak_rate, handle->initial_state_scale,
			sub_expected->get_elements(), sub_expected->get_rows(), sub_expected->get_cols(), sub_expected->get_stride(),

		handle->batched_incoming->get_elements(), handle->batched_incoming->get_rows(), handle->batched_incoming->get_cols(), handle->batched_incoming->get_strides(),
		handle->batched_expected->get_elements(), handle->batched_expected->get_rows(), handle->batched_expected->get_cols(), handle->batched_expected->get_strides(),
		handle->batched_W_ffwd->get_elements(), handle->batched_W_ffwd->get_rows(), handle->batched_W_ffwd->get_cols(), handle->batched_W_ffwd->get_strides(),
		handle->batched_u_ffwd->get_elements(), handle->batched_u_ffwd->get_rows(), handle->batched_u_ffwd->get_cols(), handle->batched_u_ffwd->get_strides(),
		handle->batched_X_in->get_elements(), handle->batched_X_in->get_rows(), handle->batched_X_in->get_cols(), handle->batched_X_in->get_strides(),
		handle->batched_W_in->get_elements(), handle->batched_W_in->get_rows(), handle->batched_W_in->get_cols(), handle->batched_W_in->get_strides(),
		handle->batched_u->get_elements(), handle->batched_u->get_rows(), handle->batched_u->get_cols(), handle->batched_u->get_strides(),
		handle->batched_p->get_elements(), handle->batched_p->get_rows(), handle->batched_p->get_cols(), handle->batched_p->get_strides(),
		handle->batched_X_res->get_elements(), handle->batched_X_res->get_rows(), handle->batched_X_res->get_cols(), handle->batched_X_res->get_strides(),
		handle->batched_X_ro->get_elements(), handle->batched_X_ro->get_rows(), handle->batched_X_ro->get_cols(), handle->batched_X_ro->get_strides(),
		handle->batched_W_ro->get_elements(), handle->batched_W_ro->get_rows(), handle->batched_W_ro->get_cols(), handle->batched_W_ro->get_strides(),
		handle->batched_error->get_elements(), handle->batched_error->get_rows(), handle->batched_error->get_cols(), handle->batched_error->get_strides(),

			sub_scheduling->get_offsets(), sub_scheduling->get_durations(), sub_scheduling->get_repetitions(),
			sub_states->get_elements(), sub_states->get_rows(), sub_states->get_cols(), sub_states->get_stride());
	synchronize();
	TRN::Helper::Observable<TRN::Core::Message::Payload<TRN::Core::Message::PRIMED>>::notify(TRN::Core::Message::Payload<TRN::Core::Message::PRIMED>());

	handle->target_expected = expected;
	handle->cycle = preamble;
	handle->max_cycle = expected->get_rows();
	handle->prediction.enqueue(handle->batched_X_ro);
}
#include <iostream>
void TRN::Core::Reservoir::update(const TRN::Core::Message::Payload<TRN::Core::Message::STIMULUS> &incoming)
{
	if (handle->cycle < handle->max_cycle)
	{
		auto sub_states = TRN::Core::Matrix::create(implementor, handle->states->get_global(), handle->cycle, 0, 1, handle->states->get_global()->get_cols());
		auto sub_scheduling = TRN::Core::Scheduling::create(implementor, { (unsigned int)0 }, { 1 });
		auto sub_expected = TRN::Core::Matrix::create(implementor, handle->target_expected, handle->cycle, 0, 1, handle->target_expected->get_cols());

		
		for (std::size_t batch = 0; batch < handle->batch_size; batch++)
		{
			auto sub_u_ffwd = TRN::Core::Matrix::create(implementor, 1, handle->reservoir_size);
			auto sub_incoming = incoming.get_stimulus()->get_matrices(batch);
			handle->batched_incoming->update(batch, sub_incoming);
			handle->batched_expected->update(batch, sub_expected);
			handle->batched_u_ffwd->update(batch, sub_u_ffwd);
		}

		implementor->get_algorithm()->generate( // generate
			handle->batch_size,
			handle->seed,
			handle->stimulus_stride, handle->reservoir_stride, handle->prediction_stride,
			handle->stimulus_size, handle->reservoir_size, handle->prediction_size,
			handle->leak_rate, handle->initial_state_scale,
			sub_expected->get_elements(), sub_expected->get_rows(), sub_expected->get_cols(), sub_expected->get_stride(),

			handle->batched_incoming->get_elements(), handle->batched_incoming->get_rows(), handle->batched_incoming->get_cols(), handle->batched_incoming->get_strides(),
			handle->batched_expected->get_elements(), handle->batched_expected->get_rows(), handle->batched_expected->get_cols(), handle->batched_expected->get_strides(),
			handle->batched_W_ffwd->get_elements(), handle->batched_W_ffwd->get_rows(), handle->batched_W_ffwd->get_cols(), handle->batched_W_ffwd->get_strides(),
			handle->batched_u_ffwd->get_elements(), handle->batched_u_ffwd->get_rows(), handle->batched_u_ffwd->get_cols(), handle->batched_u_ffwd->get_strides(),
			handle->batched_X_in->get_elements(), handle->batched_X_in->get_rows(), handle->batched_X_in->get_cols(), handle->batched_X_in->get_strides(),
			handle->batched_W_in->get_elements(), handle->batched_W_in->get_rows(), handle->batched_W_in->get_cols(), handle->batched_W_in->get_strides(),
			handle->batched_u->get_elements(), handle->batched_u->get_rows(), handle->batched_u->get_cols(), handle->batched_u->get_strides(),
			handle->batched_p->get_elements(), handle->batched_p->get_rows(), handle->batched_p->get_cols(), handle->batched_p->get_strides(),
			handle->batched_X_res->get_elements(), handle->batched_X_res->get_rows(), handle->batched_X_res->get_cols(), handle->batched_X_res->get_strides(),
			handle->batched_X_ro->get_elements(), handle->batched_X_ro->get_rows(), handle->batched_X_ro->get_cols(), handle->batched_X_ro->get_strides(),
			handle->batched_W_ro->get_elements(), handle->batched_W_ro->get_rows(), handle->batched_W_ro->get_cols(), handle->batched_W_ro->get_strides(),
			handle->batched_error->get_elements(), handle->batched_error->get_rows(), handle->batched_error->get_cols(), handle->batched_error->get_strides(),
			sub_scheduling->get_offsets(), sub_scheduling->get_durations(), sub_scheduling->get_repetitions(),
			sub_states->get_elements(), sub_states->get_rows(), sub_states->get_cols(), sub_states->get_stride());
		handle->cycle++;
		handle->prediction.enqueue(handle->batched_X_ro);
	
	}
	else
	{
		synchronize();
		handle->target_expected.reset();
		handle->max_cycle = 0;
		handle->cycle = 0;
	
		TRN::Helper::Observable<TRN::Core::Message::Payload<TRN::Core::Message::TESTED>>::notify(TRN::Core::Message::Payload<TRN::Core::Message::TESTED>());
	}
}
void TRN::Core::Reservoir::train(const std::shared_ptr<TRN::Core::Matrix> &incoming, const std::shared_ptr<TRN::Core::Matrix> &expected, const std::shared_ptr<TRN::Core::Scheduling> &scheduling)
{
	for (std::size_t batch = 0; batch < handle->batch_size; batch++)
	{
		auto sub_u_ffwd = TRN::Core::Matrix::create(implementor, handle->reservoir_size, incoming->get_rows());
		handle->batched_incoming->update(batch, incoming);
		handle->batched_expected->update(batch, expected);
		handle->batched_u_ffwd->update(batch, sub_u_ffwd);
	}

	train(incoming, expected, scheduling, handle->states->get_global());
	/*{
		std::vector<float> x;
		std::size_t x_n;
		std::vector<std::size_t> x_r, x_c;

		handle->batched_u->to(x, x_n, x_r, x_c);

	
		cv::Mat mat( x_n, x_r[0], CV_32F, x.data());
		int c = 0;
	}*/
	synchronize();
	TRN::Helper::Observable<TRN::Core::Message::Payload<TRN::Core::Message::TRAINED>>::notify(TRN::Core::Message::Payload<TRN::Core::Message::TRAINED>());
}

void TRN::Core::Reservoir::visit(std::shared_ptr<TRN::Core::Message::Payload<TRN::Core::Message::WEIGHTS>> &payload)
{
	throw std::runtime_error("TO DO");
	/*payload->set_readout(handle->W_ro);
	payload->set_feedforward(handle->W_ffwd);
	payload->set_feedback(handle->W_fbck);
	payload->set_recurrent(handle->W_rec);*/
}
void TRN::Core::Reservoir::visit(std::shared_ptr<TRN::Core::Message::Payload<TRN::Core::Message::STATES>> &payload)
{
	auto duration = payload->get_rows();

	const std::size_t size = handle->batch_size * (handle->stimulus_stride + handle->prediction_stride + handle->prediction_stride + handle->reservoir_stride);
	auto global = TRN::Core::Matrix::create(implementor, duration, size, true);
	std::size_t offset = 0;
	auto stimulus = TRN::Core::Matrix::create(implementor, global, 0, offset, duration, handle->batch_size * handle->stimulus_size); offset += handle->batch_size * handle->stimulus_stride;
	auto desired = TRN::Core::Matrix::create(implementor, global, 0, offset, duration, handle->batch_size * handle->prediction_size); offset += handle->batch_size * handle->prediction_stride;
	auto reservoir = TRN::Core::Matrix::create(implementor, global, 0, offset, duration, handle->batch_size * handle->reservoir_size); offset += handle->batch_size * handle->reservoir_stride;

	auto prediction = TRN::Core::Matrix::create(implementor, global, 0, offset, duration, handle->batch_size * handle->prediction_size); offset += handle->batch_size * handle->prediction_stride;



	assert(offset == size);
	payload->set_global(global);
	payload->set_reservoir(reservoir);
	payload->set_prediction(prediction);
	payload->set_desired(desired);
	payload->set_stimulus(stimulus);
	handle->states = payload;
}
void TRN::Core::Reservoir::visit(std::shared_ptr<TRN::Core::Message::Payload<TRN::Core::Message::FLOPS>> &payload)
{
	size_t flops_per_cycle = 0;
	flops_per_cycle += (handle->reservoir_size + handle->prediction_size) * handle->reservoir_size * 2; // w_in * x_in
	//flops_per_cycle += handle->reservoir_size * 4; // update euler
	//flops_per_cycle += handle->reservoir_size  * (3 + 50 + 10); // tanh
	//flops_per_cycle += handle->prediction_size * handle->reservoir_size * 2; //xro= W_ro * x_res
	
	payload->set_flops_per_cycle(flops_per_cycle);

	size_t flops_per_epoch_factor = 0;
	flops_per_epoch_factor += handle->reservoir_size * handle->stimulus_size * 2;
	payload->set_flops_per_epoch_factor(flops_per_epoch_factor);
}

