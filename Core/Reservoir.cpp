#include "stdafx.h"
#include "Reservoir_impl.h"


TRN::Core::Reservoir::Reservoir(
	const std::shared_ptr<TRN::Backend::Driver> &driver,
	const std::size_t &stimulus, const std::size_t &prediction,
	const std::size_t &reservoir,
	const float &leak_rate,
	const float &initial_state_scale,
	const unsigned long &seed,
	const std::size_t &batch_size,
	const std::size_t &mini_batch_size
	) :
	TRN::Helper::Bridge<TRN::Backend::Driver>(driver),
	handle(std::make_unique<Handle>())
{
	handle->stimulus_size = stimulus;
	handle->reservoir_size = reservoir;
	handle->prediction_size = prediction;
	handle->batch_size = batch_size;
	handle->mini_batch_size = mini_batch_size;
	handle->seed = seed;
	driver->get_memory()->align(stimulus, handle->stimulus_stride);
	driver->get_memory()->align(reservoir, handle->reservoir_stride);
	driver->get_memory()->align(prediction, handle->prediction_stride);

	auto x_in_cols = handle->reservoir_stride + prediction;

	handle->leak_rate = leak_rate;
	handle->initial_state_scale = initial_state_scale;
	std::vector<float> one = { 1.0f };
	std::vector<float> zero = { 0.0f };
	handle->one = TRN::Core::Matrix::create(driver,one, 1, 1);
	handle->zero = TRN::Core::Matrix::create(driver, zero, 1, 1);
	handle->batched_p = TRN::Core::Batch::create(driver, batch_size);
	handle->batched_u = TRN::Core::Batch::create(driver, batch_size);
	handle->batched_post = TRN::Core::Batch::create(driver, batch_size);
	handle->bundled_pre = TRN::Core::Bundle::create(driver, 2);
	handle->bundled_desired = TRN::Core::Bundle::create(driver, 2);
	handle->batched_incoming = TRN::Core::Batch::create(driver, batch_size);
	handle->batched_expected = TRN::Core::Batch::create(driver, batch_size);
	handle->batched_u_ffwd = TRN::Core::Batch::create(driver, batch_size);
	handle->batched_W_ffwd = TRN::Core::Batch::create(driver, batch_size);
	handle->batched_X_ro = TRN::Core::Batch::create(driver, batch_size);
	handle->batched_W_ro = TRN::Core::Batch::create(driver, batch_size);
	handle->batched_W_ro_reset = TRN::Core::Batch::create(driver, batch_size);
	handle->batched_X_res = TRN::Core::Batch::create(driver, batch_size);
	handle->batched_W_rec = TRN::Core::Batch::create(driver, batch_size);

	

	for (std::size_t bundle = 0; bundle < 2; bundle++)
	{
		auto batched_pre = TRN::Core::Batch::create(driver, batch_size);
		auto batched_desired = TRN::Core::Batch::create(driver, batch_size);
		for (std::size_t batch = 0; batch < batch_size; batch++)
		{
			auto pre = TRN::Core::Matrix::create(driver, mini_batch_size, reservoir);
			auto desired = TRN::Core::Matrix::create(driver, mini_batch_size, prediction);
			batched_pre->update(batch, pre);
			batched_desired->update(batch, desired);
		}
		handle->bundled_pre->update(bundle, batched_pre);
		handle->bundled_desired->update(bundle, batched_desired);
	}

	for (std::size_t batch = 0; batch < batch_size; batch++)
	{
		auto p = TRN::Core::Matrix::create(driver, 1, reservoir);
		auto u = TRN::Core::Matrix::create(driver, 1, reservoir);
		auto post = TRN::Core::Matrix::create(driver, mini_batch_size, prediction);
	
		auto W_ffwd = TRN::Core::Matrix::create(driver,    stimulus, reservoir, true);
		auto W_rec = TRN::Core::Matrix::create(driver, reservoir, reservoir, true);

		auto W_ro = TRN::Core::Matrix::create(driver, reservoir, prediction, true);
		auto W_ro_reset = TRN::Core::Matrix::create(driver, reservoir, prediction, true);
		auto X_res = TRN::Core::Matrix::create(driver, 1, reservoir, true);
		auto X_ro = TRN::Core::Matrix::create(driver, 1, prediction, true);
	

		handle->batched_post->update(batch, post);

		handle->batched_u->update(batch, u);
		handle->batched_p->update(batch, p);

		handle->batched_W_ffwd->update(batch, W_ffwd);
	
		handle->batched_X_res->update(batch, X_res);
		handle->batched_X_ro->update(batch, X_ro);
		handle->batched_W_ro->update(batch, W_ro);
		handle->batched_W_ro_reset->update(batch, W_ro_reset);
		handle->batched_W_rec->update(batch, W_rec);
	}
		
	
	handle->states = TRN::Core::Message::Payload<TRN::Core::Message::STATES>::create();
	handle->states->set_global(TRN::Core::Matrix::create(driver, 0, 0, false));
	
	handle->gather_states = false;



	driver->get_algorithm()->preallocate(stimulus, reservoir, prediction, batch_size);
}


void TRN::Core::Reservoir::start()
{
	handle->thread = std::thread([&]()
	{
		get_implementor()->toggle();
		std::tuple<std::shared_ptr<TRN::Core::Batch>, unsigned long long> tuple;
		while (handle->prediction.dequeue(tuple))
		{
			TRN::Helper::Observable<TRN::Core::Message::Payload<TRN::Core::Message::PREDICTION>>::notify(TRN::Core::Message::Payload<TRN::Core::Message::PREDICTION>(std::get<0>(tuple), std::get<1>(tuple)));
		}
	}
	);
}

void TRN::Core::Reservoir::stop()
{
	synchronize();
	handle->prediction.invalidate();
	if (handle->thread.joinable())
	{
		handle->thread.join();
	}
}

TRN::Core::Reservoir::~Reservoir()
{
	handle.reset();
}


std::size_t TRN::Core::Reservoir::get_batch_size()
{
	return handle->batch_size;
}

std::size_t TRN::Core::Reservoir::get_reservoir_size()
{
	return handle->reservoir_size;
}
std::size_t TRN::Core::Reservoir::get_stimulus_size()
{
	return handle->stimulus_size;
}
std::size_t TRN::Core::Reservoir::get_prediction_size()
{
	return handle->prediction_size;
}

void TRN::Core::Reservoir::synchronize()
{
	implementor->synchronize();
}

void TRN::Core::Reservoir::initialize()
{
	handle->batched_W_ro_reset->from(*handle->batched_W_ro);
	TRN::Helper::Observable<TRN::Core::Message::Payload<TRN::Core::Message::CONFIGURED>>::notify(TRN::Core::Message::Payload<TRN::Core::Message::CONFIGURED>());
}

void TRN::Core::Reservoir::initialize(const std::shared_ptr<TRN::Core::Initializer> &feedforward,
										const std::shared_ptr<TRN::Core::Initializer> &recurrent,								
										const std::shared_ptr<TRN::Core::Initializer> &readout)
{
	implementor->toggle();

	if (!feedforward)
		throw std::invalid_argument("Feedforward initializer is not initialized");
	if (!recurrent)
		throw std::invalid_argument("Recurrent initializer is not initialized");
	if (!readout)
		throw std::invalid_argument("Readout initializer is not initialized");
	// blank self connection

	feedforward->initialize(handle->seed, handle->batched_W_ffwd);
	recurrent->initialize(handle->seed, handle->batched_W_rec, true);	
	readout->initialize(handle->seed, handle->batched_W_ro);
}

void TRN::Core::Reservoir::reset_readout()
{
	handle->batched_W_ro->from(*handle->batched_W_ro_reset);
}void operator << (cv::Mat &mat, const std::shared_ptr<TRN::Core::Matrix> &matrix)
{
	std::vector<float> elements;
	std::size_t rows, cols;

	matrix->to(elements, rows, cols);
	cv::Mat tmp(rows, cols, CV_32F, elements.data());
	tmp.copyTo(mat);
}


void TRN::Core::Reservoir::test(const unsigned long long &evaluation_id, const std::shared_ptr<TRN::Core::Matrix> &incoming, const std::shared_ptr<TRN::Core::Matrix> &expected, const std::size_t &preamble, const bool &autonomous_generation, const std::size_t &supplementary_generations)
{
	if (!expected)
		throw std::invalid_argument("target expected is empty");
	if (!incoming)
		throw std::invalid_argument("target incoming is empty");
	if (preamble > expected->get_rows())
		throw std::invalid_argument("preamble (" + std::to_string(preamble) + ") is longer than the target sequence duration");

	auto sub_states = TRN::Core::Matrix::create(implementor, handle->states->get_global(), 0, 0, preamble, handle->states->get_global()->get_cols());

	std::vector<int> durations(1);
	std::vector<int> offsets(preamble);
	std::iota(offsets.begin(), offsets.end(), 0);
	durations[0] = preamble;


	/*cv::Mat cv_inc, cv_exp;

	cv_inc << incoming;
	cv_exp << expected;*/

	auto sub_scheduling = TRN::Core::Scheduling::create(offsets, durations);
	auto sub_incoming = TRN::Core::Matrix::create(implementor, incoming, 0, 0, preamble, incoming->get_cols());
	auto sub_expected = TRN::Core::Matrix::create(implementor, expected, 0, 0, preamble, expected->get_cols());

	for (std::size_t batch = 0; batch < handle->batch_size; batch++)
	{
		auto sub_u_ffwd = TRN::Core::Matrix::create(implementor,  preamble, handle->reservoir_size);
		handle->batched_incoming->update(batch, sub_incoming);
		handle->batched_expected->update(batch, sub_expected);
		handle->batched_u_ffwd->update(batch, sub_u_ffwd);
	}

	handle->target_expected = expected;
	handle->cycle = preamble;
	handle->autonomous_generation = autonomous_generation;
	handle->max_cycle = expected->get_rows() + supplementary_generations - 1;
	implementor->get_algorithm()->prime(
			handle->batch_size, handle->mini_batch_size,
			handle->seed,
			handle->stimulus_stride, handle->reservoir_stride, handle->prediction_stride,
			handle->stimulus_size, handle->reservoir_size, handle->prediction_size,
			handle->leak_rate, handle->initial_state_scale,
			sub_expected->get_elements(), sub_expected->get_rows(), sub_expected->get_cols(), sub_expected->get_stride(),

		handle->batched_incoming->get_elements(), handle->batched_incoming->get_rows(), handle->batched_incoming->get_cols(), handle->batched_incoming->get_strides(),
		handle->batched_expected->get_elements(), handle->batched_expected->get_rows(), handle->batched_expected->get_cols(), handle->batched_expected->get_strides(),
		handle->batched_W_ffwd->get_elements(), handle->batched_W_ffwd->get_rows(), handle->batched_W_ffwd->get_cols(), handle->batched_W_ffwd->get_strides(),
		handle->batched_u_ffwd->get_elements(), handle->batched_u_ffwd->get_rows(), handle->batched_u_ffwd->get_cols(), handle->batched_u_ffwd->get_strides(),
		handle->batched_X_res->get_elements(), handle->batched_X_res->get_rows(), handle->batched_X_res->get_cols(), handle->batched_X_res->get_strides(),
		handle->batched_W_rec->get_elements(), handle->batched_W_rec->get_rows(), handle->batched_W_rec->get_cols(), handle->batched_W_rec->get_strides(),
		handle->batched_u->get_elements(), handle->batched_u->get_rows(), handle->batched_u->get_cols(), handle->batched_u->get_strides(),
		handle->batched_p->get_elements(), handle->batched_p->get_rows(), handle->batched_p->get_cols(), handle->batched_p->get_strides(),
		handle->batched_X_ro->get_elements(), handle->batched_X_ro->get_rows(), handle->batched_X_ro->get_cols(), handle->batched_X_ro->get_strides(),
		handle->batched_W_ro->get_elements(), handle->batched_W_ro->get_rows(), handle->batched_W_ro->get_cols(), handle->batched_W_ro->get_strides(),
		handle->bundled_pre->get_elements(true), handle->bundled_pre->get_rows(), handle->bundled_pre->get_cols(), handle->bundled_pre->get_strides(),
		handle->batched_post->get_elements(), handle->batched_post->get_rows(), handle->batched_post->get_cols(), handle->batched_post->get_strides(),
		handle->bundled_desired->get_elements(true), handle->bundled_desired->get_rows(), handle->bundled_desired->get_cols(), handle->bundled_desired->get_strides(),
			sub_scheduling->get_offsets().data(), sub_scheduling->get_durations().data(), sub_scheduling->get_durations().size(), handle->target_expected->get_rows(),
			sub_states->get_elements(), sub_states->get_rows(), sub_states->get_cols(), sub_states->get_stride(), 
		handle->one->get_elements(), handle->zero->get_elements());
	
	TRN::Helper::Observable<TRN::Core::Message::Payload<TRN::Core::Message::PRIMED>>::notify(TRN::Core::Message::Payload<TRN::Core::Message::PRIMED>(evaluation_id));


	handle->prediction.enqueue(std::make_tuple(handle->batched_X_ro, evaluation_id));
}
#include <iostream>
void TRN::Core::Reservoir::update(const TRN::Core::Message::Payload<TRN::Core::Message::STIMULUS> &incoming)
{
	if (handle->cycle < handle->max_cycle)
	{
		auto sub_states = TRN::Core::Matrix::create(implementor, handle->states->get_global(), handle->cycle, 0, 1, handle->states->get_global()->get_cols());

		std::vector<int> durations(1);
		std::vector<int> offsets(1);
		offsets[0] = 0;
		durations[0] = 1;

		auto sub_scheduling = TRN::Core::Scheduling::create(offsets, durations);
		auto sub_expected = TRN::Core::Matrix::create(implementor, handle->target_expected, handle->cycle, 0, 1, handle->target_expected->get_cols());

		
		for (std::size_t batch = 0; batch < handle->batch_size; batch++)
		{
			auto sub_u_ffwd = TRN::Core::Matrix::create(implementor, 1, handle->reservoir_size);
			if (handle->autonomous_generation)
				handle->batched_incoming->update(batch, incoming.get_stimulus()->get_matrices(batch));
			else
				handle->batched_incoming->update(batch, TRN::Core::Matrix::create(implementor, handle->target_expected, handle->cycle, 0, 1, handle->target_expected->get_cols()));
			handle->batched_expected->update(batch, sub_expected);
			handle->batched_u_ffwd->update(batch, sub_u_ffwd);
		}

		implementor->get_algorithm()->generate( // generate
			handle->batch_size, handle->mini_batch_size,
			handle->seed,
			handle->stimulus_stride, handle->reservoir_stride, handle->prediction_stride,
			handle->stimulus_size, handle->reservoir_size, handle->prediction_size,
			handle->leak_rate, handle->initial_state_scale,
			sub_expected->get_elements(), sub_expected->get_rows(), sub_expected->get_cols(), sub_expected->get_stride(),

			handle->batched_incoming->get_elements(), handle->batched_incoming->get_rows(), handle->batched_incoming->get_cols(), handle->batched_incoming->get_strides(),
			handle->batched_expected->get_elements(), handle->batched_expected->get_rows(), handle->batched_expected->get_cols(), handle->batched_expected->get_strides(),
			handle->batched_W_ffwd->get_elements(), handle->batched_W_ffwd->get_rows(), handle->batched_W_ffwd->get_cols(), handle->batched_W_ffwd->get_strides(),
			handle->batched_u_ffwd->get_elements(), handle->batched_u_ffwd->get_rows(), handle->batched_u_ffwd->get_cols(), handle->batched_u_ffwd->get_strides(),
			handle->batched_X_res->get_elements(), handle->batched_X_res->get_rows(), handle->batched_X_res->get_cols(), handle->batched_X_res->get_strides(),
			handle->batched_W_rec->get_elements(), handle->batched_W_rec->get_rows(), handle->batched_W_rec->get_cols(), handle->batched_W_rec->get_strides(),
			handle->batched_u->get_elements(), handle->batched_u->get_rows(), handle->batched_u->get_cols(), handle->batched_u->get_strides(),
			handle->batched_p->get_elements(), handle->batched_p->get_rows(), handle->batched_p->get_cols(), handle->batched_p->get_strides(),
			handle->batched_X_ro->get_elements(), handle->batched_X_ro->get_rows(), handle->batched_X_ro->get_cols(), handle->batched_X_ro->get_strides(),
			handle->batched_W_ro->get_elements(), handle->batched_W_ro->get_rows(), handle->batched_W_ro->get_cols(), handle->batched_W_ro->get_strides(),
			handle->bundled_pre->get_elements(true), handle->bundled_pre->get_rows(), handle->bundled_pre->get_cols(), handle->bundled_pre->get_strides(),
			handle->batched_post->get_elements(), handle->batched_post->get_rows(), handle->batched_post->get_cols(), handle->batched_post->get_strides(),
			handle->bundled_desired->get_elements(true), handle->bundled_desired->get_rows(), handle->bundled_desired->get_cols(), handle->bundled_desired->get_strides(),
			sub_scheduling->get_offsets().data(), sub_scheduling->get_durations().data(), sub_scheduling->get_durations().size(), handle->target_expected->get_rows(),
			sub_states->get_elements(), sub_states->get_rows(), sub_states->get_cols(), sub_states->get_stride(),
			handle->one->get_elements(), handle->zero->get_elements());
		handle->cycle++;
	
		handle->prediction.enqueue(std::make_tuple(handle->batched_X_ro, incoming.get_evaluation_id()));
	
	}
	else
	{
		synchronize();
		handle->target_expected.reset();
		handle->max_cycle = 0;
		handle->cycle = 0;
	
		TRN::Helper::Observable<TRN::Core::Message::Payload<TRN::Core::Message::TESTED>>::notify(TRN::Core::Message::Payload<TRN::Core::Message::TESTED>(incoming.get_evaluation_id()));
	}
}



void TRN::Core::Reservoir::train(const unsigned long long &evaluation_id, const std::shared_ptr<TRN::Core::Matrix> &incoming, const std::shared_ptr<TRN::Core::Matrix> &expected, const std::shared_ptr<TRN::Core::Scheduling> &scheduling)
{
	for (std::size_t batch = 0; batch < handle->batch_size; batch++)
	{
		auto sub_u_ffwd = TRN::Core::Matrix::create(implementor,  incoming->get_rows(), handle->reservoir_size);
		handle->batched_incoming->update(batch, incoming);
		handle->batched_expected->update(batch, expected);
		handle->batched_u_ffwd->update(batch, sub_u_ffwd);
	}

	train(incoming, expected, scheduling, handle->states->get_global());
	synchronize();

	/*{
		std::vector<float> e;
		std::size_t b;
		std::vector<std::size_t> r, c;
		handle->batched_u_ffwd->to(e, b, r, c);

		cv::Mat mat(r[0] * b, c[0], CV_32F, e.data());

		int cnt = 0;
	}*/

	TRN::Helper::Observable<TRN::Core::Message::Payload<TRN::Core::Message::TRAINED>>::notify(TRN::Core::Message::Payload<TRN::Core::Message::TRAINED>(evaluation_id));
}

void TRN::Core::Reservoir::visit(std::shared_ptr<TRN::Core::Message::Payload<TRN::Core::Message::WEIGHTS>> &payload)
{
	payload->set_readout(handle->batched_W_ro);
	payload->set_feedforward(handle->batched_W_ffwd);
	payload->set_recurrent(handle->batched_W_rec);
}
void TRN::Core::Reservoir::visit(std::shared_ptr<TRN::Core::Message::Payload<TRN::Core::Message::STATES>> &payload)
{
	auto duration = payload->get_rows();

	const std::size_t size = handle->batch_size * (handle->stimulus_stride + handle->prediction_stride + handle->prediction_stride + handle->reservoir_stride);
	auto global = TRN::Core::Matrix::create(implementor, duration, size, true);
	std::size_t offset = 0;
	auto stimulus = TRN::Core::Matrix::create(implementor, global, 0, offset, duration, handle->batch_size * handle->stimulus_size); offset += handle->batch_size * handle->stimulus_stride;
	auto reservoir = TRN::Core::Matrix::create(implementor, global, 0, offset, duration, handle->batch_size * handle->reservoir_size); offset += handle->batch_size * handle->reservoir_stride;
	auto prediction = TRN::Core::Matrix::create(implementor, global, 0, offset, duration, handle->batch_size * handle->prediction_size); offset += handle->batch_size * handle->prediction_stride;
	auto desired = TRN::Core::Matrix::create(implementor, global, 0, offset, duration, handle->batch_size * handle->prediction_size); offset += handle->batch_size * handle->prediction_stride;



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
	flops_per_cycle += handle->reservoir_size * handle->reservoir_size * 2; // w_in * x_in
	flops_per_cycle += handle->reservoir_size * 4; // update euler
	flops_per_cycle += handle->reservoir_size  * (3 + 28); // tanh

	flops_per_cycle += handle->prediction_size * handle->reservoir_size * 2; //xro= W_ro * x_res
	flops_per_cycle += handle->prediction_size * (3 + 28); // tanh
	flops_per_cycle += handle->prediction_size * (3); // err
	payload->set_flops_per_cycle(flops_per_cycle);

	size_t flops_per_epoch_factor = 0;
	flops_per_epoch_factor += handle->reservoir_size * handle->stimulus_size * 2; 

	payload->set_flops_per_epoch_factor(flops_per_epoch_factor);
}


