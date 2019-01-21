#include "stdafx.h"
#include "SpatialFilter.h"
#include "Core/Loop_impl.h"
#include "Core/Bundle.h"
#include "Helper/Logger.h"
#include <iostream>
#include <random>
#include <algorithm>
#include <numeric>

class TRN::Loop::SpatialFilter::Handle
{
public :
	std::size_t batch_size;
	std::size_t stimulus_size;
	std::shared_ptr<TRN::Core::Decoder> decoder;
	std::shared_ptr<TRN::Core::Encoder> encoder;
	bool autonomous;


	std::shared_ptr<TRN::Core::Matrix> trajectory;
	std::size_t t;

	std::string tag;

	std::shared_ptr<TRN::Core::Batch> batched_predicted_position;
	std::shared_ptr<TRN::Core::Batch> batched_current_position;
	std::shared_ptr<TRN::Core::Batch> batched_previous_position;
};

TRN::Loop::SpatialFilter::SpatialFilter(const std::shared_ptr<TRN::Backend::Driver> &driver, const std::size_t &batch_size, const std::size_t &stimulus_size,
	const std::shared_ptr<TRN::Core::Encoder> &encoder,
	const std::shared_ptr<TRN::Core::Decoder> &decoder,
	const std::string &tag
	) :
	TRN::Core::Loop(driver, batch_size, stimulus_size),
	handle(std::make_unique<Handle>())
{
	handle->encoder = encoder;
	handle->decoder = decoder;
	handle->stimulus_size = stimulus_size;
	handle->batch_size = batch_size;

	handle->tag = tag;
	

	handle->batched_predicted_position = TRN::Core::Batch::create(driver, batch_size);
	handle->batched_current_position = TRN::Core::Batch::create(driver, batch_size);
	handle->batched_previous_position = TRN::Core::Batch::create(driver, batch_size);
	

	for (std::size_t batch = 0; batch < batch_size; batch++)
	{
		handle->batched_current_position->update(batch, TRN::Core::Matrix::create(driver, 1, 2));
		handle->batched_predicted_position->update(batch, TRN::Core::Matrix::create(driver, 1, 2));
		handle->batched_previous_position->update(batch, TRN::Core::Matrix::create(driver, 1, 2));
	}
}

TRN::Loop::SpatialFilter::~SpatialFilter()
{
	handle.reset();
}

void TRN::Loop::SpatialFilter::location_at(const std::size_t &t, std::shared_ptr<TRN::Core::Matrix> &location)
{
	if (t >= handle->trajectory->get_rows())
		throw std::runtime_error("Invalid time index t = " + std::to_string(t));
	location = TRN::Core::Matrix::create(implementor, handle->trajectory, t, 0, 1, handle->trajectory->get_cols());
}

void TRN::Loop::SpatialFilter::update(const TRN::Core::Message::Payload<TRN::Core::Message::TEST> &payload)
{
	TRN::Core::Loop::update(payload);
	handle->trajectory = delegate.lock()->retrieve_sequence(payload.get_label(), handle->tag);
	
	TRN::Helper::Observable<TRN::Core::Message::Payload<TRN::Core::Message::TARGET_TRAJECTORY>>::notify(handle->trajectory);

	if (payload.get_preamble() < 2)
		throw std::runtime_error("Preamble must be at least 2 time steps");
	handle->autonomous = payload.get_autonomous();
	auto t = payload.get_preamble() - 1;
	auto t_1 = t - 1;
	handle->t = t;

	std::shared_ptr<TRN::Core::Matrix> current, previous;

	location_at(t, current);
	location_at(t_1, previous);

	for (std::size_t batch = 0; batch < handle->batch_size; batch++)
	{
		handle->batched_previous_position->get_matrices(batch)->from(*previous);
		handle->batched_current_position->get_matrices(batch)->from(*current);
	}
}

void TRN::Loop::SpatialFilter::update(const TRN::Core::Message::Payload<TRN::Core::Message::PREDICTION> &payload)
{
	handle->decoder->decode(handle->batched_previous_position, handle->batched_current_position, payload.get_predicted(), handle->batched_predicted_position);
	TRN::Helper::Observable<TRN::Core::Message::Payload<TRN::Core::Message::POSITION>>::notify(TRN::Core::Message::Payload<TRN::Core::Message::POSITION>(handle->batched_predicted_position, payload.get_evaluation_id()));
	handle->encoder->encode(handle->batched_predicted_position, payload.get_evaluation_id(), TRN::Core::Loop::handle->stimulus);
}

void TRN::Loop::SpatialFilter::update(const TRN::Core::Message::Payload<TRN::Core::Message::STIMULUS> &payload)
{
	TRN::Helper::Observable<TRN::Core::Message::Payload<TRN::Core::Message::STIMULUS>>::notify(TRN::Core::Message::Payload<TRN::Core::Message::STIMULUS>(payload.get_stimulus(), payload.get_evaluation_id()));
}
void TRN::Loop::SpatialFilter::update(const TRN::Core::Message::Payload<TRN::Core::Message::POSITION> &payload)
{
	handle->t++;
	if (handle->autonomous)
	{
		for (std::size_t batch = 0; batch < handle->batch_size; batch++)
		{
			auto next = payload.get_position()->get_matrices(batch);
			auto current = handle->batched_current_position->get_matrices(batch);
			auto previous = handle->batched_previous_position->get_matrices(batch);

			previous->from(*current);
			current->from(*next);
		}
	}
	else
	{
		std::shared_ptr<TRN::Core::Matrix> next;

		location_at(handle->t, next);
		for (std::size_t batch = 0; batch < handle->batch_size; batch++)
		{
			auto current = handle->batched_current_position->get_matrices(batch);
			auto previous = handle->batched_previous_position->get_matrices(batch);

			previous->from(*current);
			current->from(*next);
		}
	}
}
void TRN::Loop::SpatialFilter::visit(std::shared_ptr<TRN::Core::Message::Payload<TRN::Core::Message::FLOPS>> &payload)
{
	handle->decoder->visit(payload);
}

std::shared_ptr<TRN::Loop::SpatialFilter> TRN::Loop::SpatialFilter::create(const std::shared_ptr<TRN::Backend::Driver> &driver, const std::size_t &batch_size, const std::size_t &stimulus_size,
	const std::shared_ptr<TRN::Core::Encoder> &encoder,
	const std::shared_ptr<TRN::Core::Decoder> &decoder,
	const std::string &tag)
{
	return std::make_shared<TRN::Loop::SpatialFilter>(driver, batch_size, stimulus_size,encoder, decoder,tag);
}