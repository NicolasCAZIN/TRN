#include "stdafx.h"
#include "SpatialFilter.h"
#include "Core/Loop_impl.h"
#include "Core/Bundle.h"
#include "Helper/Logger.h"
#include <iostream>
#include <random>
#include <algorithm>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

class TRN::Loop::SpatialFilter::Handle
{
public :
	std::size_t batch_size;
	std::size_t stimulus_size;
	std::shared_ptr<TRN::Core::Decoder> decoder;
	bool autonomous;


	std::vector<float> trajectory_coordinates;
	std::size_t trajectory_duration;
	std::size_t t;

	std::string tag;



	std::shared_ptr<TRN::Core::Batch> batched_predicted_position;
	std::shared_ptr<TRN::Core::Batch> batched_current_position;
	std::shared_ptr<TRN::Core::Batch> batched_previous_position;

	std::function<void(const unsigned long long &evaluation_id, const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols)> on_predicted_position;
	std::function<void(const unsigned long long &evaluation_id, const std::vector<float> &stimulus, const std::size_t &rows, const std::size_t &cols)> on_predicted_stimulus;

};

/*static void cartesian_to_polar(const int &x, const int &y, float &rho, float &theta)
{
	rho = std::hypotf(x, y);
	theta = std::atan2f(y, x);
}
static void polar_to_cartesian(const float &rho, const float &theta, int &x, int &y)
{
	x = std::ceil(std::cosf(theta) * rho);
	y = std::ceil(std::sinf(theta) * rho);
}

static float compute_heading(const std::vector<float> &current, const std::vector<float> &previous)
{
	float vx = current[0] - previous[0];
	float vy = current[1] - previous[1];

	return std::atan2f(vy, vx);
}*/


TRN::Loop::SpatialFilter::SpatialFilter(const std::shared_ptr<TRN::Backend::Driver> &driver, const std::size_t &batch_size, const std::size_t &stimulus_size,
	const std::function<void(const unsigned long long &evaluation_id, const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols)> &predicted_position,
	std::function<void(const unsigned long long &evaluation_id, const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols)> &estimated_position,
	const std::function<void(const unsigned long long &evaluation_id, const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols)> &predicted_stimulus,
	std::function<void(const unsigned long long &evaluation_id, const std::vector<float> &stimulus, const std::size_t &rows, const std::size_t &cols)> &perceived_stimulus,
	const std::shared_ptr<TRN::Core::Decoder> &decoder,
	const std::string &tag
	) :
	TRN::Core::Loop(driver, batch_size, stimulus_size),
	handle(std::make_unique<Handle>())
{

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
	handle->on_predicted_position = predicted_position;
	handle->on_predicted_stimulus = predicted_stimulus;


	perceived_stimulus = [this](const unsigned long long &evaluation_id, const std::vector<float> &stimulus, const std::size_t &rows, const std::size_t &cols)
	{	
		assert(rows == TRN::Core::Loop::handle->batch_size);
		for (std::size_t batch = 0; batch < TRN::Core::Loop::handle->batch_size; batch++)
		{
			std::vector<float> local(stimulus.begin() + batch * cols, stimulus.begin() + batch * cols + cols);
			assert(local.size() == cols);
			TRN::Core::Loop::handle->stimulus->get_matrices(batch)->from(local, 1, cols);
		}
		TRN::Helper::Observable<TRN::Core::Message::Payload<TRN::Core::Message::STIMULUS>>::notify(TRN::Core::Message::Payload<TRN::Core::Message::STIMULUS>(TRN::Core::Loop::handle->stimulus, evaluation_id));
	};
	estimated_position = [this](const unsigned long long &evaluation_id, const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols)
	{
		assert(rows == TRN::Core::Loop::handle->batch_size);
			
		handle->t++;
		for (std::size_t batch = 0; batch < TRN::Core::Loop::handle->batch_size; batch++)
		{
			float x, y;
			if (handle->autonomous)
			{
				std::vector<float> current(position.begin() + batch * cols, position.begin() + batch * cols + cols);
				assert(current.size() == cols);
				x = current[0];
				y = current[1];
			}
			else
			{
				location_at(handle->t, x, y);
			}
	
			std::size_t previous_rows, previous_cols;
			std::vector<float> previous;
			handle->batched_current_position->get_matrices(batch)->to(previous, previous_rows, previous_cols);
			handle->batched_current_position->get_matrices(batch)->from({x, y}, 1, cols);
	
			handle->batched_previous_position->get_matrices(batch)->from(previous, 1, cols);
		}

	};
}

TRN::Loop::SpatialFilter::~SpatialFilter()
{
	handle.reset();
}

void TRN::Loop::SpatialFilter::location_at(const std::size_t &t, float &x, float &y)
{
	if (t >= handle->trajectory_duration)
		throw std::runtime_error("Invalid time index t = " + std::to_string(t));
	x = handle->trajectory_coordinates[t * 2];
	y = handle->trajectory_coordinates[t * 2 + 1];
}

void TRN::Loop::SpatialFilter::update(const TRN::Core::Message::Payload<TRN::Core::Message::TEST> &payload)
{
	TRN::Core::Loop::update(payload);
	auto trajectory = delegate.lock()->retrieve_sequence(payload.get_label(), handle->tag);
	
	TRN::Helper::Observable<TRN::Core::Message::Payload<TRN::Core::Message::TARGET_TRAJECTORY>>::notify(trajectory);
	std::vector<float> trajectory_coordinates;
	std::size_t trajectory_rows;
	std::size_t trajectory_cols;
	trajectory->to(trajectory_coordinates, trajectory_rows, trajectory_cols);


	if (trajectory_cols != 2)
		throw std::invalid_argument("invalid position in test sequence for label " + payload.get_label() + " and tag " + handle->tag);
	if (payload.get_preamble() < 2)
		throw std::runtime_error("Preamble must be at least 2 time steps");
	handle->autonomous = payload.get_autonomous();
	handle->trajectory_coordinates = trajectory_coordinates;
	handle->trajectory_duration = trajectory_rows;
	auto t = payload.get_preamble() - 1;
	auto t_1 = t - 1;
	handle->t = t;

	float cx, cy;
	location_at(t, cx, cy);
	float px, py;
	location_at(t_1, px, py);
	std::vector<float> current({ cx , cy });
	std::vector<float> previous({ px , py });

	for (std::size_t batch = 0; batch < handle->batch_size; batch++)
	{
		handle->batched_current_position->get_matrices(batch)->from(current, 1, 2);
		handle->batched_previous_position->get_matrices(batch)->from(previous, 1, 2);

	}
}

void TRN::Loop::SpatialFilter::update(const TRN::Core::Message::Payload<TRN::Core::Message::PREDICTION> &payload)
{
	handle->decoder->decode(handle->batched_previous_position, handle->batched_current_position, payload.get_predicted(), handle->batched_predicted_position);

	std::vector<float> predicted_position;
	std::size_t predicted_position_matrices;
	std::vector<std::size_t> predicted_position_rows;
	std::vector<std::size_t> predicted_position_cols;

	handle->batched_predicted_position->to(predicted_position, predicted_position_matrices, predicted_position_rows, predicted_position_cols);


	TRN::Helper::Observable<TRN::Core::Message::Payload<TRN::Core::Message::POSITION>>::notify(TRN::Core::Message::Payload<TRN::Core::Message::POSITION>(handle->batched_predicted_position, payload.get_evaluation_id()));

	handle->on_predicted_position(payload.get_evaluation_id(), predicted_position, predicted_position_matrices, predicted_position_cols[0]);
}

void TRN::Loop::SpatialFilter::visit(std::shared_ptr<TRN::Core::Message::Payload<TRN::Core::Message::FLOPS>> &payload)
{
	handle->decoder->visit(payload);
}

std::shared_ptr<TRN::Loop::SpatialFilter> TRN::Loop::SpatialFilter::create(const std::shared_ptr<TRN::Backend::Driver> &driver, const std::size_t &batch_size, const std::size_t &stimulus_size,
	const std::function<void(const unsigned long long &evaluation_id, const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols)> &predicted_position,
	std::function<void(const unsigned long long &evaluation_id, const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols)> &estimated_position,
	const std::function<void(const unsigned long long &evaluation_id, const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols)> &predicted_stimulus,
	std::function<void(const unsigned long long &evaluation_id, const std::vector<float> &stimulus, const std::size_t &rows, const std::size_t &cols)> &perceived_stimulus,
	const std::shared_ptr<TRN::Core::Decoder> &decoder,
	const std::string &tag)
{
	return std::make_shared<TRN::Loop::SpatialFilter>(driver, batch_size, stimulus_size, predicted_position, estimated_position, predicted_stimulus, perceived_stimulus, decoder,tag);
}