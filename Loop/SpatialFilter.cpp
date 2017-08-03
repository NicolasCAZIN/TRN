#include "stdafx.h"
#include "SpatialFilter.h"
#include "Core/Loop_impl.h"
#include "Core/Bundle.h"

class TRN::Loop::SpatialFilter::Handle
{
public :
	std::size_t batch_size;
	std::size_t rows;
	std::size_t cols;
	std::size_t stimulus_size;
	float sigma;
	float radius;
	std::vector<float> x_range;
	std::vector<float> y_range;
	std::string tag;

	std::shared_ptr<TRN::Core::Matrix> x_grid;
	std::shared_ptr<TRN::Core::Matrix> y_grid;
	std::shared_ptr<TRN::Core::Batch> batched_x_grid_centered2;
	std::shared_ptr<TRN::Core::Batch> batched_y_grid_centered2;

	std::shared_ptr<TRN::Core::Batch> batched_scale;
	std::shared_ptr<TRN::Core::Batch> batched_predicted_position;
	std::shared_ptr<TRN::Core::Batch> batched_current_position;

	std::shared_ptr<TRN::Core::Batch> batched_firing_rate_map;
	std::shared_ptr<TRN::Core::Batch> batched_next_location_probability;
	std::shared_ptr<TRN::Core::Batch> batched_motion_probability;
	std::shared_ptr<TRN::Core::Batch> batched_transition_probability;

	std::shared_ptr<TRN::Core::Bundle> bundled_hypothesis_map;
	
	

	std::function<void(const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols)> on_predicted_position;
	std::function<void(const std::vector<float> &stimulus, const std::size_t &rows, const std::size_t &cols)> on_predicted_stimulus;
	std::function<std::size_t(const float &x)> x_to_col;
	std::function<std::size_t(const float &y)> y_to_row;

};

TRN::Loop::SpatialFilter::SpatialFilter(const std::shared_ptr<TRN::Backend::Driver> &driver, const std::size_t &batch_size, const std::size_t &stimulus_size,
	const std::function<void(const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols)> &predicted_position,
	std::function<void(const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols)> &estimated_position,
	const std::function<void(const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols)> &predicted_stimulus,
	std::function<void(const std::vector<float> &stimulus, const std::size_t &rows, const std::size_t &cols)> &perceived_stimulus,
	const std::size_t &rows, const std::size_t &cols,
	const std::pair<float, float> &x, const std::pair<float, float> &y,
	const std::vector<float> &response,
	const float &sigma,
	const float &radius,
	const std::string &tag
	) :
	TRN::Core::Loop(driver, batch_size, stimulus_size),
	handle(std::make_unique<Handle>())
{
	handle->stimulus_size = stimulus_size;
	handle->batch_size = batch_size;
	handle->radius = radius;
	handle->sigma = sigma;
	handle->tag = tag;
	handle->rows = rows;
	handle->cols = cols;
	


	if (rows == 0)
		throw std::invalid_argument("grid rows must be strictly positive");
	if (cols == 0)
		throw std::invalid_argument("grid cols must be strictly positive");

	if (rows * stimulus_size * cols != response.size())
		throw std::invalid_argument("place cell response table must fit dimensions rows * cols * stimulus_size");




	if (x.first > x.second)
		throw std::invalid_argument("grid x_min > grid x_max");
	if (y.first > y.second)
		throw std::invalid_argument("grid y_min > grid y_max");
	handle->on_predicted_position = predicted_position;
	handle->on_predicted_stimulus = predicted_stimulus;
	const float x_step = (x.second - x.first) / (cols - 1);
	const float y_step = (y.second - y.first) / (rows - 1);
	handle->x_range.resize(cols);
	handle->y_range.resize(rows);
	handle->x_grid = TRN::Core::Matrix::create(driver, 1, cols);
	handle->y_grid = TRN::Core::Matrix::create(driver, 1, rows);
	handle->batched_next_location_probability = TRN::Core::Batch::create(driver, batch_size);
	handle->batched_motion_probability = TRN::Core::Batch::create(driver, batch_size);
	handle->batched_transition_probability = TRN::Core::Batch::create(driver, batch_size);
	handle->bundled_hypothesis_map = TRN::Core::Bundle::create(driver, batch_size);

	handle->batched_x_grid_centered2 = TRN::Core::Batch::create(driver, batch_size);
	handle->batched_y_grid_centered2 = TRN::Core::Batch::create(driver, batch_size);
	handle->batched_scale = TRN::Core::Batch::create(driver, batch_size);
	handle->batched_predicted_position = TRN::Core::Batch::create(driver, batch_size);
	handle->batched_current_position = TRN::Core::Batch::create(driver, batch_size);

	perceived_stimulus = [this](const std::vector<float> &stimulus, const std::size_t &rows, const std::size_t &cols)
	{	
		assert(rows == TRN::Core::Loop::handle->batch_size);
		for (std::size_t batch = 0; batch < TRN::Core::Loop::handle->batch_size; batch++)
		{
			std::vector<float> local(stimulus.begin() + batch * cols, stimulus.begin() + batch * cols + cols - 1);
			assert(local.size() == cols);
			TRN::Core::Loop::handle->stimulus->get_matrices(batch)->from(local, 1, cols);
		}
		TRN::Helper::Observable<TRN::Core::Message::Payload<TRN::Core::Message::STIMULUS>>::notify(TRN::Core::Message::Payload<TRN::Core::Message::STIMULUS>(TRN::Core::Loop::handle->stimulus));
	};
	estimated_position = [this](const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols)
	{
		assert(rows == TRN::Core::Loop::handle->batch_size);
		for (std::size_t batch = 0; batch < TRN::Core::Loop::handle->batch_size; batch++)
		{
			std::vector<float> local(position.begin() + batch * cols, position.begin() + batch * cols + cols - 1);
			assert(local.size() == cols);
			handle->batched_current_position->get_matrices(batch)->from(local, 1, cols);
		}
	};

#pragma omp parallel sections
	{
#pragma omp section
		{
			handle->x_range[0] = x.first;
			for (std::size_t col = 0; col < cols - 1; col++)
				handle->x_range[col] = x.first + col * x_step;
			handle->x_range[cols - 1] = x.second;
			handle->x_grid->from(handle->x_range, 1, handle->x_range.size());
		}
#pragma omp section
		{
			handle->y_range[rows - 1] = y.first;
			for (std::size_t row = 0; row < rows - 1; row++)
				handle->y_range[rows - 1 - row] = y.first + row * y_step;
			handle->y_range[0] = y.second;
			handle->y_grid->from(handle->y_range, 1, handle->y_range.size());
		}
	}
	handle->batched_firing_rate_map = TRN::Core::Batch::create(driver, stimulus_size);

	for (std::size_t batch = 0; batch < batch_size; batch++)
	{

		auto batched_hypothesis_map = TRN::Core::Batch::create(driver, stimulus_size);
		for (std::size_t place_cell = 0; place_cell < stimulus_size; place_cell++)
		{
			auto b = &response[place_cell * rows * cols];
			auto e = b + rows * cols;
			std::vector<float> place_cell_response(b, e);
			auto firing_rate_map = TRN::Core::Matrix::create(driver, place_cell_response, rows, cols);
			handle->batched_firing_rate_map->update(place_cell, firing_rate_map);

			auto hypothesis_map = TRN::Core::Matrix::create(driver, rows, cols, true);
			batched_hypothesis_map->update(place_cell, hypothesis_map);
		}
		handle->bundled_hypothesis_map->update(batch, batched_hypothesis_map);
		auto scale = TRN::Core::Matrix::create(implementor, 1, stimulus_size);
		auto x_grid_centered2 = TRN::Core::Matrix::create(driver, 1, cols);
		auto y_grid_centered2 = TRN::Core::Matrix::create(driver, 1, rows);
		auto predicted_position = TRN::Core::Matrix::create(driver, 1, 2);
		auto current_position = TRN::Core::Matrix::create(driver, 1, 2);
		auto next_location_probability = TRN::Core::Matrix::create(implementor, rows, cols, true);
		handle->batched_scale->update(batch, scale);
		handle->batched_x_grid_centered2->update(batch, x_grid_centered2);
		handle->batched_y_grid_centered2->update(batch, y_grid_centered2);
		handle->batched_predicted_position->update(batch, predicted_position);
		handle->batched_current_position->update(batch, current_position);
		handle->batched_next_location_probability->update(batch, next_location_probability);
	}
}

TRN::Loop::SpatialFilter::~SpatialFilter()
{
	handle.reset();
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
	auto t = payload.get_preamble();

	for (std::size_t batch = 0; batch < handle->batch_size; batch++)
	{
		std::vector<float> current({ trajectory_coordinates[t * 2] , trajectory_coordinates[t * 2 + 1] });

		handle->batched_current_position->get_matrices(batch)->from(current, 1, 2);
	};
}
static inline std::size_t clamp(const std::size_t &v, const std::size_t &a, const std::size_t &b)
{
	if (v <= a)
		return a;
	else if (v >= b)
		return b;
	else 
		return v;
}
#include <iostream>
#include <random>
#include <algorithm>
#include <numeric>
void TRN::Loop::SpatialFilter::update(const TRN::Core::Message::Payload<TRN::Core::Message::PREDICTION> &payload)
{
	{
		std::vector<std::size_t>  rows, cols;
		std::size_t matrices;
		std::vector<float> predicted_stimulus;

		payload.get_predicted()->to(predicted_stimulus, matrices, rows, cols);
		assert(std::all_of(rows.begin(), rows.end(), [](const std::size_t &r) { return r == 1; }));
		assert(std::all_of(rows.begin(), rows.end(), [cols](const std::size_t &c) { return c == cols[0]; }));
		//cv::Mat predicted(rows, cols, CV_32F, predicted_stimulus.data());
		handle->on_predicted_stimulus(predicted_stimulus, matrices, cols[0]);
	}

	implementor->get_algorithm()->place_cell_location_probability(
		handle->batch_size, handle->stimulus_size,
		handle->rows, handle->cols,
		handle->sigma, handle->radius,
		(const float **)handle->batched_firing_rate_map->get_elements(), handle->batched_firing_rate_map->get_rows(), handle->batched_firing_rate_map->get_cols(), handle->batched_firing_rate_map->get_strides(),
		handle->x_grid->get_elements(), handle->x_grid->get_rows(), handle->x_grid->get_cols(), handle->x_grid->get_stride(),
		handle->y_grid->get_elements(), handle->y_grid->get_rows(), handle->y_grid->get_cols(), handle->y_grid->get_stride(),
		handle->batched_x_grid_centered2->get_elements(), handle->batched_x_grid_centered2->get_rows(), handle->batched_x_grid_centered2->get_cols(), handle->batched_x_grid_centered2->get_strides(),
		handle->batched_y_grid_centered2->get_elements(), handle->batched_y_grid_centered2->get_rows(), handle->batched_y_grid_centered2->get_cols(), handle->batched_y_grid_centered2->get_strides(),
		handle->batched_scale->get_elements(), handle->batched_scale->get_rows(), handle->batched_scale->get_cols(), handle->batched_scale->get_strides(),
		(const float **)payload.get_predicted()->get_elements(), payload.get_predicted()->get_rows(), payload.get_predicted()->get_cols(), payload.get_predicted()->get_strides(),
		(float ***)handle->bundled_hypothesis_map->get_elements(), handle->bundled_hypothesis_map->get_rows(), handle->bundled_hypothesis_map->get_cols(), handle->bundled_hypothesis_map->get_strides(),
		handle->batched_next_location_probability->get_elements(), handle->batched_next_location_probability->get_rows(), handle->batched_next_location_probability->get_cols(), handle->batched_next_location_probability->get_strides(),
		handle->batched_predicted_position->get_elements(), handle->batched_predicted_position->get_rows(), handle->batched_predicted_position->get_cols(), handle->batched_predicted_position->get_strides()
	);

	
	std::vector<float> current_position;
	std::size_t current_position_matrices;
	std::vector<std::size_t> current_position_rows;
	std::vector<std::size_t> current_position_cols;

	handle->batched_current_position->to(current_position, current_position_matrices, current_position_rows, current_position_cols);
	std::vector<float> predicted_position(current_position.size());
	std::size_t predicted_position_matrices = current_position_matrices;
	std::vector<std::size_t> predicted_position_rows(current_position_rows);
	std::vector<std::size_t> predicted_position_cols(current_position_cols);

	{
		std::vector<std::size_t>  _rows, _cols;
		std::size_t _matrices;
		std::vector<float> _elements;
		handle->batched_next_location_probability->to(_elements, _matrices, _rows, _cols);

		float *ptr = _elements.data();


		for (std::size_t matrix = 0; matrix < _matrices; matrix++)
		{
			auto rows = _rows[matrix];
			auto cols = _cols[matrix];
			std::vector<float> elements(ptr, ptr + rows * cols);
#if 1
			auto idx = std::distance(elements.begin(), std::max_element(elements.begin(), elements.end()));
			auto row = idx / cols;
			auto col = idx % cols;

#else 
			//std::vector<float> model(rows * cols);
			//std::vector<float> approximation(rows * cols);
			std::vector<float> elements_restricted(rows * cols);
			std::vector<float> cumulated_grid(rows * cols);
			std::vector<float> cumulated_row(rows);
			std::vector<float> last_column(rows);
			std::vector<std::size_t> idx(rows * cols);
			auto x = current_position[matrix * current_position_cols + 0];
			auto y = current_position[matrix * current_position_cols + 1];
			for (std::size_t k = 0; k < rows * cols; k++)
				idx[k] = k;

			//std::fill(approximation.begin(), approximation.end(), 0.0f);
			std::transform(idx.begin(), idx.end(), elements_restricted.begin(), [=](const std::size_t &k)
			{
				auto row = k / cols;
				auto col = k % cols;
				auto dx = handle->x_range[col] - x;
				auto dy = handle->y_range[row] - y;

				if (dx*dx + dy*dy < handle->radius * handle->radius)
					return elements[k];
				else
					return 0.0f;
			});
			/*cv::Mat cv_elements(rows, cols, CV_32F, elements.data());
			cv::Mat cv_elements_restricted(rows, cols, CV_32F, elements_restricted.data());*/
			float sigma = 0.3;
			auto sigma2 = sigma * sigma;
			float cx = 0.1f;
			float cy = 0.2f;
			for (std::size_t row = 0; row < rows; row++)
			{
				/*for (std::size_t col = 0; col < cols; col++)
				{
					auto dx = handle->x_range[col] - cx;
					auto dy = handle->y_range[row] - cy;

					model[row * cols + col] = expf(-(dx*dx + dy*dy) / (sigma * sigma));
				}*/
				auto ib = elements_restricted.begin() + row * cols;
				auto ie = ib + cols;
				auto ob = cumulated_grid.begin() + row * cols;
				auto oe = ob + cols;
				std::partial_sum(ib, ie, ob);
				last_column[row] = *(oe - 1);
			}

			std::partial_sum(last_column.begin(), last_column.end(), cumulated_row.begin());
			

			auto cr_b = cumulated_row.begin();
			auto cr_e = cumulated_row.end();
			auto cr_m = *cumulated_row.begin();
			auto cr_M = *(cumulated_row.end() - 1);
			std::default_random_engine generator;
			std::uniform_real_distribution<float> row_distribution(cr_m, cr_M);

			auto ry = row_distribution(generator);
			auto row = std::distance(cr_b, std::lower_bound(cr_b, cr_e, ry));

			auto cg_b = cumulated_grid.begin() + row * cols;
			auto cg_e = cg_b + cols;
			auto cg_m = *cg_b;
			auto cg_M = *(cg_e - 1);
			std::uniform_real_distribution<float> col_distribution(cg_m, cg_M);
			auto rx = col_distribution(generator);
			auto col = std::distance(cg_b, std::lower_bound(cg_b, cg_e, rx));
		
			/*for (int k = 0; k < 1000000; k++)
			{

				auto ry = row_distribution(generator);
				auto row = std::distance(cr_b, std::lower_bound(cr_b, cr_e, ry));

				auto cg_b = cumulated_grid.begin() + row * cols;
				auto cg_e = cg_b + cols;
				auto cg_m = *cg_b;
				auto cg_M = *(cg_e - 1);
				std::uniform_real_distribution<float> col_distribution(cg_m, cg_M);
				auto rx = col_distribution(generator);
				auto col = std::distance(cg_b, std::lower_bound(cg_b, cg_e, rx));
				
				approximation[row * cols + col]++;
				int c = 0;

			}*/


		
			/*cv::Mat cv_approximation(rows, cols, CV_32F, approximation.data());
			cv::Mat cv_cumulated_grid(rows, cols, CV_32F, cumulated_grid.data());
			cv::Mat cv_cumulated_row(rows, 1, CV_32F, cumulated_row.data());*/
			/*
			std::vector<float> cumulated(elements.size());
			std::vector<float> sorted(elements.size());
			std::vector<float> test(elements.size());
			// initialize original index locations
			std::fill(test.begin(), test.end(), 0.0f);
			std::vector<size_t> idx(elements.size());
			std::vector<size_t> inv_idx(elements.size());
			for (std::size_t k = 0; k < idx.size(); k++)
			{
				idx[k] = k;
			}

			// sort indexes based on comparing values in v
			std::sort(idx.begin(), idx.end(),
				[&elements](size_t i1, size_t i2) {return elements[i1] > elements[i2]; });
			std::transform(idx.begin(), idx.end(), sorted.begin(), [elements](const std::size_t &k)
			{
				return elements[k];
			});
			for (std::size_t k = 0; k < idx.size(); k++)
			{
				inv_idx[idx[k]] = k;
			}
	
			std::partial_sum(sorted.begin(), sorted.end(), cumulated.begin());


			std::default_random_engine generator;
			std::uniform_real_distribution<double> distribution(cumulated[0], cumulated[cumulated.size()- 1]);
			for (int k = 0; k < 1000000; k++)
			{
				auto v = distribution(generator);

				auto selected = std::distance(cumulated.begin(), std::lower_bound(cumulated.begin(), cumulated.end(), v));
				if (selected >= inv_idx.size())
					selected = inv_idx.size() - 1;
				auto designated = inv_idx[selected];
				test[selected]++;
			}
			cv::Mat cv_elements(rows, cols, CV_32F, elements.data());
			cv::Mat cv_test(rows, cols, CV_32F, test.data());


		*/
#endif
			predicted_position[matrix * predicted_position_cols[matrix] + 0] = handle->x_range[col];
			predicted_position[matrix * predicted_position_cols[matrix] + 1] = handle->y_range[row];

			ptr += rows * cols;
			
		}
	}

	/*{
		std::vector<std::vector<std::size_t>>  _rows, _cols;
		std::vector<std::size_t> _matrices;
		std::size_t _batches;
		std::vector<float> _elements;
		handle->bundled_hypothesis_map->to(_elements, _batches, _matrices, _rows, _cols);

		float *ptr = _elements.data();

		for (std::size_t batch = 0; batch < _batches; batch++)
		{
			for (std::size_t matrix = 0; matrix < _matrices[batch]; matrix++)
			{
				cv::Mat mat(_rows[batch][matrix], _cols[batch][matrix], CV_32F, ptr);

				ptr += mat.rows * mat.cols;
			}
		}
	}*/

	auto pb = predicted_position.begin();
	for (std::size_t batch = 0; batch < predicted_position_matrices; batch++)
	{
		auto pe = pb + predicted_position_rows[batch] * predicted_position_cols[batch];
		
		std::vector<float> local(pe, pb);
		handle->batched_predicted_position->get_matrices(batch)->from(predicted_position,  predicted_position_rows[batch], predicted_position_cols[batch]);
		pb = pe + 1;
	}

	TRN::Helper::Observable<TRN::Core::Message::Payload<TRN::Core::Message::POSITION>>::notify(TRN::Core::Message::Payload<TRN::Core::Message::POSITION>(handle->batched_predicted_position));
	//std::cout << __FUNCTION__ << " predicted position " << handle->x_range[col] << ", " << handle->y_range[handle->y_range.size() - 1 - row] << std::endl;

	handle->on_predicted_position(predicted_position, predicted_position_matrices, predicted_position_cols[0]);
}

void TRN::Loop::SpatialFilter::visit(std::shared_ptr<TRN::Core::Message::Payload<TRN::Core::Message::FLOPS>> &payload)
{
	payload->set_flops_per_epoch_factor(0);

	auto rows = handle->rows;
	auto cols = handle->cols;
	auto pc = handle->stimulus_size;
	auto pc_r = pc - (pc / 2) * 2;
	size_t flops_per_cycle = 0;
	flops_per_cycle += cols * (1 + 1);
	flops_per_cycle += rows * (1 + 1);


	flops_per_cycle += pc * (rows * (cols * (1 + 1 + 1 + 50 + 1) + 1 /*hadd*/ + 1 /*sum +=*/) + 1 + 10);

	pc /= 2;
	flops_per_cycle += pc * rows * cols * 3;

	if (pc >= 2)
	{
		while (pc > 1)
		{
			flops_per_cycle += pc * rows * cols * 1;
			pc /= 2;
		}
		flops_per_cycle += pc * rows * cols * 1;
	}
	if (pc_r > 0)
	{
		flops_per_cycle += pc * rows * cols * 2;
	}

	flops_per_cycle += rows * cols; // O(n) for max

	payload->set_flops_per_cycle(flops_per_cycle);
}

std::shared_ptr<TRN::Loop::SpatialFilter> TRN::Loop::SpatialFilter::create(const std::shared_ptr<TRN::Backend::Driver> &driver, const std::size_t &batch_size, const std::size_t &stimulus_size,
	const std::function<void(const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols)> &predicted_position,
	std::function<void(const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols)> &estimated_position,
	const std::function<void(const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols)> &predicted_stimulus,
	std::function<void(const std::vector<float> &stimulus, const std::size_t &rows, const std::size_t &cols)> &perceived_stimulus,
	const std::size_t &rows, const std::size_t &cols,
	const std::pair<float, float> &x, const std::pair<float, float> &y,
	const std::vector<float> &response,
	const float &sigma,
	const float &radius,
	const std::string &tag)
{
	return std::make_shared<TRN::Loop::SpatialFilter>(driver, batch_size, stimulus_size, predicted_position, estimated_position, predicted_stimulus, perceived_stimulus, rows, cols, x, y, response, sigma, radius,tag);
}