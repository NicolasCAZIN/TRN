#include "stdafx.h"
#include "Kernel_impl.h"
#include "Core/Decoder_impl.h"
#include "Helper/Logger.h"

TRN::Decoder::Kernel::Kernel(const std::shared_ptr<TRN::Backend::Driver> &driver,
	const std::size_t &batch_size, const std::size_t &stimulus_size,
	const std::size_t &rows, const std::size_t &cols,
	const std::pair<float, float> &arena_x, const std::pair<float, float> &arena_y,
	const float &sigma, const float &radius, const float &angle, const float &scale, const unsigned long &seed) :
	TRN::Core::Decoder(driver, batch_size, stimulus_size),
	handle(std::make_unique<Handle>())
{
	handle->seed = seed;
	handle->rows = rows;
	handle->cols = cols;
	handle->arena_x = arena_x;
	handle->arena_y = arena_y;
	handle->sigma = sigma;
	handle->radius = radius;

	handle->scale = scale;

	float corrected_angle = angle;
	if (angle > 360.0f)
	{
		while (corrected_angle > 360.0f)
			corrected_angle -= 360.0f;
		WARNING_LOGGER << "Field of view (" << angle << "°) exceeds 360°. " << corrected_angle << "° will be used instead";
	}
	else if (angle < 0.0f)
	{
		while (corrected_angle < 0.0f)
			corrected_angle += 360.0f;
		WARNING_LOGGER << "Field of view (" << angle << "°) is negative. " << corrected_angle << "° will be used instead";
	}
	handle->cos_half_angle = std::cosf((M_PI * corrected_angle / 180.0f) / 2);
	if (rows == 0)
		throw std::invalid_argument("grid rows must be strictly positive");
	if (cols == 0)
		throw std::invalid_argument("grid cols must be strictly positive");

	if (arena_x.first > arena_x.second)
		throw std::invalid_argument("grid x_min > grid x_max");
	if (arena_y.first > arena_y.second)
		throw std::invalid_argument("grid y_min > grid y_max");

	const float x_step = (arena_x.second - arena_x.first) / (cols - 1);
	const float y_step = (arena_y.second - arena_y.first) / (rows - 1);
	handle->x_range.resize(cols);
	handle->y_range.resize(rows);
	handle->x_grid = TRN::Core::Matrix::create(driver, 1, cols);
	handle->y_grid = TRN::Core::Matrix::create(driver, 1, rows);
	handle->batched_next_location_probability = TRN::Core::Batch::create(driver, batch_size);
	handle->batched_direction = TRN::Core::Batch::create(driver, batch_size);
	handle->batched_x_grid_centered = TRN::Core::Batch::create(driver, batch_size);
	handle->batched_y_grid_centered = TRN::Core::Batch::create(driver, batch_size);


	driver->get_memory()->allocate((void **)&handle->roi_row_begin, sizeof(std::size_t), batch_size);
	driver->get_memory()->allocate((void **)&handle->roi_row_end, sizeof(std::size_t), batch_size);
	driver->get_memory()->allocate((void **)&handle->roi_col_begin, sizeof(std::size_t), batch_size);
	driver->get_memory()->allocate((void **)&handle->roi_col_end, sizeof(std::size_t), batch_size);

	handle->argmax.resize(batch_size);


	handle->x_range[0] = arena_x.first;
	for (std::size_t col = 0; col < cols - 1; col++)
		handle->x_range[col] = arena_x.first + col * x_step;
	handle->x_range[cols - 1] = arena_x.second;
	handle->x_grid->from(handle->x_range, 1, handle->x_range.size());

	handle->y_range[0] = arena_y.first;
	for (std::size_t row = 0; row < rows - 1; row++)
		handle->y_range[row] = arena_y.first + (row)* y_step;
	handle->y_range[rows - 1] = arena_y.second;
	handle->y_grid->from(handle->y_range, 1, handle->y_range.size());

	if (radius > 0.0f)
	{
		auto roi_cols = std::min(cols, static_cast<std::size_t>(std::floor((radius * 2.0f) / x_step) + 1));
		auto roi_rows = std::min(rows, static_cast<std::size_t>(std::floor((radius * 2.0f) / y_step) + 1));
		std::size_t roi_cols_size;
		std::size_t roi_rows_size;

		std::size_t block_size;
		driver->get_memory()->align(1 * sizeof(float), block_size);

		driver->get_memory()->align((roi_cols) * sizeof(float) + block_size, roi_cols_size);
		driver->get_memory()->align((roi_rows) * sizeof(float), roi_rows_size);
		handle->roi_rows = roi_rows_size / sizeof(float);
		handle->roi_cols = roi_cols_size / sizeof(float);
	}
	else
	{
		handle->roi_rows = rows;
		handle->roi_cols = cols;
	}

	driver->get_memory()->allocate((void **)&handle->dev_argmax, sizeof(int *), batch_size);
	for (std::size_t batch = 0; batch < batch_size; batch++)
	{
		int *ptr;
		driver->get_memory()->allocate((void **)&ptr, sizeof(int), 1);
		handle->argmax[batch] = ptr;
		

		auto x_grid_centered = TRN::Core::Matrix::create(implementor, 1, handle->roi_cols);
		auto y_grid_centered = TRN::Core::Matrix::create(implementor, 1, handle->roi_rows);
		auto next_location_probability = TRN::Core::Matrix::create(implementor, handle->roi_rows, handle->roi_cols);
		auto direction = TRN::Core::Matrix::create(implementor, 1, 2);

		handle->batched_x_grid_centered->update(batch, x_grid_centered);
		handle->batched_y_grid_centered->update(batch, y_grid_centered);
		handle->batched_next_location_probability->update(batch, next_location_probability);
		handle->batched_direction->update(batch, direction);
	}
	handle->host_argmax = handle->argmax.data();
	driver->get_memory()->upload(handle->host_argmax, handle->dev_argmax, sizeof(int *), batch_size);
}

TRN::Decoder::Kernel::~Kernel()
{
	for (auto ptr : handle->argmax)
		implementor->get_memory()->deallocate(ptr);
	implementor->get_memory()->deallocate(handle->dev_argmax);
	handle.reset();
}

void TRN::Decoder::Kernel::decode(
	const std::shared_ptr<TRN::Core::Batch> &previous_position,
	const std::shared_ptr<TRN::Core::Batch> &current_position,
	const std::shared_ptr<TRN::Core::Batch> &predicted_activations,
	std::shared_ptr<TRN::Core::Batch> &decoded_position)
{
	implementor->get_algorithm()->compute_roi(TRN::Core::Decoder::handle->batch_size,
		TRN::Decoder::Kernel::handle->rows,
		TRN::Decoder::Kernel::handle->cols,
		TRN::Decoder::Kernel::handle->arena_x.first, TRN::Decoder::Kernel::handle->arena_x.second,
		TRN::Decoder::Kernel::handle->arena_y.first, TRN::Decoder::Kernel::handle->arena_y.second,
		TRN::Decoder::Kernel::handle->radius,
		(const float **)current_position->get_elements(), current_position->get_strides(),
		TRN::Decoder::Kernel::handle->roi_row_begin, TRN::Decoder::Kernel::handle->roi_row_end,
		TRN::Decoder::Kernel::handle->roi_col_begin, TRN::Decoder::Kernel::handle->roi_col_end
	);

	location_probability(previous_position, current_position, predicted_activations, handle->batched_next_location_probability);
	TRN::Decoder::Kernel::handle->seed += TRN::Decoder::Kernel::handle->roi_rows * TRN::Decoder::Kernel::handle->roi_cols;
	/*std::vector<float> e;
	std::size_t b;
	std::vector<std::size_t> r, c;

	handle->batched_next_location_probability->to(e, b, r, c);
	cv::Mat m(r[0] * b, c[0], CV_32F, e.data());*/
	
	
	implementor->get_algorithm()->select_most_probable_location(
		TRN::Core::Decoder::handle->batch_size, handle->rows, handle->cols,
		handle->roi_row_begin, handle->roi_row_end, handle->roi_col_begin, handle->roi_col_end,
		handle->x_grid->get_elements(), handle->x_grid->get_rows(), handle->x_grid->get_cols(), handle->x_grid->get_stride(),
		handle->y_grid->get_elements(), handle->x_grid->get_rows(), handle->x_grid->get_cols(), handle->x_grid->get_stride(),
		(const float **)handle->batched_next_location_probability->get_elements(true), handle->batched_next_location_probability->get_rows(), handle->batched_next_location_probability->get_cols(), handle->batched_next_location_probability->get_strides(),
		handle->host_argmax
	);

	implementor->get_algorithm()->assign_most_probable_location
	(
		TRN::Core::Decoder::handle->batch_size, handle->rows, handle->cols,
		handle->roi_row_begin, handle->roi_row_end, handle->roi_col_begin, handle->roi_col_end,
		handle->arena_x.first, handle->arena_x.second - handle->arena_x.first,
		handle->arena_y.first, handle->arena_y.second - handle->arena_y.first,
		(const int **)handle->dev_argmax,
		handle->batched_next_location_probability->get_strides(),
		decoded_position->get_elements()
	);

}


