#pragma once

#include "Kernel.h"

struct TRN::Decoder::Kernel::Handle
{
	std::size_t rows;
	std::size_t cols;
	std::size_t roi_rows;
	std::size_t roi_cols;
	std::pair<float, float> arena_x;
	std::pair<float, float> arena_y;
	std::size_t *roi_row_begin;
	std::size_t *roi_row_end;
	std::size_t *roi_col_begin;
	std::size_t *roi_col_end;
	std::vector<int *>argmax;
	int **host_argmax;
	int **dev_argmax;
	float sigma;
	float radius;
	//float angle;
	float scale;
	float cos_half_angle;
	unsigned long seed;
	std::vector<float> x_range;
	std::vector<float> y_range;
	std::shared_ptr<TRN::Core::Matrix> x_grid;
	std::shared_ptr<TRN::Core::Matrix> y_grid;

	std::shared_ptr<TRN::Core::Batch> batched_next_location_probability;
	std::shared_ptr<TRN::Core::Batch> batched_direction;
	std::shared_ptr<TRN::Core::Batch> batched_x_grid_centered;
	std::shared_ptr<TRN::Core::Batch> batched_y_grid_centered;
};