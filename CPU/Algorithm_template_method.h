#pragma once

#include "Implementation.h"
#include "Algorithm.h"
#include "Memory.h"
#include "Random.h"
#include <ctime>
#include <vector>




template <typename T>
static inline T tanh_ps(const T &x)
{
	static const auto minus_two = set1_ps(-2.0f);
	static const auto one = set1_ps(1.0f);
	auto e = exp_ps(mul_ps(x, minus_two));
	return div_ps(sub_ps(one, e), add_ps(one, e));
}

static inline void tanh_v(const std::size_t &cols, const float *x, float *y)
{
	//vsTanh(cols, x, y);
	std::size_t col = 0;

	if (cols - col >= _8)
	{
		for (; col + _8 - 1 < cols; col += _8)
		{
			stream_ps(&y[col + _0], tanh_ps(load_ps(&x[col + _0])));
			stream_ps(&y[col + _1], tanh_ps(load_ps(&x[col + _1])));
			stream_ps(&y[col + _2], tanh_ps(load_ps(&x[col + _2])));
			stream_ps(&y[col + _3], tanh_ps(load_ps(&x[col + _3])));
			stream_ps(&y[col + _4], tanh_ps(load_ps(&x[col + _4])));
			stream_ps(&y[col + _5], tanh_ps(load_ps(&x[col + _5])));
			stream_ps(&y[col + _6], tanh_ps(load_ps(&x[col + _6])));
			stream_ps(&y[col + _7], tanh_ps(load_ps(&x[col + _7])));
		}

	}
	if (cols - col >= _4)
	{
		for (; col + _4 - 1 < cols; col += _4)
		{
			stream_ps(&y[col + _0], tanh_ps(load_ps(&x[col + _0])));
			stream_ps(&y[col + _1], tanh_ps(load_ps(&x[col + _1])));
			stream_ps(&y[col + _2], tanh_ps(load_ps(&x[col + _2])));
			stream_ps(&y[col + _3], tanh_ps(load_ps(&x[col + _3])));
		}
	}
	if (cols - col >= _2)
	{
		for (; col + _2 - 1 < cols; col += _2)
		{
			stream_ps(&y[col + _0], tanh_ps(load_ps(&x[col + _0])));
			stream_ps(&y[col + _1], tanh_ps(load_ps(&x[col + _1])));
		}
	}
	if (cols - col >= 0)
	{
		for (; col < cols; col += _1)
		{
			stream_ps(&y[col + _0], tanh_ps(load_ps(&x[col + _0])));
		}
	}
}
static inline void exp_v(const std::size_t &cols, const float *x, float *y)
{
	//vsExp(cols, x, y);
	std::size_t col = 0;

	if (cols - col >= _8)
	{
		for (; col + _8 - 1 < cols; col += _8)
		{
			stream_ps(&y[col + _0], exp_ps(load_ps(&x[col + _0])));
			stream_ps(&y[col + _1], exp_ps(load_ps(&x[col + _1])));
			stream_ps(&y[col + _2], exp_ps(load_ps(&x[col + _2])));
			stream_ps(&y[col + _3], exp_ps(load_ps(&x[col + _3])));
			stream_ps(&y[col + _4], exp_ps(load_ps(&x[col + _4])));
			stream_ps(&y[col + _5], exp_ps(load_ps(&x[col + _5])));
			stream_ps(&y[col + _6], exp_ps(load_ps(&x[col + _6])));
			stream_ps(&y[col + _7], exp_ps(load_ps(&x[col + _7])));
		}

	}
	if (cols - col >= _4)
	{
		for (; col + _4 - 1 < cols; col += _4)
		{
			stream_ps(&y[col + _0], exp_ps(load_ps(&x[col + _0])));
			stream_ps(&y[col + _1], exp_ps(load_ps(&x[col + _1])));
			stream_ps(&y[col + _2], exp_ps(load_ps(&x[col + _2])));
			stream_ps(&y[col + _3], exp_ps(load_ps(&x[col + _3])));
		}
	}
	if (cols - col >= _2)
	{
		for (; col + _2 - 1 < cols; col += _2)
		{
			stream_ps(&y[col + _0], exp_ps(load_ps(&x[col + _0])));
			stream_ps(&y[col + _1], exp_ps(load_ps(&x[col + _1])));
		}
	}
	if (cols - col >= 0)
	{
		for (; col < cols; col += _1)
		{
			stream_ps(&y[col + _0], exp_ps(load_ps(&x[col + _0])));
		}
	}

}

struct Map
{
	Map(const float *data, const std::size_t &stride) : data(data), stride(stride)
	{

	}
	const float *data;
	const std::size_t stride;
};

struct Model
{
	Model(const float *cx, const float *cy, const float *width, float **gx2w, float **gy2w, const std::size_t *gx2w_strides, const std::size_t *gy2w_strides) :
		cx(cx), cy(cy), width(width), gx2w(gx2w), gy2w(gy2w), gx2w_strides(gx2w_strides), gy2w_strides(gy2w_strides)
	{

	}
	const float *cx;
	const float *cy;
	const float *width;
	float **gx2w;
	float **gy2w;
	const std::size_t *gx2w_strides;
	const std::size_t *gy2w_strides;
};

template <TRN::CPU::Implementation Implementation>
struct TRN::CPU::Algorithm<Implementation>::Handle
{
	std::vector<float *> temp;
};

class Widrow_Hoff
{
private:
	const float *learning_rate;
public:
	 Widrow_Hoff(const float *learning_rate) : learning_rate(learning_rate) {}
	 const float *get_learning_rate() const
	{
		return learning_rate;
	}
};
class Nothing
{
};



template <TRN::CPU::Implementation Implementation>
TRN::CPU::Algorithm<Implementation>::Algorithm() : 
	handle(std::make_unique<Handle>())
{
	vmlSetMode(VML_LA | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE);
}

template <TRN::CPU::Implementation Implementation>
TRN::CPU::Algorithm<Implementation>::~Algorithm()
{
	for (auto temp : handle->temp)
	{
		TRN::CPU::Memory<Implementation>::deallocate_implementation(temp);
	}
	handle.reset();
}
template <TRN::CPU::Implementation Implementation>
void TRN::CPU::Algorithm<Implementation>::preallocate(const std::size_t &stimulus_size, const std::size_t &reservoir_size,
	const std::size_t &prediction_size, const std::size_t &batch_size)
{
	handle->temp.resize(omp_get_max_threads());
	for (std::size_t tid = 0; tid < handle->temp.size(); tid++)
	{
		std::size_t stride;
		TRN::CPU::Memory<Implementation>::allocate_implementation((void **)&handle->temp[tid], stride, sizeof(float), stimulus_size, 1);
	}
}


template <TRN::CPU::Implementation Implementation>
static inline float 	compute_mse(const float *A, const float *B, const size_t &cols)
{
	return 0.0f;
}

template <TRN::CPU::Implementation Implementation>
void TRN::CPU::Algorithm<Implementation>::mean_square_error
(
	const std::size_t &batch_size,
	const float **batched_predicted, const std::size_t *batched_predicted_rows, const std::size_t *batched_predicted_cols, const std::size_t *batched_predicted_strides,
	const float *expected, const std::size_t &expected_rows, const std::size_t &expected_cols, const std::size_t &expected_stride,
	float *result, const std::size_t &result_rows, const std::size_t &result_cols, const std::size_t &result_stride
)
{
	int K = expected_rows * batch_size;
#pragma omp parallel for
	for (int k = 0; k < K; k++)
	{
		int batch = k % batch_size;
		int row = k / batch_size;
		const float *predicted = batched_predicted[batch];
		const std::size_t predicted_rows = batched_predicted_rows[batch];
		const std::size_t predicted_cols = batched_predicted_cols[batch];
		const std::size_t predicted_stride = batched_predicted_strides[batch];		
		assert(predicted_cols == expected_cols);
		assert(predicted_rows == expected_rows);
		auto mse = compute_mse<Implementation>(&predicted[row * predicted_stride], &expected[row * expected_stride], expected_cols);

		result[row * result_stride + batch] = mse;
	}
}


static inline float clamp(const float &x, const float &m, const float &M)
{
	return max(min(x, M), m);
}

template <TRN::CPU::Implementation Implementation>
void TRN::CPU::Algorithm<Implementation>::compute_roi(const std::size_t &batch_size,
	const std::size_t &rows, const std::size_t &cols,
	const float &x_min, const float &x_max, const float &y_min, const float &y_max,
	const float &radius,
	const float **current_position, const std::size_t *current_position_strides,
	std::size_t *roi_row_begin, std::size_t *roi_row_end, std::size_t *roi_col_begin, std::size_t *roi_col_end)
{
	if (radius > 0.0f)
	{
		auto y_range = y_max - y_min;
		auto x_range = x_max - x_min;

#pragma omp parallel for
		for (int batch = 0; batch < batch_size; batch++)
		{
			auto x = current_position[batch][0];
			auto y = current_position[batch][1];
			
			auto roi_x_min = clamp(x - radius, x_min, x_max);
			auto roi_x_max = clamp(x + radius, x_min, x_max);
			auto roi_y_min = clamp(y - radius, y_min, y_max);
			auto roi_y_max = clamp(y + radius, y_min, y_max);

			

			roi_row_begin[batch] = round_down<Implementation>((rows - 1) * ((roi_y_min - y_min) / y_range));
			roi_row_end[batch] = round_up<Implementation>((rows - 1) * ((roi_y_max - y_min) / y_range), rows);
			roi_col_begin[batch] = round_down<Implementation>((cols - 1) * ((roi_x_min - x_min) / x_range));
			roi_col_end[batch] = round_up<Implementation>((cols - 1) * ((roi_x_max - x_min) / x_range), cols);

			assert(roi_row_begin[batch] <= roi_row_end[batch]);
			assert(roi_col_begin[batch] <= roi_col_end[batch]);
		}
	}
	else
	{
		std::fill(roi_row_begin, roi_row_begin + batch_size, 0);
		std::fill(roi_row_end, roi_row_end + batch_size, rows);
		std::fill(roi_col_begin, roi_col_begin + batch_size, 0);
		std::fill(roi_col_end, roi_col_end + batch_size, cols);
	}
 }


template <TRN::CPU::Implementation Implementation>
void TRN::CPU::Algorithm<Implementation>::decode_placecells_linear(
	const std::size_t &batch_size, const std::size_t &place_cells_number,
	const float *cx,
	const float *cy,
	const float **batched_prediction, const std::size_t *batched_prediction_strides,
	float **batched_decoded_position, const std::size_t *batched_decoded_position_strides)
{
#pragma omp parallel for
	for (int batch = 0; batch < batch_size; batch++)
	{
		auto activation = batched_prediction[batch];
		auto decoded_position = batched_decoded_position[batch];

		float s = cblas_sasum(place_cells_number, activation, 1);
		
		if (s > 0.0f)
		{
			auto x = cblas_sdot(place_cells_number, activation, 1, cx, 1) / s;
			auto y = cblas_sdot(place_cells_number, activation, 1, cy, 1) / s;
			decoded_position[0] = x;
			decoded_position[1] = y;
		}
		else
		{
			std::fill(decoded_position, decoded_position + 2, 0.0f);
		}
	}
 }

/*template <TRN::CPU::Implementation Implementation>
static void prepare(const Map &parameter, std::vector<std::pair<std::size_t, std::size_t>> &batch_row_pairs, const std::size_t &batch_size, const std::size_t &place_cells_number, const std::size_t &rows, const std::size_t &cols,
	const float *x_grid, const float *y_grid, 
	const std::size_t &roi_rows, const std::size_t &roi_cols,
	const std::size_t *roi_row_begin, const std::size_t *roi_row_end, const std::size_t *roi_col_begin, const std::size_t *roi_col_end )
{
	for (int batch = 0; batch < batch_size; batch++)
	{
		const auto roi_valid_rows = roi_row_end[batch] - roi_row_begin[batch];
		for (int k = 0; k < roi_valid_rows; k++)
			batch_row_pairs.push_back(std::make_pair(batch, k));
	}
}
template <TRN::CPU::Implementation Implementation>
static void prepare(const Model &parameter, 
	std::vector<std::pair<std::size_t, std::size_t>> &batch_row_pairs,
	const std::size_t &batch_size, const std::size_t &place_cells_number, 
	const std::size_t &rows, const std::size_t &cols,
	const float *x_grid, const float *y_grid,
	const std::size_t &roi_rows, const std::size_t &roi_cols,
	const std::size_t *roi_row_begin, const std::size_t *roi_row_end, const std::size_t *roi_col_begin, const std::size_t *roi_col_end)
{
	std::vector<std::pair<std::size_t, std::size_t>> batch_col_pairs;
	for (int batch = 0; batch < batch_size; batch++)
	{
		const auto roi_valid_rows = roi_row_end[batch] - roi_row_begin[batch];
		const auto roi_valid_cols = roi_col_end[batch] - roi_col_begin[batch];
		for (int k = 0; k < roi_valid_rows; k++)
			batch_row_pairs.push_back(std::make_pair(batch, k));
		for (int k = 0; k < roi_valid_cols; k++)
			batch_col_pairs.push_back(std::make_pair(batch, k));
	}

#pragma omp parallel for
	for (int k = 0; k < batch_row_pairs.size(); k++)
	{
		auto batch = batch_row_pairs[k].first;
		auto roi_row = batch_row_pairs[k].second;

		const auto roi_valid_rows = roi_row_end[batch] - roi_row_begin[batch];
		assert(roi_valid_rows <= roi_rows);
		const auto row_stride = parameter.gy2w_strides[batch];
		const auto row_offset = roi_row_begin[batch];
		auto gy2w = parameter.gy2w[batch];
		auto gy = &y_grid[row_offset];

		diff2<Implementation>(set1_ps(gy[roi_row]), parameter.cy, parameter.width, place_cells_number, &gy2w[roi_row * row_stride]);
	}
#pragma omp parallel for
	for (int k = 0; k < batch_col_pairs.size(); k++)
	{
		auto batch = batch_col_pairs[k].first;
		auto roi_col = batch_col_pairs[k].second;
		const auto roi_valid_cols = roi_col_end[batch] - roi_col_begin[batch];

		assert(roi_valid_cols <= roi_cols);
		const auto col_stride = parameter.gx2w_strides[batch];
		const auto col_offset = roi_col_begin[batch];
		auto gx2w = parameter.gx2w[batch];
		auto gx = &x_grid[col_offset];

		diff2<Implementation>(set1_ps(gx[roi_col]), parameter.cx, parameter.width, place_cells_number, &gx2w[roi_col * col_stride]);
	}
}*/
template <TRN::CPU::Implementation Implementation>
static inline void place_cell_activation_norm2(const Map &parameter, const float *p,
	const std::size_t &roi_row, const std::size_t &roi_col,
	const std::size_t &row, const std::size_t &col, 
	const std::size_t &rows, const std::size_t &cols,
	const float &x_min, const float &x_range, const float &y_min, const float &y_range,
	const std::size_t &place_cells_number, const std::size_t &batch, 
	float *d,
	typename TRN::CPU::Traits<Implementation>::type &v)
{
	const auto offset = row * cols + col;
	for (std::size_t l = 0; l < TRN::CPU::Traits<Implementation>::step; l++)
	{
		auto a = &parameter.data[(offset + l) * parameter.stride];
		sub<Implementation>(a, p, place_cells_number, d);
		set_element(cblas_sdot(place_cells_number, d, 1, d, 1), l, v);
	}
}

template <TRN::CPU::Implementation Implementation>
static inline void place_cell_activation_norm2(const Model &parameter, const float *p,
	const std::size_t &roi_row, const std::size_t &roi_col, 
	const std::size_t &row, const std::size_t &col, 
	const std::size_t &rows, const std::size_t &cols, 
	const float &x_min, const float &x_range, const float &y_min, const float &y_range,
	const std::size_t &place_cells_number, const std::size_t &batch, 
	float *d,
	typename TRN::CPU::Traits<Implementation>::type &v)
{


	auto gy = set1_ps(row_to_y(row, rows, y_min, y_range));
	for (std::size_t l = 0; l < TRN::CPU::Traits<Implementation>::step; l++)
	{
		auto gx = set1_ps(col_to_x(col+l, cols, x_min, x_range));

		auto dp = dot_product_sub<Implementation>(gx, gy, parameter.cx, parameter.cy, parameter.width, p, place_cells_number);

		/*place_cell_activation<Implementation>(parameter.cx, parameter.cy, parameter.width, place_cells_number,gx, gy, d);
		sub<Implementation>(d, p, place_cells_number, d);
		set_element(cblas_sdot(place_cells_number, d, 1, d, 1), l, v);*/

		set_element(dp, l, v);
	}
}

template <TRN::CPU::Implementation Implementation>
static inline void place_cell_activation(const float *cx, const float *cy, const float *w, 
	const std::size_t &cols, const typename TRN::CPU::Traits<Implementation>::type &x, const typename TRN::CPU::Traits<Implementation>::type &y, float *activations)
{
	std::size_t col = 0;

	if (cols - col >= _8)
	{
		for (; col + _8 - 1 < cols; col += _8)
		{
			stream_ps(&activations[col + _0], mul_ps(load_ps(&w[col + _0]), add_ps(sqr_ps(sub_ps(load_ps(&cx[col + _0]), x)), sqr_ps(sub_ps(load_ps(&cy[col + _0]), y)))));
			stream_ps(&activations[col + _1], mul_ps(load_ps(&w[col + _1]), add_ps(sqr_ps(sub_ps(load_ps(&cx[col + _1]), x)), sqr_ps(sub_ps(load_ps(&cy[col + _1]), y)))));
			stream_ps(&activations[col + _2], mul_ps(load_ps(&w[col + _2]), add_ps(sqr_ps(sub_ps(load_ps(&cx[col + _2]), x)), sqr_ps(sub_ps(load_ps(&cy[col + _2]), y)))));
			stream_ps(&activations[col + _3], mul_ps(load_ps(&w[col + _3]), add_ps(sqr_ps(sub_ps(load_ps(&cx[col + _3]), x)), sqr_ps(sub_ps(load_ps(&cy[col + _3]), y)))));
			stream_ps(&activations[col + _4], mul_ps(load_ps(&w[col + _4]), add_ps(sqr_ps(sub_ps(load_ps(&cx[col + _4]), x)), sqr_ps(sub_ps(load_ps(&cy[col + _4]), y)))));
			stream_ps(&activations[col + _5], mul_ps(load_ps(&w[col + _5]), add_ps(sqr_ps(sub_ps(load_ps(&cx[col + _5]), x)), sqr_ps(sub_ps(load_ps(&cy[col + _5]), y)))));
			stream_ps(&activations[col + _6], mul_ps(load_ps(&w[col + _6]), add_ps(sqr_ps(sub_ps(load_ps(&cx[col + _6]), x)), sqr_ps(sub_ps(load_ps(&cy[col + _6]), y)))));
			stream_ps(&activations[col + _7], mul_ps(load_ps(&w[col + _7]), add_ps(sqr_ps(sub_ps(load_ps(&cx[col + _7]), x)), sqr_ps(sub_ps(load_ps(&cy[col + _7]), y)))));
		}
	}
	if (cols - col >= _4)
	{
		for (; col + _4 - 1 < cols; col += _4)
		{
			stream_ps(&activations[col + _0], mul_ps(load_ps(&w[col + _0]), add_ps(sqr_ps(sub_ps(load_ps(&cx[col + _0]), x)), sqr_ps(sub_ps(load_ps(&cy[col + _0]), y)))));
			stream_ps(&activations[col + _1], mul_ps(load_ps(&w[col + _1]), add_ps(sqr_ps(sub_ps(load_ps(&cx[col + _1]), x)), sqr_ps(sub_ps(load_ps(&cy[col + _1]), y)))));
			stream_ps(&activations[col + _2], mul_ps(load_ps(&w[col + _2]), add_ps(sqr_ps(sub_ps(load_ps(&cx[col + _2]), x)), sqr_ps(sub_ps(load_ps(&cy[col + _2]), y)))));
			stream_ps(&activations[col + _3], mul_ps(load_ps(&w[col + _3]), add_ps(sqr_ps(sub_ps(load_ps(&cx[col + _3]), x)), sqr_ps(sub_ps(load_ps(&cy[col + _3]), y)))));
		}
	}
	if (cols - col >= _2)
	{
		for (; col + _2 - 1 < cols; col += _2)
		{
			stream_ps(&activations[col + _0], mul_ps(load_ps(&w[col + _0]), add_ps(sqr_ps(sub_ps(load_ps(&cx[col + _0]), x)), sqr_ps(sub_ps(load_ps(&cy[col + _0]), y)))));
			stream_ps(&activations[col + _1], mul_ps(load_ps(&w[col + _1]), add_ps(sqr_ps(sub_ps(load_ps(&cx[col + _1]), x)), sqr_ps(sub_ps(load_ps(&cy[col + _1]), y)))));
		}
	}
	if (cols - col > 0)
	{
		for (; col < cols; col += _1)
		{
			stream_ps(&activations[col + _0], mul_ps(load_ps(&w[col + _0]), add_ps(sqr_ps(sub_ps(load_ps(&cx[col + _0]), x)), sqr_ps(sub_ps(load_ps(&cy[col + _0]), y)))));
		}
	}
	exp_v(cols, activations, activations);
}
template <TRN::CPU::Implementation Implementation>
void TRN::CPU::Algorithm<Implementation>::encode_placecells_model(
	const std::size_t &batch_size, const std::size_t &place_cells_number,
	const float *cx,
	const float *cy,
	const float *width,
	const float **batched_decoded_position, const std::size_t *batched_decoded_position_strides,
	float **batched_stimulus, const std::size_t *batched_stimulus_strides)
{
#pragma omp parallel for 
	for (int batch = 0; batch < batch_size; batch++)
	{
		auto position = batched_decoded_position[batch];
		auto stimulus = batched_stimulus[batch];

		auto x = set1_ps(position[0]);
		auto y = set1_ps(position[1]);

		place_cell_activation<Implementation>(cx, cy, width, place_cells_number, x, y, stimulus);
	}
}

template <TRN::CPU::Implementation Implementation, typename Parameter>
static inline void decode_placecells_bayesian
(const std::size_t &batch_size, const std::size_t &place_cells_number,
	const std::size_t &rows, const std::size_t &cols,
	const std::size_t &roi_rows, const std::size_t &roi_cols,
	const std::size_t *roi_row_begin, const std::size_t *roi_row_end, const std::size_t *roi_col_begin, const std::size_t *roi_col_end,

	const float &radius,
	const float &cos_half_angle,
	const float &scale,
	const float &sigma,
	const unsigned long &seed,
	const Parameter &parameter,
	const float &x_min, const float &x_range, const float &y_min, const float &y_range,
	const float *x_grid, const std::size_t &x_grid_stride,
	const float *y_grid, const std::size_t &y_grid_stride,
	const float **batched_previous_position, const std::size_t *batched_previous_position_stride,
	const float **batched_current_position, const std::size_t *batched_current_position_stride,
	const float **batched_predicted_activations, const std::size_t *batched_predicted_activations_stride,
	float **batched_direction, const std::size_t *batched_direction_stride,
	float **batched_x_grid_centered, const std::size_t *batched_x_grid_centered_stride,
	float **batched_y_grid_centered, const std::size_t *batched_y_grid_centered_stride,
	float **batched_location_probability, const std::size_t *batched_location_probability_strides,
	float **temp)
{
	const auto _inv_sigma2 = set1_ps(-1.0f / (2.0f*(sigma * sigma)));

	const auto K = batch_size * roi_rows;
#pragma omp parallel for 
	for (int k = 0; k < K; k++)
	{
		const auto batch = k / roi_rows;
		const auto roi_row = k % roi_rows;

		const auto roi_valid_rows = roi_row_end[batch] - roi_row_begin[batch];
		const auto stride = batched_location_probability_strides[batch];
		const auto location_probability = &batched_location_probability[batch][roi_row * stride];
		auto d = temp[omp_get_thread_num()];
		if (roi_row < roi_valid_rows)
		{
			const auto roi_row_offset = roi_row_begin[batch];

			auto row = roi_row_offset + roi_row;
			assert(0 <= row && row < rows);
			auto roi_valid_cols = roi_col_end[batch] - roi_col_begin[batch];



			const auto p = batched_predicted_activations[batch];
			std::size_t roi_col = 0;
			const auto roi_col_offset = roi_col_begin[batch];
			for (; roi_col < roi_valid_cols; roi_col += TRN::CPU::Traits<Implementation>::step)
			{
				auto col = roi_col_offset + roi_col;
				assert(0 <= col && col < cols);
				typename  TRN::CPU::Traits<Implementation>::type v;
				place_cell_activation_norm2<Implementation>(parameter, p, roi_row, roi_col, row, col, rows, cols, x_min, x_range, y_min, y_range, place_cells_number, batch, d, v);
				stream_ps(&location_probability[roi_col], mul_ps(_inv_sigma2, v));
			}
			exp_v(roi_valid_cols, location_probability, location_probability);
			for (; roi_col < roi_cols; roi_col += TRN::CPU::Traits<Implementation>::step)
				stream_ps(&location_probability[roi_col], set1_ps(0.0f));
		}
		else
		{
			std::size_t roi_col = 0;
			for (; roi_col < roi_cols; roi_col += TRN::CPU::Traits<Implementation>::step)
				stream_ps(&location_probability[roi_col], set1_ps(0.0f));
		}
	}

	if (radius > 0.0f)
	{
		const auto r2 = radius * radius;
#pragma omp parallel for
		for (int batch = 0; batch < batch_size; batch++)
		{
			auto cx = batched_current_position[batch][0];
			auto cy = batched_current_position[batch][1];
			auto px = batched_previous_position[batch][0];
			auto py = batched_previous_position[batch][1];
			auto dx = cx - px;
			auto dy = cy - py;
			auto dh = std::sqrtf(dx * dx + dy *dy);
			auto inv_norm = dh > 0.0f ? 1.0f / dh : 0.0f;

			batched_direction[batch][0] = dx *inv_norm;
			batched_direction[batch][1] = dy *inv_norm;
			auto roi_rows = roi_row_end[batch] - roi_row_begin[batch];
			auto roi_cols = roi_col_end[batch] - roi_col_begin[batch];

			diff<Implementation>(set1_ps(cy), &y_grid[roi_row_begin[batch]], roi_rows, batched_y_grid_centered[batch]);
			diff<Implementation>(set1_ps(cx), &x_grid[roi_col_begin[batch]], roi_cols, batched_x_grid_centered[batch]);
		}

		std::vector<VSLStreamStatePtr> streams(omp_get_max_threads());
#pragma omp parallel for
		for (int tid = 0; tid < streams.size(); tid++)
		{
			vslNewStream(&streams[tid], VSL_BRNG_SFMT19937, seed);
		}
#pragma omp parallel for 
		for (int k = 0; k < K; k++)
		{
			const auto batch = k / roi_rows;
			const auto roi_row = k % roi_rows;
			const auto roi_row_offset = roi_row_begin[batch];
			auto roi_valid_cols = roi_col_end[batch] - roi_col_begin[batch];
			const auto stride = batched_location_probability_strides[batch];
			const auto location_probability = &batched_location_probability[batch][roi_row * stride];
			auto tid = omp_get_thread_num();

			auto by = batched_y_grid_centered[batch][roi_row];

			auto ax = batched_direction[batch][0];
			auto ay = batched_direction[batch][1];

			if (cos_half_angle < 1.0f && ax * ax + ay * ay > 0.0f)
				inside_circle_sector<Implementation>(streams[tid], scale, cos_half_angle, ax, ay, batched_x_grid_centered[batch], by, r2, roi_valid_cols, location_probability);
			else
				inside_circle<Implementation>(streams[tid], scale, batched_x_grid_centered[batch], by, r2, roi_valid_cols, location_probability);
		}
#pragma omp parallel for
		for (int tid = 0; tid < streams.size(); tid++)
		{
			vslDeleteStream(&streams[tid]);
		}
	}
}


template <TRN::CPU::Implementation Implementation>
void TRN::CPU::Algorithm<Implementation>::decode_placecells_kernel_map
	(
		const std::size_t &batch_size, const std::size_t &place_cells_number,
		const std::size_t &rows, const std::size_t &cols,
		const std::size_t &roi_rows, const std::size_t &roi_cols,
		const std::size_t *roi_row_begin, const std::size_t *roi_row_end, const std::size_t *roi_col_begin, const std::size_t *roi_col_end,
		const float &x_min, const float &x_max, const float &y_min, const float &y_max,
		const float &radius,
		const float &cos_half_angle,
		const float &scale,
		const float &sigma,
		const unsigned long &seed,
		const float *firing_rate_map, const std::size_t &firing_rate_maps_stride,
		const float *x_grid, const std::size_t &x_grid_stride,
		const float *y_grid, const std::size_t &y_grid_stride,
		const float **batched_previous_position, const std::size_t *batched_previous_position_stride,
		const float **batched_current_position, const std::size_t *batched_current_position_stride,
		const float **batched_predicted_activations, const std::size_t *batched_predicted_activations_stride,
		float **batched_direction, const std::size_t *batched_direction_stride,
		float **batched_x_grid_centered, const std::size_t *batched_x_grid_centered_stride,
		float **batched_y_grid_centered, const std::size_t *batched_y_grid_centered_stride,
		float **batched_location_probability, const std::size_t *batched_location_probability_strides)
{
	decode_placecells_bayesian<Implementation>(
		batch_size, place_cells_number,
		rows, cols,
		roi_rows, roi_cols,
		roi_row_begin, roi_row_end, roi_col_begin, roi_col_end,
		radius,
		cos_half_angle,
		scale,
		sigma,
		seed,
		Map(firing_rate_map, firing_rate_maps_stride),
		x_min, x_max - x_min, y_min, y_max - y_min,
		x_grid, x_grid_stride,
		y_grid, y_grid_stride,
		batched_previous_position, batched_previous_position_stride,
		batched_current_position, batched_current_position_stride,
		batched_predicted_activations, batched_predicted_activations_stride,
		batched_direction, batched_direction_stride,
		batched_x_grid_centered, batched_x_grid_centered_stride,
		batched_y_grid_centered, batched_y_grid_centered_stride,
		batched_location_probability, batched_location_probability_strides,
		handle->temp.data());
}

template <TRN::CPU::Implementation Implementation>
void TRN::CPU::Algorithm<Implementation>::decode_placecells_kernel_model
(
	const std::size_t &batch_size, const std::size_t &place_cells_number,
	const std::size_t &rows, const std::size_t &cols,
	const std::size_t &roi_rows, const std::size_t &roi_cols,
	const std::size_t *roi_row_begin, const std::size_t *roi_row_end, const std::size_t *roi_col_begin, const std::size_t *roi_col_end,
	const float &x_min, const float &x_max, const float &y_min, const float &y_max,
	const float &radius,
	const float &cos_half_angle,
	const float &scale,
	const float &sigma,
	const unsigned long &seed,
	const float *cx, const float *cy, const float *width,
	float **gx2w, const std::size_t *gx2w_strides,
	float **gy2w, const std::size_t *gy2w_strides,
	const float *x_grid, const std::size_t &x_grid_stride,
	const float *y_grid, const std::size_t &y_grid_stride,
	const float **batched_previous_position, const std::size_t *batched_previous_position_stride,
	const float **batched_current_position, const std::size_t *batched_current_position_stride,
	const float **batched_predicted_activations, const std::size_t *batched_predicted_activations_stride,
	float **batched_direction, const std::size_t *batched_direction_stride,
	float **batched_x_grid_centered, const std::size_t *batched_x_grid_centered_stride,
	float **batched_y_grid_centered, const std::size_t *batched_y_grid_centered_stride,
	float **batched_location_probability, const std::size_t *batched_location_probability_strides
)
{
	decode_placecells_bayesian<Implementation>(
		batch_size, place_cells_number,
		rows, cols,
		roi_rows, roi_cols,
		roi_row_begin, roi_row_end, roi_col_begin, roi_col_end,
		radius,
		cos_half_angle,
		scale,
		sigma,
		seed,
		Model(cx, cy, width, gx2w, gy2w, gx2w_strides, gy2w_strides),
		x_min, x_max - x_min, y_min, y_max - y_min,
		x_grid, x_grid_stride,
		y_grid, y_grid_stride,
		batched_previous_position, batched_previous_position_stride,
		batched_current_position, batched_current_position_stride,
		batched_predicted_activations, batched_predicted_activations_stride,
		batched_direction, batched_direction_stride,
		batched_x_grid_centered, batched_x_grid_centered_stride,
		batched_y_grid_centered, batched_y_grid_centered_stride,
		batched_location_probability, batched_location_probability_strides,
		handle->temp.data());
}
template <TRN::CPU::Implementation Implementation>
static inline void 	weighted_sum(
	const float *A, const typename TRN::CPU::Traits<Implementation>::type  &a,
	const float *B, const typename TRN::CPU::Traits<Implementation>::type  &b,
	const std::size_t &cols, float *C)
{
	std::size_t col = 0;
	
	if (cols - col >= _8)
	{
		for (; col + _8 - 1 < cols; col += _8)
		{
			stream_ps(&C[col + _0], mul_add_ps(load_ps(&A[col + _0]), a, mul_ps(load_ps(&B[col + _0]), b)));
			stream_ps(&C[col + _1], mul_add_ps(load_ps(&A[col + _1]), a, mul_ps(load_ps(&B[col + _1]), b)));
			stream_ps(&C[col + _2], mul_add_ps(load_ps(&A[col + _2]), a, mul_ps(load_ps(&B[col + _2]), b)));
			stream_ps(&C[col + _3], mul_add_ps(load_ps(&A[col + _3]), a, mul_ps(load_ps(&B[col + _3]), b)));
			stream_ps(&C[col + _4], mul_add_ps(load_ps(&A[col + _4]), a, mul_ps(load_ps(&B[col + _4]), b)));
			stream_ps(&C[col + _5], mul_add_ps(load_ps(&A[col + _5]), a, mul_ps(load_ps(&B[col + _5]), b)));
			stream_ps(&C[col + _6], mul_add_ps(load_ps(&A[col + _6]), a, mul_ps(load_ps(&B[col + _6]), b)));
			stream_ps(&C[col + _7], mul_add_ps(load_ps(&A[col + _7]), a, mul_ps(load_ps(&B[col + _7]), b)));

		}
	}
	if (cols - col >= _4)
	{
		for (; col + _4 - 1 < cols; col += _4)
		{
			stream_ps(&C[col + _0], mul_add_ps(load_ps(&A[col + _0]), a, mul_ps(load_ps(&B[col + _0]), b)));
			stream_ps(&C[col + _1], mul_add_ps(load_ps(&A[col + _1]), a, mul_ps(load_ps(&B[col + _1]), b)));
			stream_ps(&C[col + _2], mul_add_ps(load_ps(&A[col + _2]), a, mul_ps(load_ps(&B[col + _2]), b)));
			stream_ps(&C[col + _3], mul_add_ps(load_ps(&A[col + _3]), a, mul_ps(load_ps(&B[col + _3]), b)));
		}
	}
	if (cols - col >= _2)
	{
		for (; col + _2 - 1 < cols; col += _2)
		{
			stream_ps(&C[col + _0], mul_add_ps(load_ps(&A[col + _0]), a, mul_ps(load_ps(&B[col + _0]), b)));
			stream_ps(&C[col + _1], mul_add_ps(load_ps(&A[col + _1]), a, mul_ps(load_ps(&B[col + _1]), b)));
		}
	}
	if (cols - col > 0)
	{
		for (; col < cols; col += _1)
		{
			stream_ps(&C[col + _0], mul_add_ps(load_ps(&A[col + _0]), a, mul_ps(load_ps(&B[col + _0]), b)));
		}
	}


}
template <TRN::CPU::Implementation Implementation>
static inline void 	sub_scale(
	const float *A,
	const float *B,
	const typename TRN::CPU::Traits<Implementation>::type &scale,
	const std::size_t &cols, float *C)
{
	std::size_t col = 0;
	/*if (cols - col > 06)
	{
		for (; col + _16 - 1 < cols; col += _16)
		{
			stream_ps(&C[col + _0], mul_ps(scale, sub_ps(load_ps(&A[col + _0]), load_ps(&B[col + _0]))));
			stream_ps(&C[col + _1], mul_ps(scale, sub_ps(load_ps(&A[col + _1]), load_ps(&B[col + _1]))));
			stream_ps(&C[col + _2], mul_ps(scale, sub_ps(load_ps(&A[col + _2]), load_ps(&B[col + _2]))));
			stream_ps(&C[col + _3], mul_ps(scale, sub_ps(load_ps(&A[col + _3]), load_ps(&B[col + _3]))));
			stream_ps(&C[col + _4], mul_ps(scale, sub_ps(load_ps(&A[col + _4]), load_ps(&B[col + _4]))));
			stream_ps(&C[col + _5], mul_ps(scale, sub_ps(load_ps(&A[col + _5]), load_ps(&B[col + _5]))));
			stream_ps(&C[col + _6], mul_ps(scale, sub_ps(load_ps(&A[col + _6]), load_ps(&B[col + _6]))));
			stream_ps(&C[col + _7], mul_ps(scale, sub_ps(load_ps(&A[col + _7]), load_ps(&B[col + _7]))));
			stream_ps(&C[col + _8], mul_ps(scale, sub_ps(load_ps(&A[col + _8]), load_ps(&B[col + _8]))));
			stream_ps(&C[col + _9], mul_ps(scale, sub_ps(load_ps(&A[col + _9]), load_ps(&B[col + _9]))));
			stream_ps(&C[col + _10], mul_ps(scale, sub_ps(load_ps(&A[col + _10]), load_ps(&B[col + _10]))));
			stream_ps(&C[col + _11], mul_ps(scale, sub_ps(load_ps(&A[col + _11]), load_ps(&B[col + _11]))));
			stream_ps(&C[col + _12], mul_ps(scale, sub_ps(load_ps(&A[col + _12]), load_ps(&B[col + _12]))));
			stream_ps(&C[col + _13], mul_ps(scale, sub_ps(load_ps(&A[col + _13]), load_ps(&B[col + _13]))));
			stream_ps(&C[col + _14], mul_ps(scale, sub_ps(load_ps(&A[col + _14]), load_ps(&B[col + _14]))));
			stream_ps(&C[col + _15], mul_ps(scale, sub_ps(load_ps(&A[col + _15]), load_ps(&B[col + _15]))));
		}
	}*/
	if (cols - col >= _8)
	{
		for (; col + _8 - 1 < cols; col += _8)
		{
			stream_ps(&C[col + _0], mul_ps(scale, sub_ps(load_ps(&A[col + _0]), load_ps(&B[col + _0]))));
			stream_ps(&C[col + _1], mul_ps(scale, sub_ps(load_ps(&A[col + _1]), load_ps(&B[col + _1]))));
			stream_ps(&C[col + _2], mul_ps(scale, sub_ps(load_ps(&A[col + _2]), load_ps(&B[col + _2]))));
			stream_ps(&C[col + _3], mul_ps(scale, sub_ps(load_ps(&A[col + _3]), load_ps(&B[col + _3]))));
			stream_ps(&C[col + _4], mul_ps(scale, sub_ps(load_ps(&A[col + _4]), load_ps(&B[col + _4]))));
			stream_ps(&C[col + _5], mul_ps(scale, sub_ps(load_ps(&A[col + _5]), load_ps(&B[col + _5]))));
			stream_ps(&C[col + _6], mul_ps(scale, sub_ps(load_ps(&A[col + _6]), load_ps(&B[col + _6]))));
			stream_ps(&C[col + _7], mul_ps(scale, sub_ps(load_ps(&A[col + _7]), load_ps(&B[col + _7]))));
		}
	}
	if (cols - col >= _4)
	{
		for (; col + _4 - 1 < cols; col += _4)
		{
			stream_ps(&C[col + _0], mul_ps(scale, sub_ps(load_ps(&A[col + _0]), load_ps(&B[col + _0]))));
			stream_ps(&C[col + _1], mul_ps(scale, sub_ps(load_ps(&A[col + _1]), load_ps(&B[col + _1]))));
			stream_ps(&C[col + _2], mul_ps(scale, sub_ps(load_ps(&A[col + _2]), load_ps(&B[col + _2]))));
			stream_ps(&C[col + _3], mul_ps(scale, sub_ps(load_ps(&A[col + _3]), load_ps(&B[col + _3]))));
		}
	}
	if (cols - col >= _2)
	{
		for (; col + _2 - 1 < cols; col += _2)
		{
			stream_ps(&C[col + _0], mul_ps(scale, sub_ps(load_ps(&A[col + _0]), load_ps(&B[col + _0]))));
			stream_ps(&C[col + _1], mul_ps(scale, sub_ps(load_ps(&A[col + _1]), load_ps(&B[col + _1]))));
		}
	}
	 if (cols - col > 0)
	{
		for (; col  < cols; col += _1)
		{
			stream_ps(&C[col + _0], mul_ps(scale, sub_ps(load_ps(&A[col + _0]), load_ps(&B[col + _0]))));
		}
	}

}
template <TRN::CPU::Implementation Implementation>
static inline void 	sum(
	const float *A,
	const float *B,
	const std::size_t &cols, float *C)
{
	std::size_t col = 0;
	/*if (cols - col > 06)
	{
		for (; col + _16 - 1 < cols; col += _16)
		{
			stream_ps(&C[col + _0], add_ps(load_ps(&A[col + _0]), load_ps(&B[col + _0])));
			stream_ps(&C[col + _1], add_ps(load_ps(&A[col + _1]), load_ps(&B[col + _1])));
			stream_ps(&C[col + _2], add_ps(load_ps(&A[col + _2]), load_ps(&B[col + _2])));
			stream_ps(&C[col + _3], add_ps(load_ps(&A[col + _3]), load_ps(&B[col + _3])));
			stream_ps(&C[col + _4], add_ps(load_ps(&A[col + _4]), load_ps(&B[col + _4])));
			stream_ps(&C[col + _5], add_ps(load_ps(&A[col + _5]), load_ps(&B[col + _5])));
			stream_ps(&C[col + _6], add_ps(load_ps(&A[col + _6]), load_ps(&B[col + _6])));
			stream_ps(&C[col + _7], add_ps(load_ps(&A[col + _7]), load_ps(&B[col + _7])));
			stream_ps(&C[col + _8], add_ps(load_ps(&A[col + _8]), load_ps(&B[col + _8])));
			stream_ps(&C[col + _9], add_ps(load_ps(&A[col + _9]), load_ps(&B[col + _9])));
			stream_ps(&C[col + _10], add_ps(load_ps(&A[col + _10]), load_ps(&B[col + _10])));
			stream_ps(&C[col + _11], add_ps(load_ps(&A[col + _11]), load_ps(&B[col + _11])));
			stream_ps(&C[col + _12], add_ps(load_ps(&A[col + _12]), load_ps(&B[col + _12])));
			stream_ps(&C[col + _13], add_ps(load_ps(&A[col + _13]), load_ps(&B[col + _13])));
			stream_ps(&C[col + _14], add_ps(load_ps(&A[col + _14]), load_ps(&B[col + _14])));
			stream_ps(&C[col + _15], add_ps(load_ps(&A[col + _15]), load_ps(&B[col + _15])));
		}
	}*/
	if (cols - col >= _8)
	{
		for (; col + _8 - 1 < cols; col += _8)
		{
			stream_ps(&C[col + _0], add_ps(load_ps(&A[col + _0]), load_ps(&B[col + _0])));
			stream_ps(&C[col + _1], add_ps(load_ps(&A[col + _1]), load_ps(&B[col + _1])));
			stream_ps(&C[col + _2], add_ps(load_ps(&A[col + _2]), load_ps(&B[col + _2])));
			stream_ps(&C[col + _3], add_ps(load_ps(&A[col + _3]), load_ps(&B[col + _3])));
			stream_ps(&C[col + _4], add_ps(load_ps(&A[col + _4]), load_ps(&B[col + _4])));
			stream_ps(&C[col + _5], add_ps(load_ps(&A[col + _5]), load_ps(&B[col + _5])));
			stream_ps(&C[col + _6], add_ps(load_ps(&A[col + _6]), load_ps(&B[col + _6])));
			stream_ps(&C[col + _7], add_ps(load_ps(&A[col + _7]), load_ps(&B[col + _7])));
		}
	}
	if (cols - col >= _4)
	{
		for (; col + _4 - 1 < cols; col += _4)
		{
			stream_ps(&C[col + _0], add_ps(load_ps(&A[col + _0]), load_ps(&B[col + _0])));
			stream_ps(&C[col + _1], add_ps(load_ps(&A[col + _1]), load_ps(&B[col + _1])));
			stream_ps(&C[col + _2], add_ps(load_ps(&A[col + _2]), load_ps(&B[col + _2])));
			stream_ps(&C[col + _3], add_ps(load_ps(&A[col + _3]), load_ps(&B[col + _3])));
		}
	}
	if (cols - col >= _2)
	{
		for (; col + _2 - 1 < cols; col += _2)
		{
			stream_ps(&C[col + _0], add_ps(load_ps(&A[col + _0]), load_ps(&B[col + _0])));
			stream_ps(&C[col + _1], add_ps(load_ps(&A[col + _1]), load_ps(&B[col + _1])));
		}
	}
	 if (cols - col > 0)
	{
		for (; col  < cols; col += _1)
		{
			stream_ps(&C[col + _0], add_ps(load_ps(&A[col + _0]), load_ps(&B[col + _0])));
		}
	}

}

template <TRN::CPU::Implementation Implementation>
static inline void 	weighted_acc(const float *A, const typename TRN::CPU::Traits<Implementation>::type  &a, const std::size_t &cols, float *C)
{
	std::size_t col = 0;
	
	if (cols - col >= _8)
	{
		for (; col + _8 - 1 < cols; col += _8)
		{
			stream_ps(&C[col + _0], mul_add_ps(load_ps(&A[col + _0]), a, load_ps(&C[col + _0])));
			stream_ps(&C[col + _1], mul_add_ps(load_ps(&A[col + _1]), a, load_ps(&C[col + _1])));
			stream_ps(&C[col + _2], mul_add_ps(load_ps(&A[col + _2]), a, load_ps(&C[col + _2])));
			stream_ps(&C[col + _3], mul_add_ps(load_ps(&A[col + _3]), a, load_ps(&C[col + _3])));
			stream_ps(&C[col + _4], mul_add_ps(load_ps(&A[col + _4]), a, load_ps(&C[col + _4])));
			stream_ps(&C[col + _5], mul_add_ps(load_ps(&A[col + _5]), a, load_ps(&C[col + _5])));
			stream_ps(&C[col + _6], mul_add_ps(load_ps(&A[col + _6]), a, load_ps(&C[col + _6])));
			stream_ps(&C[col + _7], mul_add_ps(load_ps(&A[col + _7]), a, load_ps(&C[col + _7])));
		}
	}
	if (cols - col >= _4)
	{
		for (; col + _4 - 1 < cols; col += _4)
		{
			stream_ps(&C[col + _0], mul_add_ps(load_ps(&A[col + _0]), a, load_ps(&C[col + _0])));
			stream_ps(&C[col + _1], mul_add_ps(load_ps(&A[col + _1]), a, load_ps(&C[col + _1])));
			stream_ps(&C[col + _2], mul_add_ps(load_ps(&A[col + _2]), a, load_ps(&C[col + _2])));
			stream_ps(&C[col + _3], mul_add_ps(load_ps(&A[col + _3]), a, load_ps(&C[col + _3])));
		}
	}
	if (cols - col >= _2)
	{
		for (; col + _2 - 1 < cols; col += _2)
		{
			stream_ps(&C[col + _0], mul_add_ps(load_ps(&A[col + _0]), a, load_ps(&C[col + _0])));
			stream_ps(&C[col + _1], mul_add_ps(load_ps(&A[col + _1]), a, load_ps(&C[col + _1])));
		}
	}
	 if (cols - col > 0)
	{
		for (; col  < cols; col += _1)
		{
			stream_ps(&C[col + _0], mul_add_ps(load_ps(&A[col + _0]), a, load_ps(&C[col + _0])));
		}
	}
}


template <TRN::CPU::Implementation Implementation>
static inline void 	diff(const typename TRN::CPU::Traits<Implementation>::type &current, const float *grid, const std::size_t &cols, float *grid_centered)
{
	std::size_t col = 0;
	if (cols - col >= _8)
	{
		for (; col + _8 - 1 < cols; col += _8)
		{
			auto d0 = sub_ps(load_ps(&grid[col + _0]), current);
			stream_ps(&grid_centered[col + _0], d0);

			auto d1 = sub_ps(load_ps(&grid[col + _1]), current);
			stream_ps(&grid_centered[col + _1], d1);

			auto d2 = sub_ps(load_ps(&grid[col + _2]), current);
			stream_ps(&grid_centered[col + _2], d2);

			auto d3 = sub_ps(load_ps(&grid[col + _3]), current);
			stream_ps(&grid_centered[col + _3], d3);

			auto d4 = sub_ps(load_ps(&grid[col + _4]), current);
			stream_ps(&grid_centered[col + _4], d4);

			auto d5 = sub_ps(load_ps(&grid[col + _5]), current);
			stream_ps(&grid_centered[col + _5], d5);

			auto d6 = sub_ps(load_ps(&grid[col + _6]), current);
			stream_ps(&grid_centered[col + _6], d6);

			auto d7 = sub_ps(load_ps(&grid[col + _7]), current);
			stream_ps(&grid_centered[col + _7], d7);
		}
	}
	if (cols - col >= _4)
	{
		for (; col + _4 - 1 < cols; col += _4)
		{
			auto d0 = sub_ps(load_ps(&grid[col + _0]), current);
			stream_ps(&grid_centered[col + _0], d0);

			auto d1 = sub_ps(load_ps(&grid[col + _1]), current);
			stream_ps(&grid_centered[col + _1], d1);

			auto d2 = sub_ps(load_ps(&grid[col + _2]), current);
			stream_ps(&grid_centered[col + _2], d2);

			auto d3 = sub_ps(load_ps(&grid[col + _3]), current);
			stream_ps(&grid_centered[col + _3], d3);
		}
	}
	if (cols - col >= _2)
	{
		for (; col + _2 - 1 < cols; col += _2)
		{
			auto d0 = sub_ps(load_ps(&grid[col + _0]), current);
			stream_ps(&grid_centered[col + _0], d0);

			auto d1 = sub_ps(load_ps(&grid[col + _1]), current);
			stream_ps(&grid_centered[col + _1], d1);
		}
	}
	if (cols - col >= _1)
	{
		for (; col < cols; col += _1)
		{
			auto d0 = sub_ps(load_ps(&grid[col + _0]), current);
			stream_ps(&grid_centered[col + _0], d0);
		}
	}
}
template <TRN::CPU::Implementation Implementation>
static inline void 	diff2(const typename TRN::CPU::Traits<Implementation>::type &current, const float *center, const float *width, const std::size_t &cols, float *grid_centered)
{
	std::size_t col = 0;
	if (cols - col >= _8)
	{
		for (; col + _8 - 1 < cols; col += _8)
		{
			auto d0 = sub_ps(current, load_ps(&center[col + _0]));
			stream_ps(&grid_centered[col + _0], mul_ps(load_ps(&width[col + _0]), mul_ps(d0, d0)));

			auto d1 = sub_ps(current, load_ps(&center[col + _1]));
			stream_ps(&grid_centered[col + _1], mul_ps(load_ps(&width[col + _1]), mul_ps(d1, d1)));

			auto d2 = sub_ps(current, load_ps(&center[col + _2]));
			stream_ps(&grid_centered[col + _2], mul_ps(load_ps(&width[col + _2]), mul_ps(d2, d2)));

			auto d3 = sub_ps(current, load_ps(&center[col + _3]));
			stream_ps(&grid_centered[col + _3], mul_ps(load_ps(&width[col + _3]), mul_ps(d3, d3)));

			auto d4 = sub_ps(current, load_ps(&center[col + _4]));
			stream_ps(&grid_centered[col + _4], mul_ps(load_ps(&width[col + _4]), mul_ps(d4, d4)));

			auto d5 = sub_ps(current, load_ps(&center[col + _5]));
			stream_ps(&grid_centered[col + _5], mul_ps(load_ps(&width[col + _5]), mul_ps(d5, d5)));

			auto d6 = sub_ps(current, load_ps(&center[col + _6]));
			stream_ps(&grid_centered[col + _6], mul_ps(load_ps(&width[col + _6]), mul_ps(d6, d6)));

			auto d7 = sub_ps(current, load_ps(&center[col + _7]));
			stream_ps(&grid_centered[col + _7], mul_ps(load_ps(&width[col + _7]), mul_ps(d7, d7)));
		}
	}
	if (cols - col >= _4)
	{
		for (; col + _4 - 1 < cols; col += _4)
		{
			auto d0 = sub_ps(current, load_ps(&center[col + _0]));
			stream_ps(&grid_centered[col + _0], mul_ps(load_ps(&width[col + _0]), mul_ps(d0, d0)));

			auto d1 = sub_ps(current, load_ps(&center[col + _1]));
			stream_ps(&grid_centered[col + _1], mul_ps(load_ps(&width[col + _1]), mul_ps(d1, d1)));

			auto d2 = sub_ps(current, load_ps(&center[col + _2]));
			stream_ps(&grid_centered[col + _2], mul_ps(load_ps(&width[col + _2]), mul_ps(d2, d2)));

			auto d3 = sub_ps(current, load_ps(&center[col + _3]));
			stream_ps(&grid_centered[col + _3], mul_ps(load_ps(&width[col + _3]), mul_ps(d3, d3)));
		}
	}
	if (cols - col >= _2)
	{
		for (; col + _2 - 1 < cols; col += _2)
		{
			auto d0 = sub_ps(current, load_ps(&center[col + _0]));
			stream_ps(&grid_centered[col + _0], mul_ps(load_ps(&width[col + _0]), mul_ps(d0, d0)));

			auto d1 = sub_ps(current, load_ps(&center[col + _1]));
			stream_ps(&grid_centered[col + _1], mul_ps(load_ps(&width[col + _1]), mul_ps(d1, d1)));
		}
	}
	if (cols - col >= _1)
	{
		for (; col < cols; col += _1)
		{
			auto d0 = sub_ps(current, load_ps(&center[col + _0]));
			stream_ps(&grid_centered[col + _0], mul_ps(load_ps(&width[col + _0]), mul_ps(d0, d0)));
		}
	}
}

template <TRN::CPU::Implementation Implementation>
static inline void 	inside_circle(VSLStreamStatePtr &stream, const float &scale, const float *bx, const float &by, const float &_r2, const std::size_t &cols, float *location_probability_row)
{
	//	 vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, 1000, r, 0.0, scale);
	const typename TRN::CPU::Traits<Implementation>::type _b_y = set1_ps(by);
	const typename TRN::CPU::Traits<Implementation>::type r2 = set1_ps(_r2);

	auto __zero = setzero_ps();

	auto _b2_y = sqr_ps(_b_y);
	
	std::size_t col = 0;

	if (cols - col >= _8)
	{
		float noise[_8];

		for (; col + _8 - 1 < cols; col += _8)
		{
			vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, _8, noise, 0.0, scale);

			auto _b0_x = load_ps(&bx[col + _0]);
			auto dp_b2_0 = mul_add_ps(_b0_x, _b0_x, _b2_y);
			auto in_0 = cmp_lt_ps(dp_b2_0, r2);
			stream_ps(&location_probability_row[col + _0], blendv_ps(__zero, add_ps(load_ps(&noise[_0]), load_ps(&location_probability_row[col + _0])), in_0));

			auto _b1_x = load_ps(&bx[col + _1]);
			auto dp_b2_1 = mul_add_ps(_b1_x, _b1_x, _b2_y);
			auto in_1 = cmp_lt_ps(dp_b2_1, r2);
			stream_ps(&location_probability_row[col + _1], blendv_ps(__zero, add_ps(load_ps(&noise[_1]), load_ps(&location_probability_row[col + _1])), in_1));

			auto _b2_x = load_ps(&bx[col + _2]);
			auto dp_b2_2 = mul_add_ps(_b2_x, _b2_x, _b2_y);
			auto in_2 = cmp_lt_ps(dp_b2_2, r2);
			stream_ps(&location_probability_row[col + _2], blendv_ps(__zero, add_ps(load_ps(&noise[_2]), load_ps(&location_probability_row[col + _2])), in_2));

			auto _b3_x = load_ps(&bx[col + _3]);
			auto dp_b2_3 = mul_add_ps(_b3_x, _b3_x, _b2_y);
			auto in_3 = cmp_lt_ps(dp_b2_3, r2);
			stream_ps(&location_probability_row[col + _3], blendv_ps(__zero, add_ps(load_ps(&noise[_3]), load_ps(&location_probability_row[col + _3])), in_3));

			auto _b4_x = load_ps(&bx[col + _4]);
			auto dp_b2_4 = mul_add_ps(_b4_x, _b4_x, _b2_y);
			auto in_4 = cmp_lt_ps(dp_b2_4, r2);
			stream_ps(&location_probability_row[col + _4], blendv_ps(__zero, add_ps(load_ps(&noise[_4]), load_ps(&location_probability_row[col + _4])), in_4));

			auto _b5_x = load_ps(&bx[col + _5]);
			auto dp_b2_5 = mul_add_ps(_b5_x, _b5_x, _b2_y);
			auto in_5 = cmp_lt_ps(dp_b2_5, r2);
			stream_ps(&location_probability_row[col + _5], blendv_ps(__zero, add_ps(load_ps(&noise[_5]), load_ps(&location_probability_row[col + _5])), in_5));

			auto _b6_x = load_ps(&bx[col + _6]);
			auto dp_b2_6 = mul_add_ps(_b6_x, _b6_x, _b2_y);
			auto in_6 = cmp_lt_ps(dp_b2_6, r2);
			stream_ps(&location_probability_row[col + _6], blendv_ps(__zero, add_ps(load_ps(&noise[_6]), load_ps(&location_probability_row[col + _6])), in_6));

			auto _b7_x = load_ps(&bx[col + _7]);
			auto dp_b2_7 = mul_add_ps(_b7_x, _b7_x, _b2_y);
			auto in_7 = cmp_lt_ps(dp_b2_7, r2);
			stream_ps(&location_probability_row[col + _7], blendv_ps(__zero, add_ps(load_ps(&noise[_7]), load_ps(&location_probability_row[col + _7])), in_7));
		}
	}
	if (cols - col > _4)
	{
		float noise[_4];
		for (; col + _4 - 1 < cols; col += _4)
		{
			vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, _4, noise, 0.0, scale);
			auto _b0_x = load_ps(&bx[col + _0]);
			auto dp_b2_0 = mul_add_ps(_b0_x, _b0_x, _b2_y);
			auto in_0 = cmp_lt_ps(dp_b2_0, r2);
			stream_ps(&location_probability_row[col + _0], blendv_ps(__zero, add_ps(load_ps(&noise[_0]), load_ps(&location_probability_row[col + _0])), in_0));

			auto _b1_x = load_ps(&bx[col + _1]);
			auto dp_b2_1 = mul_add_ps(_b1_x, _b1_x, _b2_y);
			auto in_1 = cmp_lt_ps(dp_b2_1, r2);
			stream_ps(&location_probability_row[col + _1], blendv_ps(__zero, add_ps(load_ps(&noise[_1]), load_ps(&location_probability_row[col + _1])), in_1));

			auto _b2_x = load_ps(&bx[col + _2]);
			auto dp_b2_2 = mul_add_ps(_b2_x, _b2_x, _b2_y);
			auto in_2 = cmp_lt_ps(dp_b2_2, r2);
			stream_ps(&location_probability_row[col + _2], blendv_ps(__zero, add_ps(load_ps(&noise[_2]), load_ps(&location_probability_row[col + _2])), in_2));

			auto _b3_x = load_ps(&bx[col + _3]);
			auto dp_b2_3 = mul_add_ps(_b3_x, _b3_x, _b2_y);
			auto in_3 = cmp_lt_ps(dp_b2_3, r2);
			stream_ps(&location_probability_row[col + _3], blendv_ps(__zero, add_ps(load_ps(&noise[_3]), load_ps(&location_probability_row[col + _3])), in_3));

		}
	}
	if (cols - col > _2)
	{
		float noise[_2];
		for (; col + _2 - 1 < cols; col += _2)
		{
			vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, _2, noise, 0.0, scale);
			auto _b0_x = load_ps(&bx[col + _0]);
			auto dp_b2_0 = mul_add_ps(_b0_x, _b0_x, _b2_y);
			auto in_0 = cmp_lt_ps(dp_b2_0, r2);
			stream_ps(&location_probability_row[col + _0], blendv_ps(__zero, add_ps(load_ps(&noise[_0]), load_ps(&location_probability_row[col + _0])), in_0));

			auto _b1_x = load_ps(&bx[col + _1]);
			auto dp_b2_1 = mul_add_ps(_b1_x, _b1_x, _b2_y);
			auto in_1 = cmp_lt_ps(dp_b2_1, r2);
			stream_ps(&location_probability_row[col + _1], blendv_ps(__zero, add_ps(load_ps(&noise[_1]), load_ps(&location_probability_row[col + _1])), in_1));

		}
	}
	if (cols - col > 0)
	{
		float noise[_1];
		for (; col < cols; col += _1)
		{
			vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, _1, noise, 0.0, scale);
			auto _b0_x = load_ps(&bx[col + _0]);
			auto dp_b2_0 = mul_add_ps(_b0_x, _b0_x, _b2_y);
			auto in_0 = cmp_lt_ps(dp_b2_0, r2);
			stream_ps(&location_probability_row[col + _0], blendv_ps(__zero, add_ps(load_ps(&noise[_0]), load_ps(&location_probability_row[col + _0])), in_0));
		}
	}
}


template <TRN::CPU::Implementation Implementation>
static inline void 	inside_circle_sector(VSLStreamStatePtr &stream, const float &scale, const float &cos_half_angle, const float &ax, const float &ay, const float *bx, const float &by, const float &_r2, const std::size_t &cols,  float *location_probability_row)
{
	//	 vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, 1000, r, 0.0, scale);
	const typename TRN::CPU::Traits<Implementation>::type _a_x = set1_ps(ax);
	const typename TRN::CPU::Traits<Implementation>::type _a_y = set1_ps(ay);
	const typename TRN::CPU::Traits<Implementation>::type _b_y = set1_ps(by);
	const typename TRN::CPU::Traits<Implementation>::type r2 = set1_ps(_r2);
	const typename TRN::CPU::Traits<Implementation>::type cha = set1_ps(cos_half_angle);
	auto __zero = setzero_ps();

	auto _b2_y = sqr_ps(_b_y);
	auto _ab_y = mul_ps(_a_y, _b_y);
	std::size_t col = 0;

	if (cols - col >= _8)
	{
		float noise[_8];

		for (; col + _8 - 1 < cols; col += _8)
		{
			vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, _8, noise, 0.0, scale);

			auto _b0_x = load_ps(&bx[col + _0]);
			auto dp_b2_0 = mul_add_ps(_b0_x, _b0_x, _b2_y);
			auto dp_ab_0 = mul_ps(mul_add_ps(_a_x, _b0_x, _ab_y), rsqrt_ps(dp_b2_0));
			auto in_0 = and_ps(cmp_lt_ps(cha, dp_ab_0), cmp_lt_ps(dp_b2_0, r2));
			stream_ps(&location_probability_row[col + _0], blendv_ps(__zero, add_ps(load_ps(&noise[_0]), load_ps(&location_probability_row[col + _0])), in_0));

			auto _b1_x = load_ps(&bx[col + _1]);
			auto dp_b2_1 = mul_add_ps(_b1_x, _b1_x, _b2_y);
			auto dp_ab_1 = mul_ps(mul_add_ps(_a_x, _b1_x, _ab_y), rsqrt_ps(dp_b2_1));
			auto in_1 =  and_ps(cmp_lt_ps(cha, dp_ab_1), cmp_lt_ps(dp_b2_1, r2));
			stream_ps(&location_probability_row[col + _1], blendv_ps(__zero, add_ps(load_ps(&noise[_1]), load_ps(&location_probability_row[col + _1])), in_1));

			auto _b2_x = load_ps(&bx[col + _2]);
			auto dp_b2_2 = mul_add_ps(_b2_x, _b2_x, _b2_y);
			auto dp_ab_2 = mul_ps(mul_add_ps(_a_x, _b2_x, _ab_y), rsqrt_ps(dp_b2_2));
			auto in_2 =  and_ps(cmp_lt_ps(cha, dp_ab_2), cmp_lt_ps(dp_b2_2, r2));
			stream_ps(&location_probability_row[col + _2], blendv_ps(__zero, add_ps(load_ps(&noise[_2]), load_ps(&location_probability_row[col + _2])), in_2));
		
			auto _b3_x = load_ps(&bx[col + _3]);
			auto dp_b2_3 = mul_add_ps(_b3_x, _b3_x, _b2_y);
			auto dp_ab_3 = mul_ps(mul_add_ps(_a_x, _b3_x, _ab_y), rsqrt_ps(dp_b2_3));
			auto in_3 =  and_ps(cmp_lt_ps(cha, dp_ab_3), cmp_lt_ps(dp_b2_3, r2));
			stream_ps(&location_probability_row[col + _3], blendv_ps(__zero, add_ps(load_ps(&noise[_3]), load_ps(&location_probability_row[col + _3])), in_3));

			auto _b4_x = load_ps(&bx[col + _4]);
			auto dp_b2_4 = mul_add_ps(_b4_x, _b4_x, _b2_y);
			auto dp_ab_4 = mul_ps(mul_add_ps(_a_x, _b4_x, _ab_y), rsqrt_ps(dp_b2_4));
			auto in_4 =  and_ps(cmp_lt_ps(cha, dp_ab_4), cmp_lt_ps(dp_b2_4, r2));
			stream_ps(&location_probability_row[col + _4], blendv_ps(__zero, add_ps(load_ps(&noise[_4]), load_ps(&location_probability_row[col + _4])), in_4));

			auto _b5_x = load_ps(&bx[col + _5]);
			auto dp_b2_5 = mul_add_ps(_b5_x, _b5_x, _b2_y);
			auto dp_ab_5 = mul_ps(mul_add_ps(_a_x, _b5_x, _ab_y), rsqrt_ps(dp_b2_5));
			auto in_5 =  and_ps(cmp_lt_ps(cha, dp_ab_5), cmp_lt_ps(dp_b2_5, r2));
			stream_ps(&location_probability_row[col + _5], blendv_ps(__zero, add_ps(load_ps(&noise[_5]), load_ps(&location_probability_row[col + _5])),  in_5));

			auto _b6_x = load_ps(&bx[col + _6]);
			auto dp_b2_6 = mul_add_ps(_b6_x, _b6_x, _b2_y);
			auto dp_ab_6 = mul_ps(mul_add_ps(_a_x, _b6_x, _ab_y), rsqrt_ps(dp_b2_6));
			auto in_6 =  and_ps(cmp_lt_ps(cha, dp_ab_6), cmp_lt_ps(dp_b2_6, r2));
			stream_ps(&location_probability_row[col + _6], blendv_ps(__zero, add_ps(load_ps(&noise[_6]), load_ps(&location_probability_row[col + _6])), in_6));

			auto _b7_x = load_ps(&bx[col + _7]);
			auto dp_b2_7 = mul_add_ps(_b7_x, _b7_x, _b2_y);
			auto dp_ab_7 = mul_ps(mul_add_ps(_a_x, _b7_x, _ab_y), rsqrt_ps(dp_b2_7));
			auto in_7 = and_ps(cmp_lt_ps(cha, dp_ab_7), cmp_lt_ps(dp_b2_7, r2));
			stream_ps(&location_probability_row[col + _7], blendv_ps(__zero, add_ps(load_ps(&noise[_7]), load_ps(&location_probability_row[col + _7])), in_7));
		}
	}
	if (cols - col > _4)
	{
		float noise[_4];
		for (; col + _4 - 1 < cols; col += _4)
		{
			vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, _4, noise, 0.0, scale);
			auto _b0_x = load_ps(&bx[col + _0]);
			auto dp_b2_0 = mul_add_ps(_b0_x, _b0_x, _b2_y);
			auto dp_ab_0 = mul_ps(mul_add_ps(_a_x, _b0_x, _ab_y), rsqrt_ps(dp_b2_0));
			auto in_0 = and_ps(cmp_lt_ps(cha, dp_ab_0), cmp_lt_ps(dp_b2_0, r2));
			stream_ps(&location_probability_row[col + _0], blendv_ps(__zero, add_ps(load_ps(&noise[_0]), load_ps(&location_probability_row[col + _0])), in_0));

			auto _b1_x = load_ps(&bx[col + _1]);
			auto dp_b2_1 = mul_add_ps(_b1_x, _b1_x, _b2_y);
			auto dp_ab_1 = mul_ps(mul_add_ps(_a_x, _b1_x, _ab_y), rsqrt_ps(dp_b2_1));
			auto in_1 = and_ps(cmp_lt_ps(cha, dp_ab_1), cmp_lt_ps(dp_b2_1, r2));
			stream_ps(&location_probability_row[col + _1], blendv_ps(__zero, add_ps(load_ps(&noise[_1]), load_ps(&location_probability_row[col + _1])), in_1));

			auto _b2_x = load_ps(&bx[col + _2]);
			auto dp_b2_2 = mul_add_ps(_b2_x, _b2_x, _b2_y);
			auto dp_ab_2 = mul_ps(mul_add_ps(_a_x, _b2_x, _ab_y), rsqrt_ps(dp_b2_2));
			auto in_2 = and_ps(cmp_lt_ps(cha, dp_ab_2), cmp_lt_ps(dp_b2_2, r2));
			stream_ps(&location_probability_row[col + _2], blendv_ps(__zero, add_ps(load_ps(&noise[_2]), load_ps(&location_probability_row[col + _2])), in_2));

			auto _b3_x = load_ps(&bx[col + _3]);
			auto dp_b2_3 = mul_add_ps(_b3_x, _b3_x, _b2_y);
			auto dp_ab_3 = mul_ps(mul_add_ps(_a_x, _b3_x, _ab_y), rsqrt_ps(dp_b2_3));
			auto in_3 = and_ps(cmp_lt_ps(cha, dp_ab_3), cmp_lt_ps(dp_b2_3, r2));
			stream_ps(&location_probability_row[col + _3], blendv_ps(__zero, add_ps(load_ps(&noise[_3]), load_ps(&location_probability_row[col + _3])), in_3));
		}
	}
	if (cols - col > _2)
	{
		float noise[_2];
		for (; col + _2 - 1 < cols; col += _2)
		{
			vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, _2, noise, 0.0, scale);
			auto _b0_x = load_ps(&bx[col + _0]);
			auto dp_b2_0 = mul_add_ps(_b0_x, _b0_x, _b2_y);
			auto dp_ab_0 = mul_ps(mul_add_ps(_a_x, _b0_x, _ab_y), rsqrt_ps(dp_b2_0));
			auto in_0 = and_ps(cmp_lt_ps(cha, dp_ab_0), cmp_lt_ps(dp_b2_0, r2));
			stream_ps(&location_probability_row[col + _0], blendv_ps(__zero, add_ps(load_ps(&noise[_0]), load_ps(&location_probability_row[col + _0])), in_0));

			auto _b1_x = load_ps(&bx[col + _1]);
			auto dp_b2_1 = mul_add_ps(_b1_x, _b1_x, _b2_y);
			auto dp_ab_1 = mul_ps(mul_add_ps(_a_x, _b1_x, _ab_y), rsqrt_ps(dp_b2_1));
			auto in_1 = and_ps(cmp_lt_ps(cha, dp_ab_1), cmp_lt_ps(dp_b2_1, r2));
			stream_ps(&location_probability_row[col + _1], blendv_ps(__zero, add_ps(load_ps(&noise[_1]), load_ps(&location_probability_row[col + _1])), in_1));
		}
	}
	 if (cols - col > 0)
	{
		float noise[_1];
		for (; col  < cols; col += _1)
		{
			vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, _1, noise, 0.0, scale);
			auto _b0_x = load_ps(&bx[col + _0]);
			auto dp_b2_0 = mul_add_ps(_b0_x, _b0_x, _b2_y);
			auto dp_ab_0 = mul_ps(mul_add_ps(_a_x, _b0_x, _ab_y), rsqrt_ps(dp_b2_0));
			auto in_0 = and_ps(cmp_lt_ps(cha, dp_ab_0), cmp_lt_ps(dp_b2_0, r2));
			stream_ps(&location_probability_row[col + _0], blendv_ps(__zero, add_ps(load_ps(&noise[_0]), load_ps(&location_probability_row[col + _0])), in_0));
		}
	}
}

template <TRN::CPU::Implementation Implementation>
static inline void location_hypothesis(const float *firing_rate_row, const std::size_t &cols,
	const typename TRN::CPU::Traits<Implementation>::type  &__prediction,
	const typename TRN::CPU::Traits<Implementation>::type  &__inv_sigma2,
	float *hypothesis_row)
{
	std::size_t col = 0;
	if (cols - col >= _8)
	{
		for (; col + _8 - 1 < cols; col += _8)
		{
			auto __h0 = mul_ps(sqr_ps(sub_ps(__prediction, load_ps(&firing_rate_row[col + _0]))), __inv_sigma2);
			stream_ps(&hypothesis_row[col + _0], __h0);
			auto __h1 = mul_ps(sqr_ps(sub_ps(__prediction, load_ps(&firing_rate_row[col + _1]))), __inv_sigma2);
			stream_ps(&hypothesis_row[col + _1], __h1);
			auto __h2 = mul_ps(sqr_ps(sub_ps(__prediction, load_ps(&firing_rate_row[col + _2]))), __inv_sigma2);
			stream_ps(&hypothesis_row[col + _2], __h2);
			auto __h3 = mul_ps(sqr_ps(sub_ps(__prediction, load_ps(&firing_rate_row[col + _3]))), __inv_sigma2);
			stream_ps(&hypothesis_row[col + _3], __h3);			
			auto __h4 = mul_ps(sqr_ps(sub_ps(__prediction, load_ps(&firing_rate_row[col + _4]))), __inv_sigma2);
			stream_ps(&hypothesis_row[col + _4], __h4);
			auto __h5 = mul_ps(sqr_ps(sub_ps(__prediction, load_ps(&firing_rate_row[col + _5]))), __inv_sigma2);
			stream_ps(&hypothesis_row[col + _5], __h5);
			auto __h6 = mul_ps(sqr_ps(sub_ps(__prediction, load_ps(&firing_rate_row[col + _6]))), __inv_sigma2);
			stream_ps(&hypothesis_row[col + _6], __h6);
			auto __h7 = mul_ps(sqr_ps(sub_ps(__prediction, load_ps(&firing_rate_row[col + _7]))), __inv_sigma2);
			stream_ps(&hypothesis_row[col + _7], __h7);
		}
	}
	if (cols - col >= _4)
	{
		for (; col + _4 - 1 < cols; col += _4)
		{
			auto __h0 = mul_ps(sqr_ps(sub_ps(__prediction, load_ps(&firing_rate_row[col + _0]))), __inv_sigma2);
			stream_ps(&hypothesis_row[col + _0], __h0);
			auto __h1 = mul_ps(sqr_ps(sub_ps(__prediction, load_ps(&firing_rate_row[col + _1]))), __inv_sigma2);
			stream_ps(&hypothesis_row[col + _1], __h1);
			auto __h2 = mul_ps(sqr_ps(sub_ps(__prediction, load_ps(&firing_rate_row[col + _2]))), __inv_sigma2);
			stream_ps(&hypothesis_row[col + _2], __h2);
			auto __h3 = mul_ps(sqr_ps(sub_ps(__prediction, load_ps(&firing_rate_row[col + _3]))), __inv_sigma2);
			stream_ps(&hypothesis_row[col + _3], __h3);
		}
	}
	if (cols - col >= _2)
	{
		for (; col + _2 - 1 < cols; col += _2)
		{
			auto __h0 = mul_ps(sqr_ps(sub_ps(__prediction, load_ps(&firing_rate_row[col + _0]))), __inv_sigma2);
			stream_ps(&hypothesis_row[col + _0], __h0);
			auto __h1 = mul_ps(sqr_ps(sub_ps(__prediction, load_ps(&firing_rate_row[col + _1]))), __inv_sigma2);
			stream_ps(&hypothesis_row[col + _1], __h1);
		}
	}
	 if (cols - col > 0)
	{
		for (; col  < cols; col += _1)
		{
			auto __h0 = mul_ps(sqr_ps(sub_ps(__prediction, load_ps(&firing_rate_row[col + _0]))), __inv_sigma2);
			stream_ps(&hypothesis_row[col + _0], __h0);
		}
	}
}
template <TRN::CPU::Implementation Implementation>
void TRN::CPU::Algorithm<Implementation>::decode_most_probable_location(
	const std::size_t &batch_size, const std::size_t &stimulus_size,
	const std::size_t &roi_row_begin, const std::size_t &roi_row_end,
	const std::size_t &roi_col_begin, const std::size_t &roi_col_end,
	const std::size_t &order,
	const unsigned long &seed, const float &sigma, const float &scale, const float &radius, const float &cos_half_angle,
	const float *firing_rate_map, const std::size_t &firing_rate_map_stride,
	const float *coefficients, const std::size_t &coefficients_stride,
	const float *x_grid, const std::size_t &x_grid_stride,
	const float *y_grid, const std::size_t &y_grid_stride,
	const float **batched_previous_position, const std::size_t *batched_previous_position_strides,
	const float **batched_current_position, const std::size_t *batched_current_position_strides,
	const float **batched_prediction, const std::size_t *batched_prediction_strides,
	float *hypothesis_scale, const std::size_t &hypothesis_scale_stride,
	float **batched_x_grid_centered, const std::size_t *batched_x_grid_centered_strides,
	float **batched_y_grid_centered, const std::size_t *batched_y_grid_centered_strides,
	float **batched_direction, const std::size_t *batched_direction_strides,
	float **batched_decoded_position, const std::size_t *batched_decoded_position_strides
)
{

}
template <TRN::CPU::Implementation Implementation>
void TRN::CPU::Algorithm<Implementation>::place_cell_location_probability(
	const std::size_t &batch_size, const std::size_t &place_cells_number, 
	const std::size_t &rows_begin, const std::size_t &rows_end,
	const std::size_t &cols_begin, const std::size_t &cols_end,
	const float &sigma,
	const float ** firing_rate_map, const std::size_t *firing_rate_map_rows, const std::size_t *firing_rate_map_cols, const std::size_t *firing_rate_map_strides,
	float **scale, const std::size_t *scale_rows, const std::size_t *scale_cols, const std::size_t *scale_strides,
	const float **prediction, const std::size_t *prediction_rows, const std::size_t *prediction_cols, const std::size_t *prediction_strides,
	float *** hypothesis_map, const std::size_t **hypothesis_map_rows, const std::size_t **hypothesis_map_cols, const std::size_t **hypothesis_map_strides,
	float ** location_probability, const std::size_t *location_probability_rows, const std::size_t *location_probability_cols, const std::size_t *location_probability_strides)
{
	auto cols_offset = round_down<Implementation>(cols_begin);
	auto cols_span = round_up<Implementation>(cols_end, location_probability_cols[0]) - cols_offset;
	auto rows_span = rows_end - rows_begin;
	const float _inv_sigma2 = -1.0f / (sigma*sigma);
	const auto ___inv_sigma2 = set1_ps(_inv_sigma2);

	int K = batch_size * place_cells_number;
#pragma omp parallel for
	for (int k = 0; k < K; k++)
	{
		int batch = k % batch_size;
		int place_cell = k / batch_size;
		const std::size_t hypothesis_map_stride = hypothesis_map_strides[batch][place_cell];
		const std::size_t firing_rate_map_stride = firing_rate_map_strides[place_cell];
		auto firing_rate_k = firing_rate_map[place_cell];
		auto hypothesis_k = hypothesis_map[batch][place_cell];

		const float &p = prediction[batch][place_cell];
		const auto &__prediction = set1_ps(p);
		float sum = 0.0f;
		for (std::size_t roi_row = 0; roi_row < rows_span; roi_row++)
		{
			auto row = roi_row + rows_begin;
			auto firing_rate_row = &firing_rate_k[row * firing_rate_map_stride + cols_offset];
			auto hypothesis_row = &hypothesis_k[row * hypothesis_map_stride + cols_offset];

			location_hypothesis<Implementation>(firing_rate_row, cols_span, __prediction, ___inv_sigma2, hypothesis_row);
			exp_v(cols_span, hypothesis_row, hypothesis_row);
			sum += cblas_sasum(cols_span, hypothesis_row, 1);
		}
		if (sum > 0.0f)
			scale[batch][place_cell] = 1.0f / (sum * (float)place_cells_number);
		else
			scale[batch][place_cell] = 0.0f;
	}



	const std::size_t place_cells_number_range = place_cells_number / 2;
	const std::size_t place_cells_number_remaining = place_cells_number - place_cells_number_range * 2;

	K = place_cells_number_range * batch_size;

#pragma omp parallel for
	for (int k = 0; k < K; k++)
	{
		int batch = k % batch_size;
		int place_cell = k / batch_size;

		auto place_cell_a = place_cell;
		auto place_cell_b = place_cells_number_range + place_cell;
		auto hypothesis_a = hypothesis_map[batch][place_cell_a];
		auto hypothesis_b = hypothesis_map[batch][place_cell_b];
		auto stride_a = hypothesis_map_strides[batch][place_cell_a];
		auto stride_b = hypothesis_map_strides[batch][place_cell_b];
		const auto scale_a = set1_ps(scale[batch][place_cell_a]);
		const auto scale_b = set1_ps(scale[batch][place_cell_b]);
		for (std::size_t roi_row = 0; roi_row < rows_span; roi_row++)
		{
			auto row = roi_row + rows_begin;
			auto hypothesis_a_row = &hypothesis_a[row * stride_a + cols_offset];
			auto hypothesis_b_row = &hypothesis_b[row * stride_b + cols_offset];
			weighted_sum<Implementation>(hypothesis_a_row, scale_a, hypothesis_b_row, scale_b, cols_span, hypothesis_a_row);
		}
	}

	if (place_cells_number_range >= 2)
	{

		for (std::size_t range = place_cells_number_range / 2; range > 1; range /= 2)
		{
			K = range * batch_size;
#pragma omp parallel for
			for (int k = 0; k < K; k++)
			{
				int batch = k % batch_size;
				int place_cell = k / batch_size;
				auto place_cell_a = place_cell;
				auto place_cell_b = range + place_cell;
				auto hypothesis_a = hypothesis_map[batch][place_cell_a];
				auto hypothesis_b = hypothesis_map[batch][place_cell_b];
				auto stride_a = hypothesis_map_strides[batch][place_cell_a];
				auto stride_b = hypothesis_map_strides[batch][place_cell_b];
				for (std::size_t roi_row = 0; roi_row < rows_span; roi_row++)
				{
					auto row = roi_row + rows_begin;
					auto hypothesis_a_row = &hypothesis_a[row * stride_a + cols_offset];
					auto hypothesis_b_row = &hypothesis_b[row * stride_b + cols_offset];

					sum<Implementation>(hypothesis_a_row, hypothesis_b_row, cols_span, hypothesis_a_row);
				}
			}
		}
		{
			K = batch_size * rows_span;
#pragma omp parallel for
			for (int k = 0; k < K; k++)
			{
				int batch = k % batch_size;
				int row = k / batch_size + rows_begin;
				auto hypothesis_a = hypothesis_map[batch][0];
				auto hypothesis_b = hypothesis_map[batch][1];
				auto stride_a = hypothesis_map_strides[batch][0];
				auto stride_b = hypothesis_map_strides[batch][1];
				auto hypothesis_a_row = &hypothesis_a[row * stride_a + cols_offset];
				auto hypothesis_b_row = &hypothesis_b[row * stride_b + cols_offset];
				auto location_probability_row = &location_probability[batch][row * location_probability_strides[batch]];

				sum<Implementation>(hypothesis_a_row, hypothesis_b_row, cols_span, location_probability_row);
			}
		}
	}

	if (place_cells_number_remaining > 0)
	{
		K = batch_size * rows_span;
#pragma omp parallel for
		for (int k = 0; k < K; k++)
		{
			int batch = k % batch_size;
			int row = k / batch_size + rows_begin;
			auto hypothesis_k = hypothesis_map[batch][place_cells_number_range];
			auto scale_k = set1_ps(scale[batch][place_cells_number_range]);
			auto hypothesis_k_row = &hypothesis_k[row * hypothesis_map_strides[batch][place_cells_number_range] + cols_offset];
			auto location_probability_row = &location_probability[batch][row* location_probability_strides[batch] + cols_offset];

			weighted_acc<Implementation>(hypothesis_k_row, scale_k, cols_span, location_probability_row);
		}
	}
}

template <TRN::CPU::Implementation Implementation>
static std::size_t round_down(const std::size_t &offset)
{
	return (offset / TRN::CPU::Traits<Implementation>::step) *  TRN::CPU::Traits<Implementation>::step;
}

template <TRN::CPU::Implementation Implementation>
static std::size_t round_up(const std::size_t &offset, const std::size_t &max_offset)
{
	return min(max_offset, (round_down<Implementation>(offset + TRN::CPU::Traits<Implementation>::step - 1)));
}
/*
template <TRN::CPU::Implementation Implementation>
 void TRN::CPU::Algorithm<Implementation>::restrict_to_reachable_locations(
	 const std::size_t &batch_size, const std::size_t &place_cells_number, 
	 const std::size_t &rows_begin, const std::size_t &rows_end,
	 const std::size_t &cols_begin, const std::size_t &cols_end,
	 const float &radius, const float &cos_half_angle, const float &scale, const unsigned long &seed,
	 const float *x_grid, const std::size_t &x_grid_rows, const std::size_t &x_grid_cols, const std::size_t &x_grid_stride,
	 const float *y_grid, const std::size_t &y_grid_rows, const std::size_t &y_grid_cols, const std::size_t &y_grid_stride,
	 const float **batched_previous_location, const std::size_t *batched_previous_location_rows, const std::size_t *batched_previous_location_cols, const std::size_t *batched_previous_location_stride,
	 const float **batched_current_location, const std::size_t *batched_current_location_rows, const std::size_t *batched_current_location_cols, const std::size_t *batched_current_location_stride,
	 float **batched_direction, const std::size_t *batched_direction_rows, const std::size_t *batched_direction_cols, const std::size_t *batched_direction_stride,
	 float **batched_x_grid_centered, const std::size_t *batched_x_grid_centered_rows, const std::size_t *batched_x_grid_centered_cols, const std::size_t *batched_x_grid_centered_stride,
	 float **batched_y_grid_centered, const std::size_t *batched_y_grid_centered_rows, const std::size_t *batched_y_grid_centered_cols, const std::size_t *batched_y_grid_centered_stride,
	 float  **batched_location_probability, const std::size_t *batched_location_probability_rows, const std::size_t *batched_location_probability_cols, const std::size_t *batched_location_probability_strides
 )
{
	
	
	 auto cols_offset = round_down<Implementation>(cols_begin);
	 auto cols_span = round_up<Implementation>(cols_end, x_grid_cols) - cols_offset;
	 auto rows_span = rows_end - rows_begin;

	 auto r2 = radius * radius;
#pragma omp parallel for
	 for (int batch = 0; batch < batch_size; batch++)
	 {
		 auto cx = batched_current_location[batch][0];
		 auto cy = batched_current_location[batch][1];
		 auto px = batched_previous_location[batch][0];
		 auto py = batched_previous_location[batch][1];
		 auto dx = cx - px;
		 auto dy = cy - py;
		 auto dh = std::sqrtf(dx * dx + dy *dy);
		 auto inv_norm = dh > 0.0f ? 1.0f / dh : 0.0f;
		 
		 batched_direction[batch][0] = dx *inv_norm;
		 batched_direction[batch][1] = dy *inv_norm;

		 diff<Implementation>(set1_ps(cy), y_grid, y_grid_cols, batched_y_grid_centered[batch]);
		 diff<Implementation>(set1_ps(cx), x_grid, x_grid_cols, batched_x_grid_centered[batch]);

	
	 }


	 int K = batch_size * rows_span;
#pragma omp parallel for
	 for (int k = 0; k < K; k++)
	 {
		 auto tid = omp_get_thread_num();
		 int batch = k % batch_size;
		 int roi_row = k / batch_size;
		 auto row = roi_row + rows_begin;
	
		 auto location_probability_row = &batched_location_probability[batch][row * batched_location_probability_strides[batch]];
		
		 auto by = batched_y_grid_centered[batch][row];

		 auto ax = batched_direction[batch][0];
		 auto ay = batched_direction[batch][1];
	
		 if (cos_half_angle < 1.0f && ax *ax + ay * ay > 0.0f)
			inside_circle_sector<Implementation>(streams[tid],  scale, cos_half_angle, ax, ay, batched_x_grid_centered[batch] + cols_offset,by, r2, cols_span,location_probability_row);
		 else
			inside_circle<Implementation>(streams[tid], scale, batched_x_grid_centered[batch] + cols_offset, by, r2, cols_span, location_probability_row);
	 }

	 

}
*/
/*template <TRN::CPU::Implementation Implementation>
void TRN::CPU::Algorithm<Implementation>::draw_probable_location(const std::size_t &batch_size, const std::size_t &rows, const std::size_t &cols,
	const float *x_grid, const std::size_t &x_grid_rows, const std::size_t &x_grid_cols, const std::size_t &x_grid_stride,
	const float *y_grid, const std::size_t &y_grid_rows, const std::size_t &y_grid_cols, const std::size_t &y_grid_stride,
	const float  **batched_location_probability, const std::size_t *batched_location_probability_rows, const std::size_t *batched_location_probability_cols, const std::size_t *batched_location_probability_strides,
	float **batched_reduced_location_probability, const std::size_t *batched_reduced_location_probability_rows, const std::size_t *batched_reduced_location_probability_cols, const std::size_t *batched_reduced_location_probability_stride,
	float **batched_row_cumsum, const std::size_t *batched_row_cumsum_rows, const std::size_t *batched_row_cumsum_cols, const std::size_t *batched_row_cumsum_stride,
	float **batched_col_cumsum, const std::size_t *batched_col_cumsum_rows, const std::size_t *batched_col_cumsum_cols, const std::size_t *batched_col_cumsum_stride,
	float **batched_predicted_location, const std::size_t *batched_predicted_location_rows, const std::size_t *batched_predicted_location_cols, const std::size_t *batched_predicted_location_strides)
{
 }*/


static inline float col_to_x(const  std::size_t &col, const std::size_t &cols, const float &x_min, const float &x_range)
{
	return ((col) / (float)(cols - 1)) * x_range + x_min;
}
static inline float row_to_y(const  std::size_t &row, const  std::size_t &rows, const float &y_min, const float &y_range)
{
	return ((row) / (float)(rows - 1)) * y_range + y_min;
}

template <TRN::CPU::Implementation Implementation>
void TRN::CPU::Algorithm<Implementation>::assign_most_probable_location(
	const std::size_t &batch_size, const std::size_t &rows, const std::size_t &cols,
	const std::size_t *roi_row_begin, const std::size_t *roi_row_end, const std::size_t *roi_col_begin, const std::size_t *roi_col_end,
	const float &x_min, const float &x_range, const float &y_min, const float &y_range,
	const int **batched_argmax, const std::size_t *batched_location_probability_strides,
	float **batched_predicted_location)
{
#pragma omp parallel for
	for (int batch = 0; batch < batch_size; batch++)
	{
		auto predicted_location = batched_predicted_location[batch];
		auto idx = *batched_argmax[batch];
		if (idx == 0)
		{
			predicted_location[0] = 0.0f;
			predicted_location[1] = 0.0f;
		}
		else
		{
			idx--;
			const std::size_t stride = batched_location_probability_strides[batch];
			std::size_t roi_col = idx % stride;
			std::size_t roi_row = idx / stride;
			auto row = roi_row + roi_row_begin[batch];
			auto col = roi_col + roi_col_begin[batch];
			auto x = col_to_x(col, cols, x_min, x_range);
			auto y = row_to_y(row, rows, y_min, y_range);
			predicted_location[0] = x;
			predicted_location[1] = y;
		}
	}
}


template <TRN::CPU::Implementation Implementation>
void TRN::CPU::Algorithm<Implementation>::select_most_probable_location(const std::size_t &batch_size, const std::size_t &rows, const std::size_t &cols,
	const std::size_t *roi_row_begin, const std::size_t *roi_row_end, const std::size_t *roi_col_begin, const std::size_t *roi_col_end,
	const float *x_grid, const std::size_t &x_grid_rows, const std::size_t &x_grid_cols, const std::size_t &x_grid_stride,
	const float *y_grid, const std::size_t &y_grid_rows, const std::size_t &y_grid_cols, const std::size_t &y_grid_stride,
	const float  **batched_location_probability, const std::size_t *batched_location_probability_rows, const std::size_t *batched_location_probability_cols, const std::size_t *batched_location_probability_strides,
	int **argmax
)
{
#pragma omp parallel for
	for (int batch = 0; batch < batch_size; batch++)
	{
		const float *location_probability = batched_location_probability[batch];
		const std::size_t rows = batched_location_probability_rows[batch];
		const std::size_t stride = batched_location_probability_strides[batch];
		*argmax[batch] = cblas_isamax(rows * stride, location_probability, 1);
	}
}
#define PREFETCH_T0(addr,nrOfBytesAhead) _mm_prefetch(((char *)(addr))+nrOfBytesAhead,_MM_HINT_T0)
#define PREFETCH_T1(addr,nrOfBytesAhead) _mm_prefetch(((char *)(addr))+nrOfBytesAhead,_MM_HINT_T1)
template <TRN::CPU::Implementation Implementation>
static inline float  dot_product(const float *a, const float *x, const std::size_t &cols)
{
	std::size_t col = 0;
	auto y0 = setzero_ps();
	if (cols - col >= _8)
	{
		auto y1 = setzero_ps();
		auto y2 = setzero_ps();
		auto y3 = setzero_ps();
		auto y4 = setzero_ps();
		auto y5 = setzero_ps();
		auto y6 = setzero_ps();
		auto y7 = setzero_ps();

		for (; col + _8 - 1 < cols; col += _8)
		{
			y0 = mul_add_ps(load_ps(&a[col + _0]), load_ps(&x[col + _0]), y0);
			y1 = mul_add_ps(load_ps(&a[col + _1]), load_ps(&x[col + _1]), y1);
			y2 = mul_add_ps(load_ps(&a[col + _2]), load_ps(&x[col + _2]), y2);
			y3 = mul_add_ps(load_ps(&a[col + _3]), load_ps(&x[col + _3]), y3);
			y4 = mul_add_ps(load_ps(&a[col + _4]), load_ps(&x[col + _4]), y4);
			y5 = mul_add_ps(load_ps(&a[col + _5]), load_ps(&x[col + _5]), y5);
			y6 = mul_add_ps(load_ps(&a[col + _6]), load_ps(&x[col + _6]), y6);
			y7 = mul_add_ps(load_ps(&a[col + _7]), load_ps(&x[col + _7]), y7);
		}
		y0 = add_ps(y0, y1);
		y2 = add_ps(y2, y3);
		y4 = add_ps(y4, y5);
		y6 = add_ps(y6, y7);
		y0 = add_ps(y0, y2);
		y4 = add_ps(y4, y6);
		y0 = add_ps(y0, y4);
	}
	if (cols - col >= _4)
	{
		auto y1 = setzero_ps();
		auto y2 = setzero_ps();
		auto y3 = setzero_ps();
		for (; col + _4 - 1 < cols; col += _4)
		{
			y0 = mul_add_ps(load_ps(&a[col + _0]), load_ps(&x[col + _0]), y0);
			y1 = mul_add_ps(load_ps(&a[col + _1]), load_ps(&x[col + _1]), y1);
			y2 = mul_add_ps(load_ps(&a[col + _2]), load_ps(&x[col + _2]), y2);
			y3 = mul_add_ps(load_ps(&a[col + _3]), load_ps(&x[col + _3]), y3);
		}
		y0 = add_ps(y0, y1);
		y2 = add_ps(y2, y3);
		y0 = add_ps(y0, y2);
	}
	if (cols - col >= _2)
	{
		auto y1 = setzero_ps();
		for (; col + _2 - 1 < cols; col += _2)
		{
			y0 = mul_add_ps(load_ps(&a[col + _0]), load_ps(&x[col + _0]), y0);
			y1 = mul_add_ps(load_ps(&a[col + _1]), load_ps(&x[col + _1]), y1);
		}
		y0 = add_ps(y0, y1);
	}
	if (cols - col >= 0)
	{
		for (; col < cols; col += _1)
		{
			y0 = mul_add_ps(load_ps(&a[col + _0]), load_ps(&x[col + _0]), y0);
		}
	}

	return hsum_ps(y0);
}

template <TRN::CPU::Implementation Implementation>
static inline float  dot_product_sub(
	const 	typename TRN::CPU::Traits<Implementation>::type &gx,
	const 	typename TRN::CPU::Traits<Implementation>::type &gy,
	const float *cx, const float *cy, const float *w, const float *x, const std::size_t &cols)
{
	std::size_t col = 0;
	auto y0 = setzero_ps();
	if (cols - col >= _8)
	{
		auto y1 = setzero_ps();
		auto y2 = setzero_ps();
		auto y3 = setzero_ps();
		auto y4 = setzero_ps();
		auto y5 = setzero_ps();
		auto y6 = setzero_ps();
		auto y7 = setzero_ps();

		for (; col + _8 - 1 < cols; col += _8)
		{
			auto v0 = sub_ps(exp_ps(mul_ps(load_ps(&w[col + _0]), add_ps(sqr_ps(sub_ps(gx, load_ps(&cx[col + _0]))), sqr_ps(sub_ps(gy, load_ps(&cy[col + _0])))))), load_ps(&x[col + _0]));
			auto v1 = sub_ps(exp_ps(mul_ps(load_ps(&w[col + _1]), add_ps(sqr_ps(sub_ps(gx, load_ps(&cx[col + _1]))), sqr_ps(sub_ps(gy, load_ps(&cy[col + _1])))))), load_ps(&x[col + _1]));
			auto v2 = sub_ps(exp_ps(mul_ps(load_ps(&w[col + _2]), add_ps(sqr_ps(sub_ps(gx, load_ps(&cx[col + _2]))), sqr_ps(sub_ps(gy, load_ps(&cy[col + _2])))))), load_ps(&x[col + _2]));
			auto v3 = sub_ps(exp_ps(mul_ps(load_ps(&w[col + _3]), add_ps(sqr_ps(sub_ps(gx, load_ps(&cx[col + _3]))), sqr_ps(sub_ps(gy, load_ps(&cy[col + _3])))))), load_ps(&x[col + _3]));
			auto v4 = sub_ps(exp_ps(mul_ps(load_ps(&w[col + _4]), add_ps(sqr_ps(sub_ps(gx, load_ps(&cx[col + _4]))), sqr_ps(sub_ps(gy, load_ps(&cy[col + _4])))))), load_ps(&x[col + _4]));
			auto v5 = sub_ps(exp_ps(mul_ps(load_ps(&w[col + _5]), add_ps(sqr_ps(sub_ps(gx, load_ps(&cx[col + _5]))), sqr_ps(sub_ps(gy, load_ps(&cy[col + _5])))))), load_ps(&x[col + _5]));
			auto v6 = sub_ps(exp_ps(mul_ps(load_ps(&w[col + _6]), add_ps(sqr_ps(sub_ps(gx, load_ps(&cx[col + _6]))), sqr_ps(sub_ps(gy, load_ps(&cy[col + _6])))))), load_ps(&x[col + _6]));
			auto v7 = sub_ps(exp_ps(mul_ps(load_ps(&w[col + _7]), add_ps(sqr_ps(sub_ps(gx, load_ps(&cx[col + _7]))), sqr_ps(sub_ps(gy, load_ps(&cy[col + _7])))))), load_ps(&x[col + _7]));

			y0 = mul_add_ps(v0, v0, y0);
			y1 = mul_add_ps(v1, v1, y1);
			y2 = mul_add_ps(v2, v2, y2);
			y3 = mul_add_ps(v3, v3, y3);
			y4 = mul_add_ps(v4, v4, y4);
			y5 = mul_add_ps(v5, v5, y5);
			y6 = mul_add_ps(v6, v6, y6);
			y7 = mul_add_ps(v7, v7, y7);
		}
		y0 = add_ps(y0, y1);
		y2 = add_ps(y2, y3);
		y4 = add_ps(y4, y5);
		y6 = add_ps(y6, y7);
		y0 = add_ps(y0, y2);
		y4 = add_ps(y4, y6);
		y0 = add_ps(y0, y4);
	}
	if (cols - col >= _4)
	{
		auto y1 = setzero_ps();
		auto y2 = setzero_ps();
		auto y3 = setzero_ps();

		for (; col + _4 - 1 < cols; col += _4)
		{
			auto v0 = sub_ps(exp_ps(mul_ps(load_ps(&w[col + _0]), add_ps(sqr_ps(sub_ps(gx, load_ps(&cx[col + _0]))), sqr_ps(sub_ps(gy, load_ps(&cy[col + _0])))))), load_ps(&x[col + _0]));
			auto v1 = sub_ps(exp_ps(mul_ps(load_ps(&w[col + _1]), add_ps(sqr_ps(sub_ps(gx, load_ps(&cx[col + _1]))), sqr_ps(sub_ps(gy, load_ps(&cy[col + _1])))))), load_ps(&x[col + _1]));
			auto v2 = sub_ps(exp_ps(mul_ps(load_ps(&w[col + _2]), add_ps(sqr_ps(sub_ps(gx, load_ps(&cx[col + _2]))), sqr_ps(sub_ps(gy, load_ps(&cy[col + _2])))))), load_ps(&x[col + _2]));
			auto v3 = sub_ps(exp_ps(mul_ps(load_ps(&w[col + _3]), add_ps(sqr_ps(sub_ps(gx, load_ps(&cx[col + _3]))), sqr_ps(sub_ps(gy, load_ps(&cy[col + _3])))))), load_ps(&x[col + _3]));

			y0 = mul_add_ps(v0, v0, y0);
			y1 = mul_add_ps(v1, v1, y1);
			y2 = mul_add_ps(v2, v2, y2);
			y3 = mul_add_ps(v3, v3, y3);
		}
		y0 = add_ps(y0, y1);
		y2 = add_ps(y2, y3);
		y0 = add_ps(y0, y2);
	}
	if (cols - col >= _2)
	{
		auto y1 = setzero_ps();

		for (; col + _2 - 1 < cols; col += _2)
		{
			auto v0 = sub_ps(exp_ps(mul_ps(load_ps(&w[col + _0]), add_ps(sqr_ps(sub_ps(gx, load_ps(&cx[col + _0]))), sqr_ps(sub_ps(gy, load_ps(&cy[col + _0])))))), load_ps(&x[col + _0]));
			auto v1 = sub_ps(exp_ps(mul_ps(load_ps(&w[col + _1]), add_ps(sqr_ps(sub_ps(gx, load_ps(&cx[col + _1]))), sqr_ps(sub_ps(gy, load_ps(&cy[col + _1])))))), load_ps(&x[col + _1]));

			y0 = mul_add_ps(v0, v0, y0);
			y1 = mul_add_ps(v1, v1, y1);
		}
		y0 = add_ps(y0, y1);
	}
	if (cols - col >= 0)
	{
		//typename TRN::CPU::Traits<Implementation>::type a[1];
		for (; col < cols; col += _1)
		{
			auto v0 = sub_ps(exp_ps(mul_ps(load_ps(&w[col + _0]), add_ps(sqr_ps(sub_ps(gx, load_ps(&cx[col + _0]))), sqr_ps(sub_ps(gy, load_ps(&cy[col + _0])))))), load_ps(&x[col + _0]));

			y0 = mul_add_ps(v0, v0, y0);
		}
	}

	return hsum_ps(y0);
}

template <TRN::CPU::Implementation Implementation>
static inline void sub(const float *x, const float *a, const std::size_t &cols, float *y)
{
	std::size_t col = 0;

	if (cols - col >= _8)
	{
		for (; col + _8 - 1 < cols; col += _8)
		{
			stream_ps(&y[col + _0], sub_ps(load_ps(&a[col + _0]), load_ps(&x[col + _0])));
			stream_ps(&y[col + _1], sub_ps(load_ps(&a[col + _1]), load_ps(&x[col + _1])));
			stream_ps(&y[col + _2], sub_ps(load_ps(&a[col + _2]), load_ps(&x[col + _2])));
			stream_ps(&y[col + _3], sub_ps(load_ps(&a[col + _3]), load_ps(&x[col + _3])));
			stream_ps(&y[col + _4], sub_ps(load_ps(&a[col + _4]), load_ps(&x[col + _4])));
			stream_ps(&y[col + _5], sub_ps(load_ps(&a[col + _5]), load_ps(&x[col + _5])));
			stream_ps(&y[col + _6], sub_ps(load_ps(&a[col + _6]), load_ps(&x[col + _6])));
			stream_ps(&y[col + _7], sub_ps(load_ps(&a[col + _7]), load_ps(&x[col + _7])));
		}

	}
	if (cols - col >= _4)
	{
		for (; col + _4 - 1 < cols; col += _4)
		{
			stream_ps(&y[col + _0], sub_ps(load_ps(&a[col + _0]), load_ps(&x[col + _0])));
			stream_ps(&y[col + _1], sub_ps(load_ps(&a[col + _1]), load_ps(&x[col + _1])));
			stream_ps(&y[col + _2], sub_ps(load_ps(&a[col + _2]), load_ps(&x[col + _2])));
			stream_ps(&y[col + _3], sub_ps(load_ps(&a[col + _3]), load_ps(&x[col + _3])));
		}
	}
	if (cols - col >= _2)
	{
		for (; col + _2 - 1 < cols; col += _2)
		{
			stream_ps(&y[col + _0], sub_ps(load_ps(&a[col + _0]), load_ps(&x[col + _0])));
			stream_ps(&y[col + _1], sub_ps(load_ps(&a[col + _1]), load_ps(&x[col + _1])));
		}
	}
	if (cols - col >= 0)
	{
		for (; col < cols; col += _1)
		{
			stream_ps(&y[col + _0], sub_ps(load_ps(&a[col + _0]), load_ps(&x[col + _0])));
		}
	}
}
template <TRN::CPU::Implementation Implementation>
static inline void matrix_vector_product(const std::size_t &batch_size, 
	float **batched_a, const std::size_t *batched_a_rows, const std::size_t *batched_a_cols, const std::size_t *batched_a_strides,
	float **batched_x, const std::size_t *batched_x_rows, const std::size_t *batched_x_cols, const std::size_t *batched_x_strides,
	float **batched_y, const std::size_t *batched_y_rows, const std::size_t *batched_y_cols, const std::size_t *batched_y_strides)
{

	//A = a
	//B = x
	//c = y


	/*static const float alpha = 1.0f;
	static const float beta = 0.0f;
	static const std::size_t incX = 1;
	static const std::size_t incY = 1;

#pragma omp parallel for 
	for (int batch = 0; batch < batch_size; batch++)
	{
		const std::size_t M = batched_a_rows[batch];
		const std::size_t N = batched_a_cols[batch];
		const float *A = batched_a[batch];
		const float *X = batched_x[batch];
		float *Y = batched_y[batch];
		const std::size_t lda = batched_a_strides[batch];
		cblas_sgemv(CBLAS_LAYOUT::CblasRowMajor, CBLAS_TRANSPOSE::CblasNoTrans, M, N, alpha, A, lda, X, incX, beta, Y, incY);
	}*/

	std::size_t R = 0;
	for (std::size_t batch = 0; batch < batch_size; batch++)
		R += batched_a_rows[batch];

	std::size_t K = (R + _1 - 1) / _1;
#pragma omp parallel for schedule(static, batch_size)
	for (int k = 0; k < K; k++)
	{
		std::size_t batch = k % batch_size;
		std::size_t row = (k / batch_size) * _1;
		float *x = batched_x[batch];
		float *y = batched_y[batch];
		float *a = batched_a[batch];
		auto a_stride = batched_a_strides[batch];
		auto a_rows = batched_a_rows[batch];
		auto a_cols = batched_a_cols[batch];

		auto __dp = setzero_ps();
		for (std::size_t s = 0; s < _1; s++)
		{
			set_element(dot_product<Implementation>(x, &a[a_stride * (row + s)], a_cols), s, __dp);
		}
		stream_ps(&y[row], __dp);
	}
}
static inline void sgemm(
	const CBLAS_TRANSPOSE &trans_a,
	const CBLAS_TRANSPOSE &trans_b,
	const std::size_t &batch_size,

	const float **a, const std::size_t *a_rows, const std::size_t *a_cols, const std::size_t *a_strides,
	const float **b, const std::size_t *b_rows, const std::size_t *b_cols, const std::size_t *b_strides,
	float **c, const std::size_t *c_rows, const std::size_t *c_cols, const std::size_t *c_strides,
	const float *alpha,
	const float *beta
)
{
	std::vector<CBLAS_TRANSPOSE> TransA_array(batch_size, trans_a);
	std::vector<CBLAS_TRANSPOSE> TransB_array(batch_size, trans_b);
	std::vector<std::size_t> group_size(batch_size, batch_size);
	std::vector<float>  alpha_array(batch_size, *alpha);
	std::vector<float> beta_array(batch_size, *beta);
	const std::size_t *M_array = NULL;
	const std::size_t *N_array = NULL;
	const std::size_t *K_array = NULL;
	if (trans_a == CBLAS_TRANSPOSE::CblasNoTrans)
	{
		M_array = a_rows;
		K_array = a_cols;
	}
	else
	{
		M_array = a_cols;
		K_array = a_rows;
	}
	if (trans_b == CBLAS_TRANSPOSE::CblasNoTrans)
	{
		assert(std::equal(K_array, K_array + batch_size, b_rows, b_rows + batch_size));
		K_array = b_rows;
		N_array = b_cols;
	}
	else
	{
		assert(std::equal(K_array, K_array + batch_size, b_cols, b_cols + batch_size));
		K_array = b_cols;
		N_array = b_rows;
	}
	assert(std::equal(M_array, M_array + batch_size, c_rows, c_rows + batch_size));
	assert(std::equal(N_array, N_array + batch_size, c_cols, c_cols + batch_size));

	const std::size_t *lda_array = a_strides;
	const std::size_t *ldb_array = b_strides;
	const std::size_t *ldc_array = c_strides;
	const float **A_array = a;
	const float **B_array = b;
	float **C_array = c;

	std::size_t group_count = 1;

	cblas_sgemm_batch
	(
		CBLAS_LAYOUT::CblasColMajor,
		TransA_array.data(), TransB_array.data(),
		M_array, N_array, K_array, alpha_array.data(), A_array, lda_array, B_array, ldb_array, beta_array.data(), C_array, ldc_array, group_count, group_size.data()
	);
}
template <typename Parameter>
static inline void copy_pre(const float *src, float *dst, const std::size_t &size)
{
	std::copy(src, src + size, dst);
}

template <>
static inline void copy_pre<Nothing>(const float *src, float *dst, const std::size_t &size)
{
	
}
template <TRN::CPU::Implementation Implementation, typename Parameter>
static inline void update_reservoir(
	const std::size_t &batch_size, const std::size_t &mini_batch_size, const std::size_t &mini_batch, std::size_t &bundle,
	const int *offsets, const std::size_t &ts, const std::size_t &total_duration,
	 const float **batched_w_rec, const std::size_t *batched_w_rec_rows, const std::size_t *batched_w_rec_cols, const  std::size_t *batched_w_rec_strides,
	float **batched_x_res, const std::size_t *batched_x_res_rows, const std::size_t *batched_x_res_cols, const  std::size_t *batched_x_res_strides,
	 float **batched_u, const std::size_t *batched_u_rows, const std::size_t *batched_u_cols, const  std::size_t *batched_u_strides,
	const float **batched_u_ffwd, const std::size_t *batched_u_ffwd_rows, const std::size_t *batched_u_ffwd_cols, const  std::size_t *batched_u_ffwd_strides,
	 float **batched_p, const std::size_t *batched_p_rows, const std::size_t *batched_p_cols, const  std::size_t *batched_p_strides,
	float ***bundled_pre, const std::size_t **bundled_pre_rows, const std::size_t **bundled_pre_cols, const  std::size_t **bundled_pre_strides,
	const float &leak_rate,
	const float *one, const float *zero
)
{
	const auto t = offsets[ts];
	const auto __leak_rate = set1_ps(leak_rate);


	sgemm(
		CBLAS_TRANSPOSE::CblasNoTrans,
		CBLAS_TRANSPOSE::CblasNoTrans,
		batch_size,

		(const float **)batched_w_rec, batched_w_rec_cols, batched_w_rec_rows, batched_w_rec_strides,
		(const float **)batched_x_res, batched_x_res_cols, batched_x_res_rows, batched_x_res_strides,
		batched_u, batched_u_cols, batched_u_rows, batched_u_strides, one, zero);
#pragma omp parallel for 
	for (int batch = 0; batch < batch_size; batch++)
	{
		auto rows = batched_w_rec_rows[batch];
		auto u_ffwd = &batched_u_ffwd[batch][t * batched_u_ffwd_strides[batch]];
		auto u = batched_u[batch];
		auto p = batched_p[batch];
		auto x_res = batched_x_res[batch];
		

		std::size_t row = 0;
		if (rows - row >= _8)
		{
			for (; row + _8 - 1 < rows; row += _8)
			{
				auto __p0 = load_ps(&p[row + _0]);
				__p0 = mul_add_ps(__leak_rate, sub_ps(add_ps(load_ps(&u[row + _0]), load_ps(&u_ffwd[row + _0])), __p0), __p0);
				//auto __x_res0 = tanh_ps<Implementation>(__p0);
				stream_ps(&p[row + _0], __p0);
				//stream_ps(&x_res[row + _0], __x_res0);
				

				auto __p1 = load_ps(&p[row + _1]);
				__p1 = mul_add_ps(__leak_rate, sub_ps(add_ps(load_ps(&u[row + _1]), load_ps(&u_ffwd[row + _1])), __p1), __p1);
				//auto __x_res1 = tanh_ps<Implementation>(__p1);
				stream_ps(&p[row + _1], __p1);
				//stream_ps(&x_res[row + _1], __x_res1);
		

				auto __p2 = load_ps(&p[row + _2]);
				__p2 = mul_add_ps(__leak_rate, sub_ps(add_ps(load_ps(&u[row + _2]), load_ps(&u_ffwd[row + _2])), __p2), __p2);
				//auto __x_res2 = tanh_ps<Implementation>(__p2);
				stream_ps(&p[row + _2], __p2);
				//stream_ps(&x_res[row + _2], __x_res2);
			

				auto __p3 = load_ps(&p[row + _3]);
				__p3 = mul_add_ps(__leak_rate, sub_ps(add_ps(load_ps(&u[row + _3]), load_ps(&u_ffwd[row + _3])), __p3), __p3);
				//auto __x_res3 = tanh_ps<Implementation>(__p3);
				stream_ps(&p[row + _3], __p3);
				//stream_ps(&x_res[row + _3], __x_res3);
			

				auto __p4 = load_ps(&p[row + _4]);
				__p4 = mul_add_ps(__leak_rate, sub_ps(add_ps(load_ps(&u[row + _4]), load_ps(&u_ffwd[row + _4])), __p4), __p4);
				//auto __x_res4 = tanh_ps<Implementation>(__p4);
				stream_ps(&p[row + _4], __p4);
				//stream_ps(&x_res[row + _4], __x_res4);
				

				auto __p5 = load_ps(&p[row + _5]);
				__p5 = mul_add_ps(__leak_rate, sub_ps(add_ps(load_ps(&u[row + _5]), load_ps(&u_ffwd[row + _5])), __p5), __p5);
				//auto __x_res5 = tanh_ps<Implementation>(__p5);
				stream_ps(&p[row + _5], __p5);
				//stream_ps(&x_res[row + _5], __x_res5);
				

				auto __p6 = load_ps(&p[row + _6]);
				__p6 = mul_add_ps(__leak_rate, sub_ps(add_ps(load_ps(&u[row + _6]), load_ps(&u_ffwd[row + _6])), __p6), __p6);
				//auto __x_res6 = tanh_ps<Implementation>(__p6);
				stream_ps(&p[row + _6], __p6);
				//stream_ps(&x_res[row + _6], __x_res6);
			

				auto __p7 = load_ps(&p[row + _7]);
				__p7 = mul_add_ps(__leak_rate, sub_ps(add_ps(load_ps(&u[row + _7]), load_ps(&u_ffwd[row + _7])), __p7), __p7);
				//auto __x_res7 = tanh_ps<Implementation>(__p7);
				stream_ps(&p[row + _7], __p7);
				//stream_ps(&x_res[row + _7], __x_res7);
			
			}
		}
		if (rows - row >= _4)
		{
			for (; row + _4 - 1 < rows; row += _4)
			{
				auto __p0 = load_ps(&p[row + _0]);
				__p0 = mul_add_ps(__leak_rate, sub_ps(add_ps(load_ps(&u[row + _0]), load_ps(&u_ffwd[row + _0])), __p0), __p0);
				//auto __x_res0 = tanh_ps<Implementation>(__p0);
				stream_ps(&p[row + _0], __p0);
				//stream_ps(&x_res[row + _0], __x_res0);
	

				auto __p1 = load_ps(&p[row + _1]);
				__p1 = mul_add_ps(__leak_rate, sub_ps(add_ps(load_ps(&u[row + _1]), load_ps(&u_ffwd[row + _1])), __p1), __p1);
			//	auto __x_res1 = tanh_ps<Implementation>(__p1);
				stream_ps(&p[row + _1], __p1);
				//stream_ps(&x_res[row + _1], __x_res1);
			

				auto __p2 = load_ps(&p[row + _2]);
				__p2 = mul_add_ps(__leak_rate, sub_ps(add_ps(load_ps(&u[row + _2]), load_ps(&u_ffwd[row + _2])), __p2), __p2);
				//auto __x_res2 = tanh_ps<Implementation>(__p2);
				stream_ps(&p[row + _2], __p2);
				//stream_ps(&x_res[row + _2], __x_res2);
			

				auto __p3 = load_ps(&p[row + _3]);
				__p3 = mul_add_ps(__leak_rate, sub_ps(add_ps(load_ps(&u[row + _3]), load_ps(&u_ffwd[row + _3])), __p3), __p3);
				//auto __x_res3 = tanh_ps<Implementation>(__p3);
				stream_ps(&p[row + _3], __p3);
				//stream_ps(&x_res[row + _3], __x_res3);
				
			}
		}
		if (rows - row >= _2)
		{
			for (; row + _2 - 1 < rows; row += _2)
			{
				auto __p0 = load_ps(&p[row + _0]);
				__p0 = mul_add_ps(__leak_rate, sub_ps(add_ps(load_ps(&u[row + _0]), load_ps(&u_ffwd[row + _0])), __p0), __p0);
			//	auto __x_res0 = tanh_ps<Implementation>(__p0);
				stream_ps(&p[row + _0], __p0);
				//stream_ps(&x_res[row + _0], __x_res0);


				auto __p1 = load_ps(&p[row + _1]);
				__p1 = mul_add_ps(__leak_rate, sub_ps(add_ps(load_ps(&u[row + _1]), load_ps(&u_ffwd[row + _1])), __p1), __p1);
				//auto __x_res1 = tanh_ps<Implementation>(__p1);
				stream_ps(&p[row + _1], __p1);
				//stream_ps(&x_res[row + _1], __x_res1);
			
			}
		}
		if (rows - row > 0)
		{
			for (; row + _1 - 1 < rows; row += _1)
			{
				auto __p0 = load_ps(&p[row + _0]);
				__p0 = mul_add_ps(__leak_rate, sub_ps(add_ps(load_ps(&u[row + _0]), load_ps(&u_ffwd[row + _0])), __p0), __p0);
				//auto __x_res0 = tanh_ps<Implementation>(__p0);
				stream_ps(&p[row + _0], __p0);
				//stream_ps(&x_res[row + _0], __x_res0);			
			}
		}

		tanh_v(rows, p, x_res);
		auto pre = &bundled_pre[bundle][batch][mini_batch * bundled_pre_strides[bundle][batch]];
		copy_pre<Parameter>(x_res, pre, rows);
	}
}




//	 error[row] =  learning_rate * (expected[row] - x_ro_row) * (1.0f - x_ro_row * x_ro_row);
template <TRN::CPU::Implementation Implementation>
static inline void update_readout_activations(const float *d, float *x, const std::size_t &cols)
{
	static const TRN::CPU::Traits<Implementation>::type minus_one = set1_ps(-1.0f);
	std::size_t col = 0;

	tanh_v(cols, x, x);
	if (cols - col >= _8)
	{
		for (; col + _8 - 1 < cols; col += _8)
		{
			auto x0 = load_ps(&x[col + _0]);
			stream_ps(&x[col + _0], mul_ps(sub_ps(x0, load_ps(&d[col + _0])), mul_add_ps(x0, x0, minus_one)));
			auto x1 = load_ps(&x[col + _1]);
			stream_ps(&x[col + _1], mul_ps(sub_ps(x1, load_ps(&d[col + _1])), mul_add_ps(x1, x1, minus_one)));
			auto x2 = load_ps(&x[col + _2]);
			stream_ps(&x[col + _2], mul_ps(sub_ps(x2, load_ps(&d[col + _2])), mul_add_ps(x2, x2, minus_one)));
			auto x3 = load_ps(&x[col + _3]);
			stream_ps(&x[col + _3], mul_ps(sub_ps(x3, load_ps(&d[col + _3])), mul_add_ps(x3, x3, minus_one)));
			auto x4 = load_ps(&x[col + _4]);
			stream_ps(&x[col + _4], mul_ps(sub_ps(x4, load_ps(&d[col + _4])), mul_add_ps(x4, x4, minus_one)));
			auto x5 = load_ps(&x[col + _5]);
			stream_ps(&x[col + _5], mul_ps(sub_ps(x5, load_ps(&d[col + _5])), mul_add_ps(x5, x5, minus_one)));
			auto x6 = load_ps(&x[col + _6]);
			stream_ps(&x[col + _6], mul_ps(sub_ps(x6, load_ps(&d[col + _6])), mul_add_ps(x6, x6, minus_one)));
			auto x7 = load_ps(&x[col + _7]);
			stream_ps(&x[col + _7], mul_ps(sub_ps(x7, load_ps(&d[col + _7])), mul_add_ps(x7, x7, minus_one)));
		}
	}
	if (cols - col >= _4)
	{
		for (; col + _4 - 1 < cols; col += _4)
		{
			auto x0 = load_ps(&x[col + _0]);
			stream_ps(&x[col + _0], mul_ps(sub_ps(x0, load_ps(&d[col + _0])), mul_add_ps(x0, x0, minus_one)));
			auto x1 = load_ps(&x[col + _1]);
			stream_ps(&x[col + _1], mul_ps(sub_ps(x1, load_ps(&d[col + _1])), mul_add_ps(x1, x1, minus_one)));
			auto x2 = load_ps(&x[col + _2]);
			stream_ps(&x[col + _2], mul_ps(sub_ps(x2, load_ps(&d[col + _2])), mul_add_ps(x2, x2, minus_one)));
			auto x3 = load_ps(&x[col + _3]);
			stream_ps(&x[col + _3], mul_ps(sub_ps(x3, load_ps(&d[col + _3])), mul_add_ps(x3, x3, minus_one)));
		}
	}
	if (cols - col >= _2)
	{
		for (; col + _2 - 1 < cols; col += _2)
		{
			auto x0 = load_ps(&x[col + _0]);
			stream_ps(&x[col + _0], mul_ps(sub_ps(x0, load_ps(&d[col + _0])), mul_add_ps(x0, x0, minus_one)));
			auto x1 = load_ps(&x[col + _1]);
			stream_ps(&x[col + _1], mul_ps(sub_ps(x1, load_ps(&d[col + _1])), mul_add_ps(x1, x1, minus_one)));
		}
	}
	if (cols - col > 0)
	{
		for (; col < cols; col += _1)
		{
			auto x0 = load_ps(&x[col + _0]);
			stream_ps(&x[col + _0], mul_ps(sub_ps(x0, load_ps(&d[col + _0])), mul_add_ps(x0, x0, minus_one)));
		}
	}
}

template <TRN::CPU::Implementation Implementation>
static inline void update_readout_activations(float *x, const std::size_t &cols)
{
	tanh_v(cols, x, x);
	/*std::size_t col = 0;

	if (cols - col >= _8)
	{
		for (; col + _8 - 1 < cols; col += _8)
		{
			stream_ps(&x[col + _0], tanh_ps<Implementation>(load_ps(&x[col + _0])));
			stream_ps(&x[col + _1], tanh_ps<Implementation>(load_ps(&x[col + _1])));
			stream_ps(&x[col + _2], tanh_ps<Implementation>(load_ps(&x[col + _2])));
			stream_ps(&x[col + _3], tanh_ps<Implementation>(load_ps(&x[col + _3])));
			stream_ps(&x[col + _4], tanh_ps<Implementation>(load_ps(&x[col + _4])));
			stream_ps(&x[col + _5], tanh_ps<Implementation>(load_ps(&x[col + _5])));
			stream_ps(&x[col + _6], tanh_ps<Implementation>(load_ps(&x[col + _6])));
			stream_ps(&x[col + _7], tanh_ps<Implementation>(load_ps(&x[col + _7])));
		}
	}
	if (cols - col >= _4)
	{
		for (; col + _4 - 1 < cols; col += _4)
		{
			stream_ps(&x[col + _0], tanh_ps<Implementation>(load_ps(&x[col + _0])));
			stream_ps(&x[col + _1], tanh_ps<Implementation>(load_ps(&x[col + _1])));
			stream_ps(&x[col + _2], tanh_ps<Implementation>(load_ps(&x[col + _2])));
			stream_ps(&x[col + _3], tanh_ps<Implementation>(load_ps(&x[col + _3])));
		}
	}
	if (cols - col >= _2)
	{
		for (; col + _2 - 1 < cols; col += _2)
		{
			stream_ps(&x[col + _0], tanh_ps<Implementation>(load_ps(&x[col + _0])));
			stream_ps(&x[col + _1], tanh_ps<Implementation>(load_ps(&x[col + _1])));
		}
	}
	if (cols - col > 0)
	{
		for (; col < cols; col += _1)
		{
			stream_ps(&x[col + _0], tanh_ps<Implementation>(load_ps(&x[col + _0])));
		}
	}*/
}

template <bool gather_states>
static void copy_states(
	const std::size_t &batch_size, const std::size_t &ts,
	const std::size_t &t, std::size_t &offset,
	const float **batched_x, const std::size_t &x_rows, const std::size_t &x_cols, const std::size_t &x_stride,
	float *states, const std::size_t &states_rows, const std::size_t &states_cols, const std::size_t &states_stride)
{}

template <>
static void copy_states<true>(
	const std::size_t &batch_size, const std::size_t &ts,
	const std::size_t &t, std::size_t &offset,
	const float **batched_x, const std::size_t &x_rows, const std::size_t &x_cols, const std::size_t &x_stride,
	float *states, const std::size_t &states_rows, const std::size_t &states_cols, const std::size_t &states_stride)
{
	float *states_ts = &states[ts * states_stride + offset];
	const int K = batch_size * x_rows;
#pragma omp parallel for
	for (int k = 0; k < K; k++)
	{
		auto row = k % x_rows;
		auto batch = k / x_rows;
		auto size = x_rows * x_stride;
		auto x = &batched_x[batch][(row + t) * x_stride];
		std::copy(x, x + x_cols, &states_ts[row *states_stride + batch * x_stride]);
	}

	offset += x_stride * batch_size;
}
//generation mode
template <TRN::CPU::Implementation Implementation, bool gather_states>
static inline void update_readout(
	std::future<void> &f,
	const std::size_t &batch_size, const std::size_t &mini_batch_size, std::size_t &mini_batch, std::size_t &bundle,
	const int *offsets, const std::size_t &ts, const std::size_t &total_duration, std::size_t &offset,
	const Nothing &parameter,
	float **batched_x_res, const std::size_t *batched_x_res_rows, const std::size_t *batched_x_res_cols, const std::size_t *batched_x_res_strides,
	float **batched_x_ro, const std::size_t *batched_x_ro_rows, const std::size_t *batched_x_ro_cols, const std::size_t *batched_x_ro_strides,
	float **batched_expected, const std::size_t *batched_expected_rows, const std::size_t *batched_expected_cols, const std::size_t *batched_expected_strides,
	float ***bundled_pre, const std::size_t **bundled_pre_rows, const std::size_t **bundled_pre_cols, const std::size_t **bundled_pre_strides,
	float **batched_post, const std::size_t *batched_post_rows, const std::size_t *batched_post_cols, const std::size_t *batched_post_strides,
	float ***bundled_desired, const std::size_t **bundled_desired_rows, const std::size_t **bundled_desired_cols, const std::size_t **bundled_desired_strides,
	float **batched_w_ro, const std::size_t *batched_w_ro_rows, const std::size_t *batched_w_ro_cols, const std::size_t *batched_w_ro_strides,
	const float *one, const float *zero,
	float *states, const std::size_t &states_rows, const std::size_t &states_cols, const std::size_t &states_stride)
{
	sgemm(
		CBLAS_TRANSPOSE::CblasNoTrans,
		CBLAS_TRANSPOSE::CblasNoTrans,
		batch_size,
		(const float **)batched_w_ro, batched_w_ro_cols, batched_w_ro_rows, batched_w_ro_strides,
		(const float **)batched_x_res, batched_x_res_cols, batched_x_res_rows, batched_x_res_strides,
		batched_x_ro, batched_x_ro_cols, batched_x_ro_rows, batched_x_ro_strides, one , zero);
#pragma omp parallel for
	for (int batch = 0; batch < batch_size; batch++)
	{
		auto x_ro = batched_x_ro[batch];
		auto cols = batched_x_ro_cols[batch];
		update_readout_activations<Implementation>(x_ro, cols);
	}
	copy_states<gather_states>(
		batch_size, ts, 0, offset,
		(const float **)batched_x_res, *batched_x_res_rows, *batched_x_res_cols, *batched_x_res_strides,
		states, states_rows, states_cols, states_stride);
	copy_states<gather_states>(
		batch_size, ts, 0, offset,
		(const float **)batched_x_ro, *batched_x_ro_rows, *batched_x_ro_cols, *batched_x_ro_strides,
		states, states_rows, states_cols, states_stride);
	if (ts < total_duration - 1)
	{
		const auto t = ts + 1;
		copy_states<gather_states>(
			batch_size, ts, ts + 1, offset,
			(const float **)batched_expected, *batched_expected_rows, *batched_expected_cols, *batched_expected_strides,
			states, states_rows, states_cols, states_stride);
	}
}

template <TRN::CPU::Implementation Implementation, bool gather_states>
static inline void update_readout(
	std::future<void> &f,
	const std::size_t &batch_size, const std::size_t &mini_batch_size, std::size_t &mini_batch, std::size_t &bundle,
	const int *offsets, const std::size_t &ts, const std::size_t &total_duration, std::size_t &offset,
	const Widrow_Hoff &parameter,
	float **batched_x_res, const std::size_t *batched_x_res_rows, const std::size_t *batched_x_res_cols, const std::size_t *batched_x_res_strides,
	float **batched_x_ro, const std::size_t *batched_x_ro_rows, const std::size_t *batched_x_ro_cols, const std::size_t *batched_x_ro_strides,
	float **batched_expected, const std::size_t *batched_expected_rows, const std::size_t *batched_expected_cols, const std::size_t *batched_expected_strides,
	float ***bundled_pre, const std::size_t **bundled_pre_rows, const std::size_t **bundled_pre_cols, const std::size_t **bundled_pre_strides,
	float **batched_post, const std::size_t *batched_post_rows, const std::size_t *batched_post_cols, const std::size_t *batched_post_strides,
	float ***bundled_desired, const std::size_t **bundled_desired_rows, const std::size_t **bundled_desired_cols, const std::size_t **bundled_desired_strides,
	float **batched_w_ro, const std::size_t *batched_w_ro_rows, const std::size_t *batched_w_ro_cols, const std::size_t *batched_w_ro_strides,
	const float *one, const float *zero,
	float *states, const std::size_t &states_rows, const std::size_t &states_cols, const std::size_t &states_stride)
{
	if (ts < total_duration - 1)
	{
		const auto t = offsets[ts + 1];
#pragma omp parallel for
		for (int batch = 0; batch < batch_size; batch++)
		{
			const auto cols = batched_expected_cols[batch];
			const auto expected = &batched_expected[batch][t * batched_expected_strides[batch]];
			auto desired = &bundled_desired[bundle][batch][mini_batch * bundled_desired_strides[bundle][batch]];
			std::copy(expected, expected + cols, desired);
		}
	}
	mini_batch++;
	auto remaining = total_duration - (ts + 1);
	if (mini_batch == mini_batch_size || remaining == 0)
	{
		if (f.valid())
			f.wait();
		f=std::async(std::launch::deferred, [=]()
		{
			std::size_t local_offset = offset;
			std::vector<std::size_t> effective_mini_batch(batch_size, mini_batch);

			sgemm(
				CBLAS_TRANSPOSE::CblasNoTrans,
				CBLAS_TRANSPOSE::CblasNoTrans,
				batch_size,
				(const float **)batched_w_ro, batched_w_ro_cols, batched_w_ro_rows, batched_w_ro_strides,
				(const float **)bundled_pre[bundle], bundled_pre_cols[bundle], effective_mini_batch.data(), bundled_pre_strides[bundle],
				batched_post, batched_post_cols, effective_mini_batch.data(), batched_post_strides, one, zero);

			const auto K = batch_size * mini_batch;
#pragma omp parallel for
			for (int k = 0; k < K; k++)
			{
				auto row = k % mini_batch;
				auto batch = k / mini_batch;
				auto desired = &bundled_desired[bundle][batch][row * bundled_desired_strides[bundle][batch]];
				auto post = &batched_post[batch][row * batched_post_strides[batch]];
				auto cols = batched_post_cols[batch];

				update_readout_activations<Implementation>(desired, post, cols);
			}
			auto copy_ts = ts - mini_batch + 1;
  			copy_states<gather_states>(
				batch_size, copy_ts, 0, local_offset,
				(const float **)bundled_pre[bundle], mini_batch, **bundled_pre_cols, **bundled_pre_strides,
				states, states_rows, states_cols, states_stride);

			copy_states<gather_states>(
				batch_size, copy_ts, 0, local_offset,
				(const float **)batched_post, mini_batch, *batched_post_cols, *batched_post_strides,
				states, states_rows, states_cols, states_stride);

			copy_states<gather_states>(
				batch_size, copy_ts, 0, local_offset,
				(const float **)bundled_desired[bundle], mini_batch, **bundled_desired_cols, **bundled_desired_strides,
				states, states_rows, states_cols, states_stride);
			sgemm(
				CBLAS_TRANSPOSE::CblasNoTrans,
				CBLAS_TRANSPOSE::CblasTrans,

				batch_size,
				(const float **)batched_post, batched_post_cols, effective_mini_batch.data(), batched_post_strides,
				(const float **)bundled_pre[bundle], bundled_pre_cols[bundle], effective_mini_batch.data(), bundled_pre_strides[bundle],

				batched_w_ro, batched_w_ro_cols, batched_w_ro_rows, batched_w_ro_strides,
				 parameter.get_learning_rate(), one);
		});

		mini_batch = 0;
		bundle = 1 - bundle;
	}
}


template <bool overwrite_states>
static inline void initialize_states(const std::size_t &batch_size, unsigned long &seed, float **batched_ptr, const std::size_t *batched_rows, const std::size_t *batched_cols, const std::size_t *batched_strides, const float &initial_state_scale)
{

}

template <>
static inline void initialize_states<true>(const std::size_t &batch_size, unsigned long &seed, float **batched_ptr, const std::size_t *batched_rows, const std::size_t *batched_cols, const std::size_t *batched_strides, const float &initial_state_scale)
{
	TRN::CPU::Random::uniform_implementation(seed, batched_ptr, batch_size, batched_rows, batched_cols, batched_strides, false, -initial_state_scale, initial_state_scale, 0.0f);

	seed += batch_size * batched_rows[0] * batched_cols[0];
}


template<TRN::CPU::Implementation Implementation, bool gather_states, bool overwrite_states, typename Parameter>
static inline void update_model(
	const std::size_t &batch_size,
	const std::size_t &mini_batch_size,
	unsigned long &seed,
	const Parameter &parameter,
	const std::size_t &stimulus_stride, const std::size_t &reservoir_stride, const std::size_t &prediction_stride,
	const std::size_t &stimulus_size, const std::size_t &reservoir_size, const std::size_t &prediction_size,
	const float &leak_rate, const float &initial_state_scale,
	float **batched_incoming, const std::size_t *batched_incoming_rows, const std::size_t *batched_incoming_cols, const std::size_t *batched_incoming_strides,
	float **batched_expected, const std::size_t *batched_expected_rows, const std::size_t *batched_expected_cols, const std::size_t *batched_expected_strides,
	float **batched_w_ffwd, const std::size_t *batched_w_ffwd_rows, const std::size_t *batched_w_ffwd_cols, const std::size_t *batched_w_ffwd_strides,
	float **batched_u_ffwd, const std::size_t *batched_u_ffwd_rows, const std::size_t *batched_u_ffwd_cols, const std::size_t *batched_u_ffwd_strides,
	float **batched_x_res, const std::size_t *batched_x_res_rows, const std::size_t *batched_x_res_cols, const std::size_t *batched_x_res_strides,
	float **batched_w_rec, const std::size_t *batched_w_rec_rows, const std::size_t *batched_w_rec_cols, const std::size_t *batched_w_rec_strides, 
	float **batched_u, const std::size_t *batched_u_rows, const std::size_t *batched_u_cols, const std::size_t *batched_u_strides,
	float **batched_p, const std::size_t *batched_p_rows, const std::size_t *batched_p_cols, const std::size_t *batched_p_strides,
	float **batched_x_ro, const std::size_t *batched_x_ro_rows, const std::size_t *batched_x_ro_cols, const std::size_t *batched_x_ro_strides,
	float **batched_w_ro, const std::size_t *batched_w_ro_rows, const std::size_t *batched_w_ro_cols, const std::size_t *batched_w_ro_strides,
	float ***bundled_pre, const std::size_t **bundled_pre_rows, const std::size_t **bundled_pre_cols, const std::size_t **bundled_pre_strides,
	float **batched_post, const std::size_t *batched_post_rows, const std::size_t *batched_post_cols, const std::size_t *batched_post_strides,
	float ***bundled_desired, const std::size_t **bundled_desired_rows, const std::size_t **bundled_desired_cols, const std::size_t **bundled_desired_strides,
	const int *offsets, const int *durations, const std::size_t &repetitions, const std::size_t &total_duration,
	float *states_samples, const std::size_t &states_rows, const std::size_t &states_cols, const std::size_t &states_stride,
	const float *one, const float *zero)
{
	sgemm(
		CBLAS_TRANSPOSE::CblasNoTrans,
		CBLAS_TRANSPOSE::CblasNoTrans,
		batch_size,
		(const float **)batched_w_ffwd, batched_w_ffwd_cols, batched_w_ffwd_rows, batched_w_ffwd_strides,
		(const float **)batched_incoming, batched_incoming_cols, batched_incoming_rows, batched_incoming_strides,
		batched_u_ffwd, batched_u_ffwd_cols, batched_u_ffwd_rows, batched_u_ffwd_strides, one, zero
	);

	std::size_t ts = 0;
	std::size_t mini_batch = 0;
	std::size_t bundle = 0;

	std::future<void> f;
	for (std::size_t repetition = 0; repetition < repetitions; repetition++)
	{
		initialize_states<overwrite_states>(batch_size, seed, batched_p, batched_p_rows, batched_p_cols, batched_p_strides, initial_state_scale);
		initialize_states<overwrite_states>(batch_size, seed, batched_x_res, batched_x_res_rows, batched_x_res_cols, batched_x_res_strides, initial_state_scale);

		for (std::size_t k = 0; k < durations[repetition]; k++, ts++)
		{
			update_reservoir<Implementation, Parameter>(
				batch_size, mini_batch_size, mini_batch, bundle,
				offsets, ts, total_duration,
				(const float **)batched_w_rec, batched_w_rec_rows, batched_w_rec_cols, batched_w_rec_strides,
				batched_x_res, batched_x_res_rows, batched_x_res_cols, batched_x_res_strides,
				batched_u, batched_u_rows, batched_u_cols, batched_u_strides,
				(const float **)batched_u_ffwd, batched_u_ffwd_rows, batched_u_ffwd_cols, batched_u_ffwd_strides,
				batched_p, batched_p_rows, batched_p_cols, batched_p_strides, 
				bundled_pre, bundled_pre_rows, bundled_pre_cols, bundled_pre_strides,
				leak_rate, one, zero);
			std::size_t offset = 0;
			copy_states<gather_states>(
				batch_size, ts, (std::size_t) offsets[ts], offset,
				(const float **)batched_incoming, 1, *batched_incoming_cols, *batched_incoming_strides,
				states_samples, states_rows, states_cols, states_stride);
			update_readout<Implementation, gather_states>
				(
				f,
					batch_size, mini_batch_size, mini_batch, bundle,
					offsets, ts, total_duration, offset,
					parameter,
					batched_x_res, batched_x_res_rows, batched_x_res_cols, batched_x_res_strides,
					batched_x_ro, batched_x_ro_rows, batched_x_ro_cols, batched_x_ro_strides,
					batched_expected, batched_expected_rows, batched_expected_cols, batched_expected_strides,
					bundled_pre, bundled_pre_rows, bundled_pre_cols, bundled_pre_strides,
					batched_post, batched_post_rows, batched_post_cols, batched_post_strides,
					bundled_desired, bundled_desired_rows, bundled_desired_cols, bundled_desired_strides,
					batched_w_ro, batched_w_ro_rows, batched_w_ro_cols, batched_w_ro_strides,one, zero, 
					states_samples, states_rows, states_cols, states_stride
					); 
		}
	}
	if (f.valid())
		f.wait();
}
template <TRN::CPU::Implementation Implementation>
void TRN::CPU::Algorithm<Implementation>::learn_widrow_hoff(
	const std::size_t &batch_size,
	const std::size_t &mini_batch_size,
	unsigned long &seed,
	const std::size_t &stimulus_stride, const std::size_t &reservoir_stride, const std::size_t &prediction_stride,
	const std::size_t &stimulus_size, const std::size_t &reservoir_size, const std::size_t &prediction_size,
	const float &leak_rate, const float &initial_state_scale,
	float *expected, const std::size_t &expected_rows, const std::size_t &expected_cols, const std::size_t &expected_stride,
	float **batched_incoming, const std::size_t *batched_incoming_rows, const std::size_t *batched_incoming_cols, const std::size_t *batched_incoming_strides,
	float **batched_expected, const std::size_t *batched_expected_rows, const std::size_t *batched_expected_cols, const std::size_t *batched_expected_strides,
	float **batched_w_ffwd, const std::size_t *batched_w_ffwd_rows, const std::size_t *batched_w_ffwd_cols, const std::size_t *batched_w_ffwd_strides,
	float **batched_u_ffwd, const std::size_t *batched_u_ffwd_rows, const std::size_t *batched_u_ffwd_cols, const std::size_t *batched_u_ffwd_strides,
	float **batched_x_res, const std::size_t *batched_x_res_rows, const std::size_t *batched_x_res_cols, const std::size_t *batched_x_res_strides,
	float **batched_w_rec, const std::size_t *batched_w_rec_rows, const std::size_t *batched_w_rec_cols, const std::size_t *batched_w_rec_strides, 
	float **batched_u, const std::size_t *batched_u_rows, const std::size_t *batched_u_cols, const std::size_t *batched_u_strides,
	float **batched_p, const std::size_t *batched_p_rows, const std::size_t *batched_p_cols, const std::size_t *batched_p_strides,
	float **batched_x_ro, const std::size_t *batched_x_ro_rows, const std::size_t *batched_x_ro_cols, const std::size_t *batched_x_ro_strides,
	float **batched_w_ro, const std::size_t *batched_w_ro_rows, const std::size_t *batched_w_ro_cols, const std::size_t *batched_w_ro_strides,
	float ***bundled_pre, const std::size_t **bundled_pre_rows, const std::size_t **bundled_pre_cols, const std::size_t **bundled_pre_strides,
	float **batched_post, const std::size_t *batched_post_rows, const std::size_t *batched_post_cols, const std::size_t *batched_post_strides,
	float ***bundled_desired, const std::size_t **bundled_desired_rows, const std::size_t **bundled_desired_cols, const std::size_t **bundled_desired_strides,
	const int *offsets, const int *durations, const std::size_t &repetitions, const std::size_t &total_duration,
	float *states_samples, const std::size_t &states_rows, const std::size_t &states_cols, const std::size_t &states_stride,

	
	const float *one, const float *zero, const float *learning_rate)
{
	if (states_samples == NULL)
	{
		update_model<Implementation, false, true>(
			batch_size, mini_batch_size, seed, Widrow_Hoff(learning_rate),
			stimulus_stride, reservoir_stride, prediction_stride,
			stimulus_size, reservoir_size, prediction_size,
			leak_rate, initial_state_scale,
			batched_incoming, batched_incoming_rows, batched_incoming_cols, batched_incoming_strides,
			batched_expected, batched_expected_rows, batched_expected_cols, batched_expected_strides,
			batched_w_ffwd, batched_w_ffwd_rows, batched_w_ffwd_cols, batched_w_ffwd_strides,
			batched_u_ffwd, batched_u_ffwd_rows, batched_u_ffwd_cols, batched_u_ffwd_strides,
			batched_x_res, batched_x_res_rows, batched_x_res_cols, batched_x_res_strides,
			batched_w_rec, batched_w_rec_rows, batched_w_rec_cols, batched_w_rec_strides,
			batched_u, batched_u_rows, batched_u_cols, batched_u_strides,
			batched_p, batched_p_rows, batched_p_cols, batched_p_strides,
			batched_x_ro, batched_x_ro_rows, batched_x_ro_cols, batched_x_ro_strides,
			batched_w_ro, batched_w_ro_rows, batched_w_ro_cols, batched_w_ro_strides,
			bundled_pre, bundled_pre_rows, bundled_pre_cols, bundled_pre_strides,
			batched_post, batched_post_rows, batched_post_cols, batched_post_strides,
			bundled_desired, bundled_desired_rows, bundled_desired_cols, bundled_desired_strides,
			offsets, durations, repetitions,total_duration,
			states_samples, states_rows, states_cols, states_stride,
			one, zero
			);
	}
	else
	{
		update_model<Implementation, true, true>(
			batch_size, mini_batch_size, seed, Widrow_Hoff(learning_rate),
			stimulus_stride, reservoir_stride, prediction_stride,
			stimulus_size, reservoir_size, prediction_size,
			leak_rate, initial_state_scale,
			batched_incoming, batched_incoming_rows, batched_incoming_cols, batched_incoming_strides,
			batched_expected, batched_expected_rows, batched_expected_cols, batched_expected_strides,
			batched_w_ffwd, batched_w_ffwd_rows, batched_w_ffwd_cols, batched_w_ffwd_strides,
			batched_u_ffwd, batched_u_ffwd_rows, batched_u_ffwd_cols, batched_u_ffwd_strides,
			batched_x_res, batched_x_res_rows, batched_x_res_cols, batched_x_res_strides,
			batched_w_rec, batched_w_rec_rows, batched_w_rec_cols, batched_w_rec_strides,
			batched_u, batched_u_rows, batched_u_cols, batched_u_strides,
			batched_p, batched_p_rows, batched_p_cols, batched_p_strides,
			batched_x_ro, batched_x_ro_rows, batched_x_ro_cols, batched_x_ro_strides,
			batched_w_ro, batched_w_ro_rows, batched_w_ro_cols, batched_w_ro_strides,
			bundled_pre, bundled_pre_rows, bundled_pre_cols, bundled_pre_strides,
			batched_post, batched_post_rows, batched_post_cols, batched_post_strides,
			bundled_desired, bundled_desired_rows, bundled_desired_cols, bundled_desired_strides,
			offsets, durations, repetitions,total_duration,
			states_samples, states_rows, states_cols, states_stride,
			one, zero
			);
	}
}
template <TRN::CPU::Implementation Implementation>
void TRN::CPU::Algorithm<Implementation>::prime(
	const std::size_t &batch_size,
	const std::size_t &mini_batch_size,
	unsigned long &seed,
	const std::size_t &stimulus_stride, const std::size_t &reservoir_stride, const std::size_t &prediction_stride,
	const std::size_t &stimulus_size, const std::size_t &reservoir_size, const std::size_t &prediction_size,
	const float &leak_rate, const float &initial_state_scale,
	float *expected, const std::size_t &expected_rows, const std::size_t &expected_cols, const std::size_t &expected_stride,
	
	float **batched_incoming, const std::size_t *batched_incoming_rows, const std::size_t *batched_incoming_cols, const std::size_t *batched_incoming_strides,
	float **batched_expected, const std::size_t *batched_expected_rows, const std::size_t *batched_expected_cols, const std::size_t *batched_expected_strides,
	float **batched_w_ffwd, const std::size_t *batched_w_ffwd_rows, const std::size_t *batched_w_ffwd_cols, const std::size_t *batched_w_ffwd_strides,
	float **batched_u_ffwd, const std::size_t *batched_u_ffwd_rows, const std::size_t *batched_u_ffwd_cols, const std::size_t *batched_u_ffwd_strides,
	float **batched_x_res, const std::size_t *batched_x_res_rows, const std::size_t *batched_x_res_cols, const std::size_t *batched_x_res_strides,
	float **batched_w_rec, const std::size_t *batched_w_rec_rows, const std::size_t *batched_w_rec_cols, const std::size_t *batched_w_rec_strides, 
	float **batched_u, const std::size_t *batched_u_rows, const std::size_t *batched_u_cols, const std::size_t *batched_u_strides,
	float **batched_p, const std::size_t *batched_p_rows, const std::size_t *batched_p_cols, const std::size_t *batched_p_strides,
	float **batched_x_ro, const std::size_t *batched_x_ro_rows, const std::size_t *batched_x_ro_cols, const std::size_t *batched_x_ro_strides,
	float **batched_w_ro, const std::size_t *batched_w_ro_rows, const std::size_t *batched_w_ro_cols, const std::size_t *batched_w_ro_strides,
	float ***bundled_pre, const std::size_t **bundled_pre_rows, const std::size_t **bundled_pre_cols, const std::size_t **bundled_pre_strides,
	float **batched_post, const std::size_t *batched_post_rows, const std::size_t *batched_post_cols, const std::size_t *batched_post_strides,
	float ***bundled_desired, const std::size_t **bundled_desired_rows, const std::size_t **bundled_desired_cols, const std::size_t **bundled_desired_strides,
	const int *offsets, const int *durations, const std::size_t &repetitions, const std::size_t &total_duration,
	float *states_samples, const std::size_t &states_rows, const std::size_t &states_cols, const std::size_t &states_stride,
	const float *one, const float *zero)
{


	if (states_samples == NULL)
	{
		update_model<Implementation, false, true>(
			batch_size, mini_batch_size, seed, Nothing(),
			stimulus_stride, reservoir_stride, prediction_stride,
			stimulus_size, reservoir_size, prediction_size,
			leak_rate, initial_state_scale,
			batched_incoming, batched_incoming_rows, batched_incoming_cols, batched_incoming_strides,
			batched_expected, batched_expected_rows, batched_expected_cols, batched_expected_strides,
			batched_w_ffwd, batched_w_ffwd_rows, batched_w_ffwd_cols, batched_w_ffwd_strides,
			batched_u_ffwd, batched_u_ffwd_rows, batched_u_ffwd_cols, batched_u_ffwd_strides,
			batched_x_res, batched_x_res_rows, batched_x_res_cols, batched_x_res_strides,
			batched_w_rec, batched_w_rec_rows, batched_w_rec_cols, batched_w_rec_strides,
			batched_u, batched_u_rows, batched_u_cols, batched_u_strides,
			batched_p, batched_p_rows, batched_p_cols, batched_p_strides,
			batched_x_ro, batched_x_ro_rows, batched_x_ro_cols, batched_x_ro_strides,
			batched_w_ro, batched_w_ro_rows, batched_w_ro_cols, batched_w_ro_strides,
			bundled_pre, bundled_pre_rows, bundled_pre_cols, bundled_pre_strides,
			batched_post, batched_post_rows, batched_post_cols, batched_post_strides,
			bundled_desired, bundled_desired_rows, bundled_desired_cols, bundled_desired_strides,
			offsets, durations, repetitions, total_duration,
			states_samples, states_rows, states_cols, states_stride, one, zero);
	}
	else
	{
		update_model<Implementation, true, true>(
			batch_size, mini_batch_size, seed, Nothing(),
			stimulus_stride, reservoir_stride, prediction_stride,
			stimulus_size, reservoir_size, prediction_size,
			leak_rate, initial_state_scale,
			batched_incoming, batched_incoming_rows, batched_incoming_cols, batched_incoming_strides,
			batched_expected, batched_expected_rows, batched_expected_cols, batched_expected_strides,
			batched_w_ffwd, batched_w_ffwd_rows, batched_w_ffwd_cols, batched_w_ffwd_strides,
			batched_u_ffwd, batched_u_ffwd_rows, batched_u_ffwd_cols, batched_u_ffwd_strides,
			batched_x_res, batched_x_res_rows, batched_x_res_cols, batched_x_res_strides,
			batched_w_rec, batched_w_rec_rows, batched_w_rec_cols, batched_w_rec_strides,
			batched_u, batched_u_rows, batched_u_cols, batched_u_strides,
			batched_p, batched_p_rows, batched_p_cols, batched_p_strides,
			batched_x_ro, batched_x_ro_rows, batched_x_ro_cols, batched_x_ro_strides,
			batched_w_ro, batched_w_ro_rows, batched_w_ro_cols, batched_w_ro_strides,
			bundled_pre, bundled_pre_rows, bundled_pre_cols, bundled_pre_strides,
			batched_post, batched_post_rows, batched_post_cols, batched_post_strides,
			bundled_desired, bundled_desired_rows, bundled_desired_cols, bundled_desired_strides,
			offsets, durations, repetitions,total_duration,
			states_samples, states_rows, states_cols, states_stride, one, zero);
	}

}

static std::mutex mutex;

template <TRN::CPU::Implementation Implementation>
void TRN::CPU::Algorithm<Implementation>::generate(
	const std::size_t &batch_size,
	const std::size_t &mini_batch_size,
	unsigned long &seed,
	const std::size_t &stimulus_stride, const std::size_t &reservoir_stride, const std::size_t &prediction_stride,
	const std::size_t &stimulus_size, const std::size_t &reservoir_size, const std::size_t &prediction_size,
	const float &leak_rate, const float &initial_state_scale,
	float *expected, const std::size_t &expected_rows, const std::size_t &expected_cols, const std::size_t &expected_stride,
	
	float **batched_incoming, const std::size_t *batched_incoming_rows, const std::size_t *batched_incoming_cols, const std::size_t *batched_incoming_strides,
	float **batched_expected, const std::size_t *batched_expected_rows, const std::size_t *batched_expected_cols, const std::size_t *batched_expected_strides,
	float **batched_w_ffwd, const std::size_t *batched_w_ffwd_rows, const std::size_t *batched_w_ffwd_cols, const std::size_t *batched_w_ffwd_strides,
	float **batched_u_ffwd, const std::size_t *batched_u_ffwd_rows, const std::size_t *batched_u_ffwd_cols, const std::size_t *batched_u_ffwd_strides,
	float **batched_x_res, const std::size_t *batched_x_res_rows, const std::size_t *batched_x_res_cols, const std::size_t *batched_x_res_strides,
	float **batched_w_rec, const std::size_t *batched_w_rec_rows, const std::size_t *batched_w_rec_cols, const std::size_t *batched_w_rec_strides,
	float **batched_u, const std::size_t *batched_u_rows, const std::size_t *batched_u_cols, const std::size_t *batched_u_strides,
	float **batched_p, const std::size_t *batched_p_rows, const std::size_t *batched_p_cols, const std::size_t *batched_p_strides,
	float **batched_x_ro, const std::size_t *batched_x_ro_rows, const std::size_t *batched_x_ro_cols, const std::size_t *batched_x_ro_strides,
	float **batched_w_ro, const std::size_t *batched_w_ro_rows, const std::size_t *batched_w_ro_cols, const std::size_t *batched_w_ro_strides,
	float ***bundled_pre, const std::size_t **bundled_pre_rows, const std::size_t **bundled_pre_cols, const std::size_t **bundled_pre_strides,
	float **batched_post, const std::size_t *batched_post_rows, const std::size_t *batched_post_cols, const std::size_t *batched_post_strides,
	float ***bundled_desired, const std::size_t **bundled_desired_rows, const std::size_t **bundled_desired_cols, const std::size_t **bundled_desired_strides,
	const int *offsets, const int *durations, const std::size_t &repetitions, const std::size_t &total_duration,
	float *states_samples, const std::size_t &states_rows, const std::size_t &states_cols, const std::size_t &states_stride,
	const float *one, const float *zero)
{
	if (states_samples == NULL)
	{
		update_model<Implementation, false, false>(
			batch_size, mini_batch_size, seed, Nothing(),
			stimulus_stride, reservoir_stride, prediction_stride,
			stimulus_size, reservoir_size, prediction_size,
			leak_rate, initial_state_scale,
			batched_incoming, batched_incoming_rows, batched_incoming_cols, batched_incoming_strides,
			batched_expected, batched_expected_rows, batched_expected_cols, batched_expected_strides,
			batched_w_ffwd, batched_w_ffwd_rows, batched_w_ffwd_cols, batched_w_ffwd_strides,
			batched_u_ffwd, batched_u_ffwd_rows, batched_u_ffwd_cols, batched_u_ffwd_strides,
			batched_x_res, batched_x_res_rows, batched_x_res_cols, batched_x_res_strides,
			batched_w_rec, batched_w_rec_rows, batched_w_rec_cols, batched_w_rec_strides,
			batched_u, batched_u_rows, batched_u_cols, batched_u_strides,
			batched_p, batched_p_rows, batched_p_cols, batched_p_strides,
			batched_x_ro, batched_x_ro_rows, batched_x_ro_cols, batched_x_ro_strides,
			batched_w_ro, batched_w_ro_rows, batched_w_ro_cols, batched_w_ro_strides,
			bundled_pre, bundled_pre_rows, bundled_pre_cols, bundled_pre_strides,
			batched_post, batched_post_rows, batched_post_cols, batched_post_strides,
			bundled_desired, bundled_desired_rows, bundled_desired_cols, bundled_desired_strides,
			offsets, durations, repetitions,total_duration,
			states_samples, states_rows, states_cols, states_stride, one, zero);
	}
	else
	{
		update_model<Implementation, true, false>(
			batch_size, mini_batch_size, seed, Nothing(),
			stimulus_stride, reservoir_stride, prediction_stride,
			stimulus_size, reservoir_size, prediction_size,
			leak_rate, initial_state_scale,
			batched_incoming, batched_incoming_rows, batched_incoming_cols, batched_incoming_strides,
			batched_expected, batched_expected_rows, batched_expected_cols, batched_expected_strides,
			batched_w_ffwd, batched_w_ffwd_rows, batched_w_ffwd_cols, batched_w_ffwd_strides,
			batched_u_ffwd, batched_u_ffwd_rows, batched_u_ffwd_cols, batched_u_ffwd_strides,
			batched_x_res, batched_x_res_rows, batched_x_res_cols, batched_x_res_strides,
			batched_w_rec, batched_w_rec_rows, batched_w_rec_cols, batched_w_rec_strides,
			batched_u, batched_u_rows, batched_u_cols, batched_u_strides,
			batched_p, batched_p_rows, batched_p_cols, batched_p_strides,
			batched_x_ro, batched_x_ro_rows, batched_x_ro_cols, batched_x_ro_strides,
			batched_w_ro, batched_w_ro_rows, batched_w_ro_cols, batched_w_ro_strides,
			bundled_pre, bundled_pre_rows, bundled_pre_cols, bundled_pre_strides,
			batched_post, batched_post_rows, batched_post_cols, batched_post_strides,
			bundled_desired, bundled_desired_rows, bundled_desired_cols, bundled_desired_strides,
			offsets, durations, repetitions,total_duration,
			states_samples, states_rows, states_cols, states_stride, one, zero);
	}

}


template <TRN::CPU::Implementation Implementation>
std::shared_ptr<TRN::CPU::Algorithm<Implementation>> TRN::CPU::Algorithm<Implementation>::create()
{
	return std::make_shared<TRN::CPU::Algorithm<Implementation>>();
}





