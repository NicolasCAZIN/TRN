#include "stdafx.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <helper_math.h>
#include <cub/cub.cuh>

#include "Context.h"
#include "Algorithm.cuh"
#include "Random.cuh"
#include "Driver.cuh"

#define BLOCK_X 32
#define BLOCK_Y 32
#define warpSize 32

static inline void sgemm(
	const cublasHandle_t handle,
	const cublasOperation_t &trans_a,
	const cublasOperation_t &trans_b,
	const int batch_size,

	const float **a, const int a_rows, const int a_cols, const int a_stride,
	const float **b, const int b_rows, const int b_cols, const int b_stride,
	float **c, const int c_rows, const int c_cols, const int c_stride,
	const float *alpha, 
	const float *beta
)
{
	int m, n, k;
	if (trans_a == cublasOperation_t::CUBLAS_OP_N)
	{
		m = a_rows;
		k = a_cols;
	}
	else
	{
		m = a_cols;
		k = a_rows;
	}
	if (trans_b == cublasOperation_t::CUBLAS_OP_N)
	{
		assert(k == b_rows);
		k = b_rows;
		n = b_cols;
	}
	else
	{
		assert(k == b_cols);
		k = b_cols;
		n = b_rows;
	}
	assert(m == c_rows);
	assert(n = c_cols);


	checkCudaErrors(cublasSgemmBatched(handle,
		trans_a, trans_b,
		m, n, k,
		alpha,
		a, a_stride,
		b, b_stride,
		beta,
		c, c_stride,
		batch_size
	));
}

__device__
static inline std::size_t round_down(const std::size_t &offset)
{
	return (offset / warpSize) *  warpSize;
}
__device__
static inline std::size_t round_up(const std::size_t &offset, const std::size_t &max_offset)
{
	return min(max_offset, (round_down(offset + warpSize - 1)));
}

__global__
static void compute_roi_kernel(const std::size_t batch_size,
	const std::size_t rows, const std::size_t cols,
	const float x_min, const float x_max, const float y_min, const float y_max,
	const float x_range, const float y_range,
	const float radius,
	const float **current_position, const std::size_t &current_position_strides,
	std::size_t *roi_row_begin, std::size_t *roi_row_end, std::size_t *roi_col_begin, std::size_t *roi_col_end)
{
	for (int batch = blockIdx.x * blockDim.x + threadIdx.x; batch < batch_size; batch += gridDim.x * blockDim.x)
	{
		auto p = *reinterpret_cast<float2 *>(const_cast<float *>(current_position[batch]));
	

		auto roi_x_min = clamp(p.x - radius, x_min, x_max);
		auto roi_x_max = clamp(p.x + radius, x_min, x_max);
		auto roi_y_min = clamp(p.y - radius, y_min, y_max);
		auto roi_y_max = clamp(p.y + radius, y_min, y_max);

		roi_row_begin[batch] = (rows - 1) * ((roi_y_min - y_min) / y_range);
		roi_row_end[batch] = (rows - 1) * ((roi_y_max - y_min) / y_range);
		roi_col_begin[batch] = round_down((cols - 1) * ((roi_x_min - x_min) / x_range));
		roi_col_end[batch] = round_up((cols - 1) * ((roi_x_max - x_min) / x_range), cols);

		assert(roi_row_begin[batch] <= roi_row_end[batch]);
		assert(roi_col_begin[batch] <= roi_col_end[batch]);
	}
}
__global__
static void compute_roi_kernel(const std::size_t batch_size,
	const std::size_t rows, const std::size_t cols,
	std::size_t *roi_row_begin, std::size_t *roi_row_end, std::size_t *roi_col_begin, std::size_t *roi_col_end)
{
	for (int batch = blockIdx.x * blockDim.x + threadIdx.x; batch < batch_size; batch += gridDim.x * blockDim.x)
	{
		roi_row_begin[batch] = 0;
		roi_row_end[batch] = rows;
		roi_col_begin[batch] = 0;
		roi_col_end[batch] = cols;
	}
}

void compute_roi(
	const cudaStream_t *streams,
	const cublasHandle_t *handles,
	const cudaEvent_t *events,
	const std::size_t &batch_size,
	const std::size_t &rows, const std::size_t &cols,
	const float &x_min, const float &x_max, const float &y_min, const float &y_max,
	const float &radius,
	const float **current_position, const std::size_t &current_position_strides,
	std::size_t *__restrict__ roi_row_begin, std::size_t *__restrict__ roi_row_end, std::size_t *__restrict__ roi_col_begin, std::size_t *__restrict__ roi_col_end)
{
	dim3 grid, block;

	block.x = BLOCK_X * BLOCK_Y;
	grid.x = (batch_size + block.x - 1) / block.x;
	if (radius > 0.0f)
	{
		compute_roi_kernel << <grid, block, 0, streams[0] >> > (
			batch_size, rows, cols, x_min, x_max, y_min, y_max,
			x_max - x_min, y_max - y_min, radius, current_position, current_position_strides,
			roi_row_begin, roi_row_end, roi_col_begin, roi_col_end
			);
	}
	else
	{
		compute_roi_kernel << <grid, block, 0, streams[0] >> > (
			batch_size, rows, cols,
			roi_row_begin, roi_row_end, roi_col_begin, roi_col_end
			);
	}
	checkCudaErrors(cudaGetLastError());
}

template <typename Parameter >
static inline void prepare(
	const cudaStream_t *streams,
	const cublasHandle_t *handles,
	const cudaEvent_t *events,
	const Parameter &parameter, 
	const std::size_t &batch_size, const std::size_t &place_cells_number, const std::size_t &rows, const std::size_t &cols,
	const float *x_grid, const float *y_grid,
	const std::size_t &roi_rows, const std::size_t &roi_cols,
	const std::size_t *roi_row_begin, const std::size_t *roi_row_end, const std::size_t *roi_col_begin, const std::size_t *roi_col_end)
{
}

__global__
static  void diff2_kernel(
	const int batch_size, 
	const int place_cells_number,
	const std::size_t *roi_begin, const std::size_t *roi_end,
	const float * __restrict__ grid,
	const float * __restrict__ center, const float * __restrict__ width,
	float ** __restrict__ batched_grid_centered2, const int grid_centered2_stride)
{
	for (int batch = blockIdx.z * blockDim.z + threadIdx.z; batch < batch_size; batch += gridDim.z * blockDim.z)
	{
		auto grid_centered2 = batched_grid_centered2[batch];
		for (int k = roi_begin[batch] + blockIdx.y * blockDim.y + threadIdx.y; k < roi_end[batch]; k += gridDim.y * blockDim.y)
		{
			for (int place_cell = blockIdx.x * blockDim.x + threadIdx.x; place_cell < place_cells_number; place_cell += gridDim.x * blockDim.x)
			{
				auto d = grid[k] - center[place_cell];
				grid_centered2[k * grid_centered2_stride + place_cell] = d * d * width[place_cell];
			}
		}
	}
}

template <>
static inline void prepare(
	const cudaStream_t *streams,
	const cublasHandle_t *handles,
	const cudaEvent_t *events,
	const Model &parameter,
	const std::size_t &batch_size, const std::size_t &place_cells_number, const std::size_t &rows, const std::size_t &cols,
	const float *x_grid, const float *y_grid,
	const std::size_t &roi_rows, const std::size_t &roi_cols,
	const std::size_t *roi_row_begin, const std::size_t *roi_row_end, const std::size_t *roi_col_begin, const std::size_t *roi_col_end)
{
	{
		dim3 grid, block;

		block.x = BLOCK_X;
		block.y = BLOCK_Y;
		block.z = 1;

		grid.x = (place_cells_number + block.x - 1) / block.x;
		grid.y = (roi_cols + block.y - 1) / block.y;
		grid.z = (batch_size + block.z - 1) / block.z;

		diff2_kernel<<<grid, block, 0, streams[0]>>>(batch_size, place_cells_number, roi_col_begin, roi_col_end, x_grid, parameter.cx, parameter.width, parameter.gx2w, parameter.gx2w_strides);
		checkCudaErrors(cudaGetLastError());
	}
	{
		dim3 grid, block;

		block.x = BLOCK_X;
		block.y = BLOCK_Y;
		block.z = 1;

		grid.x = (place_cells_number + block.x - 1) / block.x;
		grid.y = (roi_rows + block.y - 1) / block.y;
		grid.z = (batch_size + block.z - 1) / block.z;

		diff2_kernel << <grid, block, 0, streams[0] >> >(batch_size, place_cells_number, roi_row_begin, roi_row_end, y_grid, parameter.cy, parameter.width, parameter.gy2w, parameter.gy2w_strides);
		checkCudaErrors(cudaGetLastError());
	}
}

__device__
struct float44
{
	float4 a, b, c, d;
};

__device__ __forceinline__
float44 inline operator * (const float44 &a, const float44 &b)
{
	float44 c;

	c.a = a.a * b.a;
	c.b = a.b * b.b;
	c.c = a.c * b.c;
	c.d = a.d * b.d;

	return c;
}
__device__ __forceinline__
float44 inline operator + (const float44 &a, const float44 &b)
{
	float44 c;

	c.a = a.a + b.a;
	c.b = a.b + b.b;
	c.c = a.c + b.c;
	c.d = a.d + b.d;

	return c;
}

__device__ __forceinline__
static inline float4 tanhf(const float4 a)
{
	float4 b;

	b.x = tanhf(a.x);
	b.y = tanhf(a.y);
	b.z = tanhf(a.z);
	b.w = tanhf(a.w);

	return (b);
}
__device__ __forceinline__
static inline float4 expf(const float4 a)
{
	float4 b;

	b.x = expf(a.x);
	b.y = expf(a.y);
	b.z = expf(a.z);
	b.w = expf(a.w);

	return (b);
}
__device__ __forceinline__
static inline float col_to_x(const int &col, const int &cols, const float &x_min, const float &x_range)
{
	return ((col) / (float)(cols - 1)) * x_range + x_min;
}
__device__ __forceinline__
static inline float row_to_y(const int &row, const int &rows, const float &y_min, const float &y_range)
{
	return ((row) / (float)(rows - 1)) * y_range + y_min;
}


template <bool use_circle>
__device__ __forceinline__
static inline float4 inside_circle(
	const int batch,
	const float radius2,
	const unsigned long seed,
	const float2 previous_position,
	const float2 current_position,
	const int row, const int col0,
	const int rows, const int cols,
	const float &x_min, const float &x_range, const float &y_min, const float &y_range,
	const float4 p, const float &cos_half_angle, const float &scale)
{
	return p;
}


template <>
__device__ __forceinline__
static inline float4 inside_circle<true>(
	const int batch,
	const float radius2,
	const unsigned long seed,
	const float2 previous_position,
	const float2 current_position,
	const int row, const int col0,
	const int rows, const int cols,
	const float &x_min, const float &x_range, const float &y_min, const float &y_range,
	const float4 p, const float &cos_half_angle, const float &scale)
{
		auto a = current_position - previous_position;
		auto da = dot(a, a);

		float4 bx;
		float by;

		bx.x = col_to_x(col0 + 0, cols, x_min, x_range) - current_position.x;
		bx.y = col_to_x(col0 + 1, cols, x_min, x_range) - current_position.x;
		bx.z = col_to_x(col0 + 2, cols, x_min, x_range) - current_position.x;
		bx.w = col_to_x(col0 + 3, cols, x_min, x_range) - current_position.x;
		by = (row_to_y(row, rows, y_min, y_range) - current_position.y);
		const float b2y = by * by;
		const float4 dp_b2 = bx * bx + b2y;


		int4 in;

		in.x = dp_b2.x < radius2;
		in.y = dp_b2.y < radius2;
		in.z = dp_b2.z < radius2;
		in.w = dp_b2.w < radius2;

		if (da > 0.0f && cos_half_angle < 1.0f)
		{
			a *= rsqrtf(da);

			const float aby = a.y * by;

			float4 dp_ab = (a.x * bx + aby);
			if (dp_b2.x > 0.0f)
			{
				in.x &= cos_half_angle < dp_ab.x * rsqrtf(dp_b2.x);
			}
			if (dp_b2.y > 0.0f)
			{
				in.y &= cos_half_angle < dp_ab.y * rsqrtf(dp_b2.y);
			}
			if (dp_b2.z > 0.0f)
			{
				in.z &= cos_half_angle < dp_ab.z * rsqrtf(dp_b2.z);
			}
			if (dp_b2.w > 0.0f)
			{
				in.w &= cos_half_angle < dp_ab.w * rsqrtf(dp_b2.w);
			}
		}
		

		curandStatePhilox4_32_10_t state;

		// seed a random number generator
		curand_init(seed + col0 * rows + row + batch * rows * cols, 0, 0, &state);

		auto r = curand_uniform4(&state) * scale;

		float4 s;
		s.x = in.x ? p.x + r.x : 0.0f;
		s.y = in.y ? p.y + r.y : 0.0f;
		s.z = in.z ? p.z + r.z : 0.0f;
		s.w = in.w ? p.w + r.w : 0.0f;

		return s;
}

template <bool use_circle>
__global__
static void decode_placecells_bayesian_kernel(
	const int batch_size,
	const int rows, const int cols,
	const int place_cells_number,
	const int roi_rows, const int roi_cols,
	const std::size_t *roi_row_begin, const std::size_t *roi_row_end, const std::size_t *roi_col_begin, const std::size_t *roi_col_end,
	const Model parameter,
	const float x_min, const float x_range, const float y_min, const float y_range,
	const unsigned long seed,
	const float scale,
	const float radius2,
	const float cos_half_angle,
	const float inv_sigma2,
	const float **batched_previous_position, 
	const float **batched_current_position,
	const float **batched_predicted_activations,

	float **batched_location_probability, const int location_probability_stride
)
{
	for (int batch = blockIdx.z * blockDim.z + threadIdx.z; batch < batch_size; batch += gridDim.z * blockDim.z)
	{
		//const int roi_valid_rows = min((int)(roi_row_end[batch] - roi_row_begin[batch]), rows);
	//	const int roi_valid_cols = min((int)(roi_col_end[batch] - roi_col_begin[batch]), cols);
		float *location_probability = batched_location_probability[batch];
		const float *previous_position = batched_previous_position[batch];
		const float *current_position = batched_current_position[batch];
		const float *predicted_activations = batched_predicted_activations[batch];

		for (int roi_row = blockIdx.y * blockDim.y + threadIdx.y; roi_row < roi_rows; roi_row += gridDim.y * blockDim.y)
		{
			for (int roi_col = blockIdx.x * blockDim.x + threadIdx.x; roi_col < roi_cols / 4; roi_col += gridDim.x * blockDim.x)
			{
				const int roi_col4 = roi_col * 4;
				const int row = roi_row_begin[batch] + roi_row;
				const int col = roi_col_begin[batch] + roi_col4;

				float4 gx;
				float gy;

				gx.x = col_to_x(col + 0, cols, x_min, x_range);
				gx.y = col_to_x(col + 1, cols, x_min, x_range);
				gx.z = col_to_x(col + 2, cols, x_min, x_range);
				gx.w = col_to_x(col + 3, cols, x_min, x_range);
				gy = row_to_y(row, rows, y_min, y_range);

				float4 v = make_float4(0.0f);
#pragma unroll 
				for (int pc = 0; pc < place_cells_number; pc++)
				{
					/*float4 p = reinterpret_cast<float4*>(const_cast<float *>(predicted_activations))[pc];
					float4 cx = reinterpret_cast<float4*>(const_cast<float *>(parameter.cx))[pc];
					float4 cy = reinterpret_cast<float4*>(const_cast<float *>(parameter.cy))[pc];
					float4 w = reinterpret_cast<float4*>(const_cast<float *>(parameter.width))[pc];*/
					float p = predicted_activations[pc];
					float cx = parameter.cx[pc];
					float cy = parameter.cy[pc];
					float w = parameter.width[pc];
					auto dx = gx - cx;
					auto dy = gy - cy;

					float4 d = expf((dx * dx + dy * dy) * w) - p;
					v += d * d;
				}
				float4 p = expf(inv_sigma2 * v);
				float4 l = inside_circle<use_circle>(batch,
					radius2,
					seed,
					*reinterpret_cast<float2 *>(const_cast<float *>(previous_position)),
					*reinterpret_cast<float2 *>(const_cast<float *>(current_position)),

					row, col,
					rows, cols,
					x_min, x_range, y_min, y_range,
					p, cos_half_angle, scale);
				reinterpret_cast<float4 *>(&location_probability[roi_row * location_probability_stride])[roi_col] = l;
			}
		}
	}
}

void decode_placecells_bayesian(
	const cudaStream_t *streams,
	const cublasHandle_t *handles,
	const cudaEvent_t *events,
	const std::size_t &batch_size, const std::size_t &place_cells_number,
	const std::size_t &rows, const std::size_t &cols,
	const std::size_t &roi_rows, const std::size_t &roi_cols,
	const std::size_t *roi_row_begin, const std::size_t *roi_row_end, const std::size_t *roi_col_begin, const std::size_t *roi_col_end,
	const float &radius,
	const float &cos_half_angle,
	const float &scale,
	const float &sigma,
	const unsigned long &seed,
	const Model &parameter,
	const float &x_min, const float &x_max, const float &y_min, const float &y_max,
	const float *x_grid, const std::size_t &x_grid_stride,
	const float *y_grid, const std::size_t &y_grid_stride,
	const float **batched_previous_position, const std::size_t &batched_previous_position_stride,
	const float **batched_current_position, const std::size_t &batched_current_position_stride,
	const float **batched_predicted_activations, const std::size_t &batched_predicted_activations_stride,
	float **batched_direction, const std::size_t &batched_direction_stride,
	float **batched_x_grid_centered, const std::size_t &batched_x_grid_centered_stride,
	float **batched_y_grid_centered, const std::size_t &batched_y_grid_centered_stride,
	float **batched_location_probability, const std::size_t &batched_location_probability_stride)
{
	//prepare(streams, handles, events, parameter, batch_size, place_cells_number, rows, cols, x_grid, y_grid, roi_rows, roi_cols, roi_row_begin, roi_row_end, roi_col_begin, roi_col_end);

	dim3 grid, block;

	block.x = BLOCK_X;
	block.y = BLOCK_Y;
	block.z = 1;

	grid.x = (roi_cols / 4 + block.x - 1) / block.x;
	grid.y = (roi_rows + block.y - 1) / block.y;
	grid.z = (batch_size  + block.z - 1) / block.z;

	/*for (std::size_t batch = 0; batch < batch_size; batch++)
	{*/
	//	auto stream_number = batch % TRN::GPU::Context::STREAM_NUMBER;*/
		if (radius > 0.0f)
		{

				decode_placecells_bayesian_kernel<true> << <grid, block, 0, streams[0] >> > (
					batch_size,
					rows, cols, place_cells_number,
					roi_rows, roi_cols,
					roi_row_begin, roi_row_end, roi_col_begin, roi_col_end,
					parameter,
					x_min, x_max - x_min,
					y_min, y_max - y_min,
					seed,
					scale,
					radius * radius,
					cos_half_angle,
					-1.0f / (2.0f * sigma * sigma),
					batched_previous_position,
					batched_current_position,
					batched_predicted_activations,
					batched_location_probability, batched_location_probability_stride
					);
		
		}
		else
		{
			decode_placecells_bayesian_kernel<false> << <grid, block, 0, streams[0] >> > (
				batch_size,
				rows, cols, place_cells_number,
				roi_rows, roi_cols,
				roi_row_begin, roi_row_end, roi_col_begin, roi_col_end,
				parameter,
				x_min, x_max - x_min,
				y_min, y_max - y_min,
				seed,
				scale,
				radius * radius,
				cos_half_angle,
				-1.0f / (2.0f * sigma * sigma),
				batched_previous_position,
				batched_current_position,
				batched_predicted_activations,
				batched_location_probability, batched_location_probability_stride
				);
		}
		checkCudaErrors(cudaGetLastError());
	/*}*/

	//checkCudaErrors(cudaDeviceSynchronize());
}

template <bool use_circle>
__global__
static void decode_placecells_bayesian_kernel(
	const int batch,
	const int rows, const int cols,
	const int place_cells_number,
	const int roi_rows, const int roi_cols,
	const std::size_t *roi_row_begin, const std::size_t *roi_row_end, const std::size_t *roi_col_begin, const std::size_t *roi_col_end,
	const Map parameter,
	const float x_min, const float x_range, const float y_min, const float y_range,
	const unsigned long seed,
	const float scale,
	const float radius2,
	const float cos_half_angle,
	const float inv_sigma2,
	const float **batched_previous_position,
	const float **batched_current_position,
	const float **batched_predicted_activations,

	float **batched_location_probability, const int location_probability_stride
)
{
	const int roi_row = blockIdx.z * blockDim.z + threadIdx.z;
	const int roi_col = blockIdx.y * blockDim.y + threadIdx.y;
	const int pc = blockIdx.x * blockDim.x + threadIdx.x;

	float *location_probability = batched_location_probability[batch];
	const float *previous_position = batched_previous_position[batch];
	const float *current_position = batched_current_position[batch];
	const float *predicted_activations = batched_predicted_activations[batch];
	const int row = roi_row_begin[batch] + roi_row;
	const int col = roi_col_begin[batch] + roi_col * 4;
	const int roi_valid_cols = roi_col_end[batch] - roi_col_begin[batch];
	const int roi_valid_rows = roi_row_end[batch] - roi_row_begin[batch];

	if (roi_row < roi_valid_rows && roi_col * 4 < roi_valid_cols)
	{
		typedef cub::BlockReduce<float4, 256, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY> BlockReduceT;
		__shared__ typename BlockReduceT::TempStorage temp_storage;

		const float p_ = predicted_activations[threadIdx.x];

		//const int rc = ;
		float4 d;
		d.x = parameter.data[(row * cols + (col + 0)) * parameter.stride + threadIdx.x] - p_;
		d.y = parameter.data[(row * cols + (col + 1)) * parameter.stride + threadIdx.x] - p_;
		d.z = parameter.data[(row * cols + (col + 2)) * parameter.stride + threadIdx.x] - p_;
		d.w = parameter.data[(row * cols + (col + 3)) * parameter.stride + threadIdx.x] - p_;

		float4 v = BlockReduceT(temp_storage).Sum(d * d);
		if (threadIdx.x == 0)
		{
			float4 p = expf(inv_sigma2 * v);
			float4 l = inside_circle<use_circle>(batch,
				radius2,
				seed,
				*reinterpret_cast<float2 *>(const_cast<float *>(previous_position)),
				*reinterpret_cast<float2 *>(const_cast<float *>(current_position)),

				row, col,
				rows, cols,
				x_min, x_range, y_min, y_range,
				p, cos_half_angle, scale);
			reinterpret_cast<float4 *>(&location_probability[roi_row * location_probability_stride])[roi_col] = l;
		}
	}
	else
	{
		if (threadIdx.x == 0)
		{
			reinterpret_cast<float4 *>(&location_probability[roi_row * location_probability_stride])[roi_col] = make_float4(0.0f);
		}
	}
}
void decode_placecells_bayesian(
	const cudaStream_t *streams,
	const cublasHandle_t *handles,
	const cudaEvent_t *events,
	const std::size_t &batch_size, const std::size_t &place_cells_number,
	const std::size_t &rows, const std::size_t &cols,
	const std::size_t &roi_rows, const std::size_t &roi_cols,
	const std::size_t *roi_row_begin, const std::size_t *roi_row_end, const std::size_t *roi_col_begin, const std::size_t *roi_col_end,
	const float &radius,
	const float &cos_half_angle,
	const float &scale,
	const float &sigma,
	const unsigned long &seed,
	const Map &parameter,
	const float &x_min, const float &x_max, const float &y_min, const float &y_max,
	const float *x_grid, const std::size_t &x_grid_stride,
	const float *y_grid, const std::size_t &y_grid_stride,
	const float **batched_previous_position, const std::size_t &batched_previous_position_stride,
	const float **batched_current_position, const std::size_t &batched_current_position_stride,
	const float **batched_predicted_activations, const std::size_t &batched_predicted_activations_stride,
	float **batched_direction, const std::size_t &batched_direction_stride,
	float **batched_x_grid_centered, const std::size_t &batched_x_grid_centered_stride,
	float **batched_y_grid_centered, const std::size_t &batched_y_grid_centered_stride,
	float **batched_location_probability, const std::size_t &batched_location_probability_stride)
{
	//prepare(streams, handles, events, parameter, batch_size, place_cells_number, rows, cols, x_grid, y_grid, roi_rows, roi_cols, roi_row_begin, roi_row_end, roi_col_begin, roi_col_end);

	dim3 grid, block;

	block.x = 256;
	block.y = 1;
	block.z = 1;

	grid.x = (256 + block.x - 1) / block.x;
	grid.y = (roi_cols / 4 + block.y - 1) / block.y;
	grid.z = (roi_rows + block.z - 1) / block.z;

	for (std::size_t batch = 0; batch < batch_size; batch++)
	{
	//	auto stream_number = batch % TRN::GPU::Context::STREAM_NUMBER;*/
	if (radius > 0.0f)
	{

		decode_placecells_bayesian_kernel<true> << <grid, block, 0, streams[0] >> > (
			batch,
			rows, cols, place_cells_number,
			roi_rows, roi_cols,
			roi_row_begin, roi_row_end, roi_col_begin, roi_col_end,
			parameter,
			x_min, x_max - x_min,
			y_min, y_max - y_min,
			seed,
			scale,
			radius * radius,
			cos_half_angle,
			-1.0f / (2.0f * sigma * sigma),
			batched_previous_position,
			batched_current_position,
			batched_predicted_activations,
			batched_location_probability, batched_location_probability_stride
			);

	}
	else
	{
		decode_placecells_bayesian_kernel<false> << <grid, block, 0, streams[0] >> > (
			batch,
			rows, cols, place_cells_number,
			roi_rows, roi_cols,
			roi_row_begin, roi_row_end, roi_col_begin, roi_col_end,
			parameter,
			x_min, x_max - x_min,
			y_min, y_max - y_min,
			seed,
			scale,
			radius * radius,
			cos_half_angle,
			-1.0f / (2.0f * sigma * sigma),
			batched_previous_position,
			batched_current_position,
			batched_predicted_activations,
			batched_location_probability, batched_location_probability_stride
			);
	}
	checkCudaErrors(cudaGetLastError());
	}

	//checkCudaErrors(cudaDeviceSynchronize());
}

void compute_mean_square_error
(
	const cudaStream_t *streams,
	const cublasHandle_t *handles,
	const std::size_t &batch_size,
	const float **batched_predicted, const std::size_t &batched_predicted_rows, const std::size_t &batched_predicted_cols, const std::size_t &batched_predicted_stride,
	const float *expected, const std::size_t &expected_rows, const std::size_t &expected_cols, const std::size_t &expected_stride,
	float *result, const std::size_t &result_rows, const std::size_t &result_cols, const std::size_t &result_stride
)
{

	throw std::runtime_error("Not yet implemented");
}


/*__global__
static void decode_placecells_linear_kernel(
	const int batch_size, const int place_cells_number,
	const float *cx,
	const float *cy,
	const float **batched_prediction, const int batched_prediction_strides,
	float **batched_decoded_position, const int batched_decoded_position_strides)
{
	cudaStream_t *streams;
	cublasHandle_t *handless;

	assert(cudaMalloc(&streams, sizeof(cudaStream_t) * batch_size));
	assert(cudaMalloc(&handles, sizeof(cublasHandle_t) * batch_size));


	for (int batch = 0; batch < batch_size; batch++)
	{
		float sum;
		float x, y;

		assert(cudaStreamCreateWithFlags(&streams[batch], cudaStreamNonBlocking));
		assert(cublasCreate(&handles[batch]));
		assert(cublasSetStream(handles[batch], streams[batch]));
		assert(cublasSasum(handles[batch], place_cells_number, batched_prediction[batch], 1, &sum));
		assert(cublasSdot(handles[batch], place_cells_number, batched_prediction[batch], 1, cx, 1, &x));
		assert(cublasSdot(handles[batch], place_cells_number, batched_prediction[batch], 1, cy, 1, &y));

		batched_decoded_position[batch][0] = x / sum;
		batched_decoded_position[batch][1] = y / sum;
	}


	for (int batch = 0; batch < batch_size; batch++)
	{	
		assert(cudaStreamSynchronize(streams[batch]));
		assert(cudaStreamDestroy(streams[batch]));
		
		assert(cublasDestroy(handles[batch]));

	}
	assert(cudaFree(streams));
	assert(cudaFree(handles));
}*/

void compute_decode_placecells_linear(
	const cudaStream_t *streams,
	const cublasHandle_t *handles,
	const std::size_t &batch_size, const std::size_t &place_cells_number,
	const float *cx,
	const float *cy,
	const float **batched_prediction, const std::size_t &batched_prediction_strides,
	float **batched_decoded_position, const std::size_t &batched_decoded_position_strides)
{

	for (int batch = 0; batch < batch_size; batch++)
	{
		float sum;
		float x, y;

		checkCudaErrors(cublasSasum(*handles, place_cells_number, batched_prediction[batch], 1, &sum));
		checkCudaErrors(cublasSdot(*handles, place_cells_number, batched_prediction[batch], 1, cx, 1, &x));
		checkCudaErrors(cublasSdot(*handles, place_cells_number, batched_prediction[batch], 1, cy, 1, &y));

		float position[2] = { x / sum, y / sum };

		checkCudaErrors(cudaMemcpyAsync(batched_decoded_position[batch], position, sizeof(position), cudaMemcpyKind::cudaMemcpyHostToDevice, *streams));

	}

	/*decode_placecells_linear_kernel << <1, 1, 0, *streams>> > (batch_size, place_cells_number, cx, cy, batched_prediction, batched_prediction_strides, batched_decoded_position, batched_decoded_position_strides);
	checkCudaErrors(cudaGetLastError());*/
}
__global__
static void encode_placecells_model_kernel(const int batch_size, const int place_cells_number,
	const float *__restrict__ cx,
	const float *__restrict__ cy,
	const float *__restrict__ w,
	const float **__restrict__ batched_decoded_position, const int batched_decoded_position_strides,
	float **__restrict__ batched_stimulus, const int batched_stimulus_strides
)
{
	for (int batch = blockIdx.y * blockDim.y + threadIdx.y; batch < batch_size; batch += gridDim.y * blockDim.y)
	{
		auto p = *reinterpret_cast<float2 *>(const_cast<float *>(batched_decoded_position[batch]));
		auto stimulus= batched_stimulus[batch];
		for (int place_cell = blockIdx.x * blockDim.x + threadIdx.x; place_cell < place_cells_number; place_cell += gridDim.x * blockDim.x)
		{
			auto dx = p.x - cx[place_cell];
			auto dy = p.y - cy[place_cell];
			stimulus[place_cell] = expf((dx * dx + dy * dy) * w[place_cell]);
		}
	}
}

void compute_encode_placecells_model(const cudaStream_t *streams,
	const cublasHandle_t *handles,
	const cudaEvent_t *events,
	const std::size_t &batch_size, const std::size_t &place_cells_number,
	const float *cx,
	const float *cy,
	const float *w,
	const float **batched_decoded_position, const std::size_t &batched_decoded_position_strides,
	float **batched_stimulus, const std::size_t &batched_stimulus_strides
)
{
	dim3 grid, block;

	block.x = BLOCK_X;
	block.y = 1;

	grid.x = (place_cells_number  + block.x - 1) / block.x;
	grid.y = (batch_size + block.y - 1) / block.y;

	encode_placecells_model_kernel << <grid, block, 0, streams[0] >> > (
		batch_size, place_cells_number, 
		cx, cy, w, 
		batched_decoded_position, batched_decoded_position_strides,
		batched_stimulus, batched_stimulus_strides);

	checkCudaErrors(cudaGetLastError());
}

__global__
static void update_reservoir_kernel(
	const int batch_size, const int mini_batch, const int t,
	const float leak_rate,
	const float ** __restrict__ batched_u, const int batched_u_rows, const int batched_u_cols, const int batched_u_stride,
	const float ** __restrict__ batched_u_ffwd, const int batched_u_ffwd_rows, const int batched_u_ffwd_cols, const int batched_u_ffwd_stride,

	float ** __restrict__ batched_p, const int batched_p_rows, const int batched_p_cols, const int batched_p_stride,
	float ** __restrict__ batched_x_res, const int batched_x_res_rows, const int batched_x_res_cols, const int batched_x_res_stride,
	float ** __restrict__ batched_pre, const int batched_pre_rows, const int batched_pre_cols, const int batched_pre_stride
)
{
	for (int batch = blockIdx.y * blockDim.y + threadIdx.y; batch < batch_size; batch += gridDim.y * blockDim.y)
	{
		float4 *U = reinterpret_cast<float4 *>(const_cast<float *>(batched_u[batch]));
		float4 *U_ffwd = reinterpret_cast<float4 *>(const_cast<float *>(&batched_u_ffwd[batch][t * batched_u_ffwd_stride]));
		float4 *X = reinterpret_cast<float4 *>(batched_x_res[batch]);
		float4 *P = reinterpret_cast<float4 *>(batched_p[batch]);
		float4 *PRE = reinterpret_cast<float4 *>(&batched_pre[batch][mini_batch * batched_pre_stride]);

		for (int col = blockIdx.x * blockDim.x + threadIdx.x; col < batched_x_res_cols / 4; col += gridDim.x * blockDim.x)
		{
			auto p = P[col];
			p += leak_rate * (U[col] + U_ffwd[col] - p);
			auto x = tanhf(p);
			P[col] = p;
			X[col] = x;	
			PRE[col] = x;
		}
	}
}

__host__
static inline void update_reservoir
(
	const cudaStream_t *streams,
	const cublasHandle_t *handles,
	const cudaEvent_t *events,
	const std::size_t &batch_size, const std::size_t &mini_batch_size, const std::size_t &mini_batch, const std::size_t &bundle,
	const int *offsets, const std::size_t &ts, const std::size_t &total_duration,
	const float **batched_w_rec, const std::size_t &batched_w_rec_rows, const std::size_t &batched_w_rec_cols, const  std::size_t &batched_w_rec_stride,
	float **batched_x_res, const std::size_t &batched_x_res_rows, const std::size_t &batched_x_res_cols, const  std::size_t &batched_x_res_stride,
	float **batched_u, const std::size_t &batched_u_rows, const std::size_t &batched_u_cols, const  std::size_t &batched_u_stride,
	const float **batched_u_ffwd, const std::size_t &batched_u_ffwd_rows, const std::size_t &batched_u_ffwd_cols, const  std::size_t &batched_u_ffwd_stride,
	float **batched_p, const std::size_t &batched_p_rows, const std::size_t &batched_p_cols, const  std::size_t &batched_p_stride,
	float ***bundled_pre, const std::size_t &bundled_pre_rows, const std::size_t &bundled_pre_cols, const  std::size_t &bundled_pre_stride,
	const float &leak_rate,
	const float *one, const float *zero
)
{
	sgemm(
			handles[0],
			cublasOperation_t::CUBLAS_OP_N,
			cublasOperation_t::CUBLAS_OP_N,
			batch_size,

			(const float **)batched_w_rec, batched_w_rec_cols, batched_w_rec_rows, batched_w_rec_stride,
			(const float **)batched_x_res, batched_x_res_cols, batched_x_res_rows, batched_x_res_stride,
			batched_u, batched_u_cols, batched_u_rows, batched_u_stride, one, zero);

	auto t = offsets[ts];
	dim3 grid, block;

	block.x = warpSize * 4;
	block.y = 1;

	grid.x = (batched_x_res_cols / 4 + block.x - 1) / block.x;
	grid.y = (batch_size + block.y - 1) / block.y;

	update_reservoir_kernel << < grid, block, 0, streams[0] >> > (
		batch_size, mini_batch, t,
		leak_rate,
		(const float **)batched_u, batched_u_rows, batched_u_cols, batched_u_stride,
		(const float **)batched_u_ffwd, batched_u_ffwd_rows, batched_u_ffwd_cols, batched_u_ffwd_stride,
		batched_p, batched_p_rows, batched_p_cols, batched_p_stride,
		batched_x_res, batched_x_res_rows,  batched_x_res_cols, batched_x_res_stride,
		bundled_pre[bundle], bundled_pre_rows, bundled_pre_cols, bundled_pre_stride);
	checkCudaErrors(cudaGetLastError());
}


template <bool gather_states>
static void copy_states(
	const cudaStream_t *streams, const std::size_t &batch_size, const std::size_t &t, const std::size_t &ts,
	const std::size_t &stimulus_size,
	const std::size_t &reservoir_size,
	const std::size_t &prediction_size,
	const std::size_t &stimulus_stride,
	const std::size_t &reservoir_stride,
	const std::size_t &prediction_stride,
	const float **batched_incoming, const std::size_t &batched_incoming_rows, const std::size_t &batched_incoming_cols, const std::size_t &batched_incoming_strides,
	const float **batched_expected, const std::size_t &batched_expected_rows, const std::size_t &batched_expected_cols, const std::size_t &batched_expected_strides,
	const float **batched_x_ro, const std::size_t &batched_x_ro_rows, const std::size_t &batched_x_ro_cols, const std::size_t &batched_x_ro_strides,
	const float **batched_x_res, const std::size_t &batched_x_res_rows, const std::size_t &batched_x_res_cols, const std::size_t &batched_x_res_strides,
	float *states, const std::size_t &states_rows, const std::size_t &states_cols, const std::size_t &states_stride)
{}

template <>
static void copy_states<true>(const cudaStream_t *streams, const std::size_t &batch_size, const std::size_t &t, const std::size_t &ts,
	const std::size_t &stimulus_size,
	const std::size_t &reservoir_size,
	const std::size_t &prediction_size,
	const std::size_t &stimulus_stride,
	const std::size_t &reservoir_stride,
	const std::size_t &prediction_stride,
	const float **batched_incoming, const std::size_t &batched_incoming_rows, const std::size_t &batched_incoming_cols, const std::size_t &batched_incoming_strides,
	const float **batched_expected, const std::size_t &batched_expected_rows, const std::size_t &batched_expected_cols, const std::size_t &batched_expected_strides,
	const float **batched_x_ro, const std::size_t &batched_x_ro_rows, const std::size_t &batched_x_ro_cols, const std::size_t &batched_x_ro_strides,
	const float **batched_x_res, const std::size_t &batched_x_res_rows, const std::size_t &batched_x_res_cols, const std::size_t &batched_x_res_strides,
	float *states, const std::size_t &states_rows, const std::size_t &states_cols, const std::size_t &states_stride)
{	
	std::vector<float *> incoming_ptr(batch_size);
	std::vector<float *> expected_ptr(batch_size);
	std::vector<float *> x_ro_ptr(batch_size);
	std::vector<float *> x_res_ptr(batch_size);
	/*cudaStream_t incoming, expected, x_ro, x_res;*/
	/*cudaEvent_t incoming_terminated, expected_terminated, x_ro_terminated, x_res_terminated;
	checkCudaErrors(cudaStreamCreateWithFlags(&incoming, cudaStreamNonBlocking));
	checkCudaErrors(cudaStreamCreateWithFlags(&expected, cudaStreamNonBlocking));
	checkCudaErrors(cudaStreamCreateWithFlags(&x_ro, cudaStreamNonBlocking));
	checkCudaErrors(cudaStreamCreateWithFlags(&x_res, cudaStreamNonBlocking));*/
	checkCudaErrors(cudaMemcpyAsync(incoming_ptr.data(), batched_incoming, batch_size * sizeof(float *), cudaMemcpyKind::cudaMemcpyDeviceToHost, *streams));
	checkCudaErrors(cudaMemcpyAsync(expected_ptr.data(), batched_expected, batch_size * sizeof(float *), cudaMemcpyKind::cudaMemcpyDeviceToHost, *streams));
	checkCudaErrors(cudaMemcpyAsync(x_ro_ptr.data(), batched_x_ro, batch_size * sizeof(float *), cudaMemcpyKind::cudaMemcpyDeviceToHost, *streams));
	checkCudaErrors(cudaMemcpyAsync(x_res_ptr.data(), batched_x_res, batch_size * sizeof(float *), cudaMemcpyKind::cudaMemcpyDeviceToHost, *streams));

	/*checkCudaErrors(cudaEventCreate(&incoming_terminated));
	checkCudaErrors(cudaEventCreate(&expected_terminated));
	checkCudaErrors(cudaEventCreate(&x_ro_terminated));
	checkCudaErrors(cudaEventCreate(&x_res_terminated));

	checkCudaErrors(cudaStreamSynchronize(incoming));
	checkCudaErrors(cudaStreamSynchronize(expected));
	checkCudaErrors(cudaStreamSynchronize(x_ro));*/
	checkCudaErrors(cudaStreamSynchronize(*streams));

//	std::size_t offset = 0;
	float *states_ts = &states[ts * states_stride];

	for (std::size_t batch = 0; batch < batch_size; batch++)
	{
		std::size_t offset = 0;
		std::size_t  stimulus_col = batch * stimulus_stride + batch_size * offset;
		checkCudaErrors(cudaMemcpyAsync(
			states_ts + stimulus_col,
			&incoming_ptr[batch][t * batched_incoming_strides],
			sizeof(float) * stimulus_size, cudaMemcpyKind::cudaMemcpyDeviceToDevice, *streams));
		offset += stimulus_stride;

		std::size_t  desired_col = batch * prediction_stride + batch_size * offset;
		checkCudaErrors(cudaMemcpyAsync(
			states_ts + desired_col,
			&expected_ptr[batch][t * batched_expected_strides],
			sizeof(float) * prediction_size, cudaMemcpyKind::cudaMemcpyDeviceToDevice, *streams));
		offset += prediction_stride;

		std::size_t  reservoir_col = batch * reservoir_stride + batch_size * offset;
		checkCudaErrors(cudaMemcpyAsync(
			states_ts + reservoir_col,
			x_res_ptr[batch],
			sizeof(float) * reservoir_size, cudaMemcpyKind::cudaMemcpyDeviceToDevice, *streams));
		offset += reservoir_stride;

		std::size_t  predicted_col = batch * prediction_stride + batch_size * offset;
		checkCudaErrors(cudaMemcpyAsync(
			states_ts + predicted_col,
			x_ro_ptr[batch],
			sizeof(float) * prediction_size, cudaMemcpyKind::cudaMemcpyDeviceToDevice, *streams));
		offset += prediction_stride;
	}
	
}


template <bool overwrite_states>
static inline void initialize_states(const cudaStream_t *streams,  unsigned long &seed,
	const std::size_t &batch_size,
	float **batched_ptr, const std::size_t &batched_ptr_rows, const std::size_t &batched_ptr_cols, const std::size_t &batched_ptr_stride,
	const float &initial_state_scale)
{
}
template <>
static inline void initialize_states<true>(const cudaStream_t *streams, unsigned long &seed,
	const std::size_t &batch_size,
	float **batched_ptr, const std::size_t &batched_ptr_rows, const std::size_t &batched_ptr_cols, const std::size_t &batched_ptr_stride,
	const float &initial_state_scale)
{
	random_uniform(*streams, seed, -initial_state_scale, initial_state_scale, 0.0f, batch_size, batched_ptr_rows, batched_ptr_cols, batched_ptr, batched_ptr_stride, false);
	seed += batch_size * batched_ptr_rows * batched_ptr_cols;
}





__global__
static void update_readout_activations_kernel(
	const int batch_size, 
	float ** __restrict__ batched_x, const int batched_x_rows, const int batched_x_cols, const int batched_x_stride
)
{
	for (int batch = blockIdx.y * blockDim.y + threadIdx.y; batch < batch_size; batch += gridDim.y * blockDim.y)
	{
		float *X = batched_x[batch];
		for (int col = blockIdx.x * blockDim.x + threadIdx.x; col < batched_x_cols; col += gridDim.x * blockDim.x)
		{
			X[col] = tanhf(X[col]);
		}
	}
}

__global__
static void update_readout_activations_kernel(
	const int batch_size,
	const float ** __restrict__ batched_d, const int batched_d_rows, const int batched_d_cols, const int batched_d_stride,
	float ** __restrict__ batched_x, const int batched_x_rows, const int batched_x_cols, const int batched_x_stride
)
{
	for (int batch = blockIdx.z * blockDim.z + threadIdx.z; batch < batch_size; batch += gridDim.z * blockDim.z)
	{
		const float *D = batched_d[batch];
		float *X = batched_x[batch];
		for (int row = blockIdx.y * blockDim.y + threadIdx.y; row < batched_x_rows; row += gridDim.y * blockDim.y)
		{
			for (int col = blockIdx.x * blockDim.x + threadIdx.x; col < batched_x_cols; col += gridDim.x * blockDim.x)
			{
				auto x = tanhf(X[row * batched_x_stride + col]);
				X[row * batched_x_stride + col] = (x - D[row * batched_d_stride + col]) * (x * x - 1.0f);
			}
		}
	}
}

__global__
static void update_readout_desired_kernel(
	const int batch_size, const int t, const int mini_batch,
	const float ** __restrict__ batched_expected, const int batched_expected_rows, const int batched_expected_cols, const int batched_expected_stride,
	float ** __restrict__ batched_desired, const int batched_desired_rows, const int batched_desired_cols, const int batched_desired_stride
)
{
	for (int batch = blockIdx.y * blockDim.y + threadIdx.y; batch < batch_size; batch += gridDim.y * blockDim.y)
	{
		const float *E = &batched_expected[batch][t * batched_expected_stride];
		float *D = &batched_desired[batch][mini_batch * batched_desired_stride];
		for (int col = blockIdx.x * blockDim.x + threadIdx.x; col < batched_desired_cols; col += gridDim.x * blockDim.x)
		{
			D[col] = E[col];
		}
	}
}
/*template<class T>
__global__ void Global_write(T* out, T value, size_t N)
{
	size_t i;
	for (i = 4 * blockDim.x*blockIdx.x + threadIdx.x;
		i < N - 4 * blockDim.x*blockIdx.x;
		i += 4 * gridDim.x*blockDim.x;) {
		out[i + 0 * blockDim.x] = value;
		out[i + 1 * blockDim.x] = value;
		out[i + 2 * blockDim.x] = value;
		out[i + 3 * blockDim.x] = value;
	}
	if (i + 0 * blockDim.x < N) out[i + 0 * blockDim.x] = value;
	if (i + 1 * blockDim.x < N) out[i + 1 * blockDim.x] = value;
	if (i + 2 * blockDim.x < N) out[i + 2 * blockDim.x] = value;
	if (i + 3 * blockDim.x < N) out[i + 3 * blockDim.x] = value;
}*/


static inline void update_readout(
	const cudaStream_t *streams,
	const cublasHandle_t *handles,
	const cudaEvent_t *events,
	const std::size_t &batch_size, const std::size_t &mini_batch_size, std::size_t &mini_batch, std::size_t &bundle,
	const int *offsets, const std::size_t &ts, const std::size_t &total_duration,
	const Nothing &parameter,
	 float **batched_x_res, const std::size_t &batched_x_res_rows, const std::size_t &batched_x_res_cols, const std::size_t &batched_x_res_strides,
	float **batched_x_ro, const std::size_t &batched_x_ro_rows, const std::size_t &batched_x_ro_cols, const std::size_t &batched_x_ro_strides,
	 float **batched_expected, const std::size_t &batched_expected_rows, const std::size_t &batched_expected_cols, const std::size_t &batched_expected_strides,
	 float ***bundled_pre, const std::size_t &bundled_pre_rows, const std::size_t &bundled_pre_cols, const std::size_t &bundled_pre_strides,
	float **batched_post, const std::size_t &batched_post_rows, const std::size_t &batched_post_cols, const std::size_t &batched_post_strides,
	float ***bundled_desired, const std::size_t &bundled_desired_rows, const std::size_t &bundled_desired_cols, const std::size_t &bundled_desired_strides,
	float **batched_w_ro, const std::size_t &batched_w_ro_rows, const std::size_t &batched_w_ro_cols, const std::size_t &batched_w_ro_strides,
	const float *one, const float *zero)
{
	sgemm(
		handles[0],
		cublasOperation_t::CUBLAS_OP_N, cublasOperation_t::CUBLAS_OP_N,
		batch_size,
		(const float **)batched_w_ro, batched_w_ro_cols, batched_w_ro_rows, batched_w_ro_strides,
		(const float **)batched_x_res, batched_x_res_cols, batched_x_res_rows, batched_x_res_strides,
		batched_x_ro, batched_x_ro_cols, batched_x_ro_rows, batched_x_ro_strides, one, zero
	);

	dim3 grid, block;

	block.x = warpSize * 4;
	block.y = 1;

	grid.x = (batched_x_ro_cols + block.x - 1) / block.x;
	grid.y = (batch_size + block.z - 1) / block.z;

	update_readout_activations_kernel << <grid, block, 0, streams[0] >> > (
		batch_size,
		batched_x_ro, batched_x_ro_cols, batched_x_ro_rows, batched_x_ro_strides
		);
	checkCudaErrors(cudaGetLastError());
}
static inline void update_readout(
	const cudaStream_t *streams,
	const cublasHandle_t *handles,
	const cudaEvent_t *events,
	const std::size_t &batch_size, const std::size_t &mini_batch_size, std::size_t &mini_batch, std::size_t &bundle,
	const int *offsets, const std::size_t &ts, const std::size_t &total_duration,
	const Widrow_Hoff &parameter,
	float **batched_x_res, const std::size_t &batched_x_res_rows, const std::size_t &batched_x_res_cols, const std::size_t &batched_x_res_strides,
	float **batched_x_ro, const std::size_t &batched_x_ro_rows, const std::size_t &batched_x_ro_cols, const std::size_t &batched_x_ro_strides,
	float **batched_expected, const std::size_t &batched_expected_rows, const std::size_t &batched_expected_cols, const std::size_t &batched_expected_strides,
	float ***bundled_pre, const std::size_t &bundled_pre_rows, const std::size_t &bundled_pre_cols, const std::size_t &bundled_pre_strides,
	float **batched_post, const std::size_t &batched_post_rows, const std::size_t &batched_post_cols, const std::size_t &batched_post_strides,
	float ***bundled_desired, const std::size_t &bundled_desired_rows, const std::size_t &bundled_desired_cols, const std::size_t &bundled_desired_strides,
	float **batched_w_ro, const std::size_t &batched_w_ro_rows, const std::size_t &batched_w_ro_cols, const std::size_t &batched_w_ro_strides,
	const float *one, const float *zero)
{
	const int stream_number = 1;
	if (ts < total_duration - 1)
	{
		const auto t = offsets[ts + 1];

		dim3 grid, block;

		block.x = warpSize * 4;
		block.y = 1;


		grid.x = (batched_expected_cols + block.x - 1) / block.x;
		grid.y = (batch_size + block.y - 1) / block.y;

		update_readout_desired_kernel << <grid, block, 0, streams[0] >> > (
			batch_size, t, mini_batch,
			(const float **)batched_expected, batched_expected_rows, batched_expected_cols, batched_expected_strides,
			bundled_desired[bundle], bundled_desired_rows, bundled_desired_cols, bundled_desired_strides
			);
		checkCudaErrors(cudaGetLastError());
	}
	mini_batch++;
	if (mini_batch == mini_batch_size || ts == total_duration - 1)
	{
		sgemm(
			handles[stream_number],
			cublasOperation_t::CUBLAS_OP_N, cublasOperation_t::CUBLAS_OP_N,
			batch_size,
			(const float **)batched_w_ro, batched_w_ro_cols, batched_w_ro_rows, batched_w_ro_strides,
			(const float **)bundled_pre[bundle], bundled_pre_cols, mini_batch, bundled_pre_strides,
			batched_post, batched_post_cols, mini_batch, batched_post_strides, one, zero);

		dim3 block, grid;

		block.x = warpSize * 4;
		block.y = 1;
		block.z = 1;

		grid.x = (bundled_desired_cols + block.x - 1) / block.x;
		grid.y = (mini_batch + block.y - 1) / block.y;
		grid.z = (batch_size + block.z - 1) / block.z;

		update_readout_activations_kernel << <grid, block, 0, streams[stream_number] >> > (
			batch_size,
			(const float **)bundled_desired[bundle], mini_batch, bundled_desired_cols, bundled_desired_strides,
			batched_post, mini_batch, batched_post_cols, batched_post_strides
			);
		checkCudaErrors(cudaGetLastError());

		sgemm(
			handles[stream_number],
			cublasOperation_t::CUBLAS_OP_N, cublasOperation_t::CUBLAS_OP_T,

			batch_size,
			(const float **)batched_post, batched_post_cols, mini_batch, batched_post_strides,
			(const float **)bundled_pre[bundle], bundled_pre_cols, mini_batch, bundled_pre_strides,

			batched_w_ro, batched_w_ro_cols, batched_w_ro_rows, batched_w_ro_strides,
			 parameter.get_learning_rate(), one);

		mini_batch = 0;
		bundle = 1 - bundle;
	}
}

template<bool gather_states, bool overwrite_states, typename Parameter>
void update_model(
	const std::size_t &batch_size,
	const std::size_t &mini_batch_size,
	unsigned long &seed,
	const cudaStream_t *streams,
	const cublasHandle_t *handles,
	const cudaEvent_t *events,
	const Parameter &parameter,
	const std::size_t &stimulus_stride, const std::size_t &reservoir_stride, const std::size_t &prediction_stride,
	const std::size_t &stimulus_size, const std::size_t &reservoir_size, const std::size_t &prediction_size,
	const float &leak_rate, const float &initial_state_scale,
	float **batched_incoming, const std::size_t &batched_incoming_rows, const std::size_t &batched_incoming_cols, const std::size_t &batched_incoming_strides,
	float **batched_expected, const std::size_t &batched_expected_rows, const std::size_t &batched_expected_cols, const std::size_t &batched_expected_strides,
	float **batched_w_ffwd, const std::size_t &batched_w_ffwd_rows, const std::size_t &batched_w_ffwd_cols, const std::size_t &batched_w_ffwd_strides,
	float **batched_u_ffwd, const std::size_t &batched_u_ffwd_rows, const std::size_t &batched_u_ffwd_cols, const std::size_t &batched_u_ffwd_strides,
	float **batched_x_res, const std::size_t &batched_x_res_rows, const std::size_t &batched_x_res_cols, const std::size_t &batched_x_res_strides,
	float **batched_w_rec, const std::size_t &batched_w_rec_rows, const std::size_t &batched_w_rec_cols, const std::size_t &batched_w_rec_strides,
	float **batched_u, const std::size_t &batched_u_rows, const std::size_t &batched_u_cols, const std::size_t &batched_u_strides,
	float **batched_p, const std::size_t &batched_p_rows, const std::size_t &batched_p_cols, const std::size_t &batched_p_strides,
	float **batched_x_ro, const std::size_t &batched_x_ro_rows, const std::size_t &batched_x_ro_cols, const std::size_t &batched_x_ro_strides,
	float **batched_w_ro, const std::size_t &batched_w_ro_rows, const std::size_t &batched_w_ro_cols, const std::size_t &batched_w_ro_strides,
	float ***bundled_pre, const std::size_t &bundled_pre_rows, const std::size_t &bundled_pre_cols, const std::size_t &bundled_pre_strides,
	float **batched_post, const std::size_t &batched_post_rows, const std::size_t &batched_post_cols, const std::size_t &batched_post_strides,
	float ***bundled_desired, const std::size_t &bundled_desired_rows, const std::size_t &bundled_desired_cols, const std::size_t &bundled_desired_strides,
	const int *offsets, const int *durations, const std::size_t &repetitions, const std::size_t &total_duration,
	float *states_samples, const std::size_t &states_rows, const std::size_t &states_cols, const std::size_t &states_stride,
	const float *one, const float *zero
	)
{
	sgemm(
		handles[0],
		cublasOperation_t::CUBLAS_OP_N, cublasOperation_t::CUBLAS_OP_N,
		 batch_size,
		(const float **)batched_w_ffwd, batched_w_ffwd_cols, batched_w_ffwd_rows, batched_w_ffwd_strides,
		(const float **)batched_incoming, batched_incoming_cols, batched_incoming_rows, batched_incoming_strides,
		batched_u_ffwd, batched_u_ffwd_cols, batched_u_ffwd_rows, batched_u_ffwd_strides, one, zero
	);
	

	std::size_t mini_batch = 0;
	std::size_t bundle = 0;
	std::size_t ts = 0;

	for (std::size_t repetition = 0; repetition < repetitions; repetition++)
	{
		initialize_states<overwrite_states>(streams,  seed, batch_size,
			batched_p, batched_p_rows, batched_p_cols, batched_p_strides, initial_state_scale);
		initialize_states<overwrite_states>(streams, seed, batch_size,
			(float **)batched_x_res, batched_x_res_rows, batched_x_res_cols, batched_x_res_strides, initial_state_scale);

		for (std::size_t k = 0; k < durations[repetition]; k++, ts++)
		{
			update_reservoir
			(
				streams, handles, events,
				batch_size, mini_batch_size, mini_batch, bundle,
				offsets, ts, total_duration,
				(const float **)batched_w_rec, batched_w_rec_rows, batched_w_rec_cols, batched_w_rec_strides,
				batched_x_res, batched_x_res_rows, batched_x_res_cols, batched_x_res_strides,
				batched_u, batched_u_rows, batched_u_cols, batched_u_strides,
				(const float **)batched_u_ffwd, batched_u_ffwd_rows, batched_u_ffwd_cols, batched_u_ffwd_strides,
				batched_p, batched_p_rows, batched_p_cols, batched_p_strides,
				bundled_pre, bundled_pre_rows, bundled_pre_cols, bundled_pre_strides,
				leak_rate, one, zero);
			update_readout
				(
					streams, handles, events,
					batch_size, mini_batch_size, mini_batch, bundle,
					offsets, ts, total_duration,
					parameter,
					batched_x_res, batched_x_res_rows, batched_x_res_cols, batched_x_res_strides,
					batched_x_ro, batched_x_ro_rows, batched_x_ro_cols, batched_x_ro_strides,
					batched_expected, batched_expected_rows, batched_expected_cols, batched_expected_strides,
					bundled_pre, bundled_pre_rows, bundled_pre_cols, bundled_pre_strides,
					batched_post, batched_post_rows, batched_post_cols, batched_post_strides,
					bundled_desired, bundled_desired_rows, bundled_desired_cols, bundled_desired_strides,
					batched_w_ro, batched_w_ro_rows, batched_w_ro_cols, batched_w_ro_strides,one, zero

					);
			
		}
	}
	checkCudaErrors(cudaEventRecord(events[0], streams[1]));
	checkCudaErrors(cudaStreamWaitEvent(streams[0], events[0], 0));
}

/*__global__
static void location_hypothesis_kernel(
	const int batch_size,
	const int place_cells_number, 
	const int rows_offset, const int rows_span,
	const int cols_offset, const int cols_span,
	const float minus_inv_sigma2,
	const float   ** __restrict__ batched_firing_rate_map, const int firing_rate_stride,
	const float  ** __restrict__ batched_prediction, const int prediction_stride,
	float  *** __restrict__ batched_hypothesis_map, const int hypothesis_stride,
	float  ** __restrict__ batched_scale, const int scale_stride)
{
	for (int idx = blockIdx.z * blockDim.z + threadIdx.z; idx < batch_size * place_cells_number; idx += gridDim.z * blockDim.z)
	{
		auto place_cell = idx % place_cells_number;
		auto batch = idx / place_cells_number;
		float *scale = batched_scale[batch];
		const float *firing_rate_map = batched_firing_rate_map[place_cell];
		float *hypothesis_map = batched_hypothesis_map[batch][place_cell];
		const float4 p = make_float4(batched_prediction[batch][place_cell]);
		float4 sum4 = make_float4(0.0f);

		for (int roi_row = blockIdx.y * blockDim.y + threadIdx.y; roi_row < rows_span; roi_row += gridDim.y * blockDim.y)
		{
			auto row = roi_row + rows_offset;
			for (int roi_col = blockIdx.x * blockDim.x + threadIdx.x; roi_col < cols_span; roi_col += gridDim.x * blockDim.x)
			{
				auto col = roi_col + cols_offset;
				const float4 value = reinterpret_cast<float4 *>(const_cast<float *>(&firing_rate_map[row * firing_rate_stride]))[col] - p;
				const float4 response = expf(value * value * minus_inv_sigma2);
				sum4 += response;
				reinterpret_cast<float4 *>(&hypothesis_map[row * hypothesis_stride])[ col] = response;
			}
		}
		float sum = sum4.x + sum4.y + sum4.z + sum4.w;
		sum = warpReduceSum(sum);
		if ((threadIdx.x & 31) == 0)
		{
			atomicAdd(&scale[place_cell], sum * place_cells_number);
		}

	}
}*/

__global__
static void direction_kernel(const int batch_size,
	const float ** __restrict__ batched_previous_location,
	const float ** __restrict__ batched_current_location,
	float ** __restrict__ batched_direction)
{
	for (int batch = blockIdx.x * blockDim.x + threadIdx.x; batch < batch_size; batch += gridDim.x * blockDim.x)
	{
		float2 p = reinterpret_cast<float2 *>(const_cast<float *>(batched_previous_location[batch]))[0];
		float2 c = reinterpret_cast<float2 *>(const_cast<float *>(batched_current_location[batch]))[0];

		float2 d = c - p;


		auto d2 = dot(d, d);
		const float inv_norm = d2 > 0.0f ? rsqrtf(d2) : 0.0f;


		reinterpret_cast<float2 *>(batched_direction[batch])[0] = d * inv_norm;
	}
}

template <int coordinate>
__global__
static void range_centered_kernel(
	const int batch_size,
	const int cols,
	const float **__restrict__ batched_current_location,
	const float   * __restrict__ range,
	float **__restrict__ batched_range_centered
	 )
{
	for (int batch = blockIdx.y * blockDim.y + threadIdx.y; batch < batch_size; batch += gridDim.y * blockDim.y)
	{
		const float4 c = make_float4(batched_current_location[batch][coordinate]);
		float4 *range_centered = reinterpret_cast<float4 *>(batched_range_centered[batch]);
		for (int col = blockIdx.x * blockDim.x + threadIdx.x; col < cols; col += gridDim.x * blockDim.x)
		{
			range_centered[col] = reinterpret_cast<float4 *>(const_cast<float *>(range))[col] - c;
		}
	}
}

void compute_direction(
	const cudaStream_t *streams,
	const cublasHandle_t *handles,
	const std::size_t &batch_size,
	const float **batched_previous_location, const std::size_t &batched_previous_location_rows, const std::size_t &batched_previous_location_cols, const std::size_t &batched_previous_location_stride,
	const float **batched_current_location, const std::size_t &batched_current_location_rows, const std::size_t &batched_current_location_cols, const std::size_t &batched_current_location_stride,
	float **batched_direction, const std::size_t &batched_direction_rows, const std::size_t &batched_direction_cols, const std::size_t &batched_direction_stride)
{
	dim3 grid, block;
	block.x = 128;
	block.y = 1;
	block.z = 1;

	grid.x = (batch_size + block.x - 1) / block.x;
	direction_kernel << <grid, block, 0, *streams>> > (batch_size,
		batched_previous_location,
		batched_current_location,
		batched_direction);
	checkCudaErrors(cudaGetLastError());
}

void compute_reachable_locations(
	const cudaStream_t *streams,
	const cublasHandle_t *handles,
	const std::size_t &batch_size, const std::size_t &place_cells_number, 
	const std::size_t &rows_begin, const std::size_t &rows_end,
	const std::size_t &cols_begin, const std::size_t &cols_end,
	const float &radius, const float &cos_half_angle, const float &scale, const unsigned long &seed,
	const float *x_grid, const std::size_t &x_grid_rows, const std::size_t &x_grid_cols, const std::size_t &x_grid_stride,
	const float *y_grid, const std::size_t &y_grid_rows, const std::size_t &y_grid_cols, const std::size_t &y_grid_stride,
	const float **batched_current_location, const std::size_t &batched_current_location_rows, const std::size_t &batched_current_location_cols, const std::size_t &batched_current_location_stride,
	const float **batched_direction, const std::size_t &batched_direction_rows, const std::size_t &batched_direction_cols, const std::size_t &batched_direction_stride,
	float **batched_x_grid_centered, const std::size_t &batched_x_grid_centered_rows, const std::size_t &batched_x_grid_centered_cols, const std::size_t &batched_x_grid_centered_stride,
	float **batched_y_grid_centered, const std::size_t &batched_y_grid_centered_rows, const std::size_t &batched_y_grid_centered_cols, const std::size_t &batched_y_grid_centered_stride,
	float  **batched_location_probability, const std::size_t &batched_location_probability_rows, const std::size_t &batched_location_probability_cols, const std::size_t &batched_location_probability_strides)
{


	/*{
		dim3 grid, block;
		block.x = 128;
		block.y = 1;
		block.z = 1;

		grid.y = (batch_size + block.y - 1) / block.y;

		grid.x = (x_grid_cols / 4 + block.x - 1) / block.x;
		range_centered_kernel  <0> << <grid, block, 0, *streams>> > (batch_size, x_grid_cols / 4, batched_current_location, x_grid, batched_x_grid_centered);
		checkCudaErrors(cudaGetLastError());

		grid.x = (y_grid_cols / 4 + block.x - 1) / block.x;
		range_centered_kernel <1> << <grid, block, 0, *streams>> > (batch_size, y_grid_cols / 4, batched_current_location, y_grid, batched_y_grid_centered);
		checkCudaErrors(cudaGetLastError());
	}

	dim3 grid, block;
	block.x = BLOCK_X;
	block.y = BLOCK_Y;
	block.z = 1;
	auto cols_offset = round_down(cols_begin);
	auto cols_span = (round_up(cols_end, x_grid_cols) - cols_offset);
	auto rows_span = (rows_end - rows_begin);

	grid.x = (cols_span / 4 + block.x - 1) / block.x;
	grid.y = (rows_span + block.y - 1) / block.y;
	grid.z = (batch_size + block.z - 1) / block.z;

		inside_circle_sector_kernel << <grid, block, 0, *streams>> > (batch_size, 
			rows_begin, rows_span,
			cols_offset, cols_span / 4,
			batched_location_probability_strides, radius * radius, cos_half_angle, scale, seed,
			(const float **)batched_x_grid_centered, (const float **)batched_y_grid_centered,
			batched_current_location, batched_direction, batched_location_probability);
		checkCudaErrors(cudaGetLastError());
		*/
}

#define BLOCK_BATCH 16
#define BLOCK_PLACE_CELLS 256


__global__
static void assign_location_kernel(
	const int batch_size,
	const std::size_t *__restrict__ roi_row_begin, 
	const std::size_t *__restrict__ roi_col_begin,
	const int rows, const int cols,
	const int roi_stride,
	const float x_min, const float x_range, const float y_min, const float y_range,
	const int **__restrict__ batched_argmax,
	float **__restrict__ batched_predicted_location
)
{
	for (int batch = blockIdx.x * blockDim.x + threadIdx.x; batch < batch_size; batch += gridDim.x * blockDim.x)
	{
		auto predicted_location = batched_predicted_location[batch];
		auto idx = *batched_argmax[batch];
		float2 l;
		if (idx == 0)
		{
			l = make_float2(0.0f);
		}
		else
		{
			idx--;
			auto roi_col = idx % roi_stride;
			auto roi_row = idx / roi_stride;
			auto col = roi_col_begin[batch] + roi_col;
			auto row = roi_row_begin[batch] + roi_row;
			l.x = col_to_x(col, cols, x_min, x_range);
			l.y = row_to_y(row, rows, y_min, y_range);
		}

		*reinterpret_cast<float2 *>(predicted_location) = l;
	}
}
void compute_assign_most_probable_location(
	const cudaStream_t *streams, const cublasHandle_t *handles, const cudaEvent_t *events,
	const std::size_t &batch_size, const std::size_t &rows, const std::size_t &cols,
	const std::size_t *roi_row_begin, const std::size_t *roi_row_end, const std::size_t *roi_col_begin, const std::size_t *roi_col_end,
	const float &x_min, const float &x_range, const float &y_min, const float &y_range,
	const int **batched_argmax, const std::size_t &batched_location_probability_strides,
	float **batched_predicted_location)
{
	dim3 grid, block;

	block.x = warpSize * 4;
	grid.x = (batch_size + block.x - 1) / block.x;

	assign_location_kernel << <grid, block, 0, streams[0] >> > 
		(
			batch_size, roi_row_begin, roi_col_begin, rows, cols, batched_location_probability_strides, x_min, x_range, y_min, y_range, batched_argmax, batched_predicted_location
		);

	checkCudaErrors(cudaGetLastError());
}
	


void compute_select_most_probable_location(const cudaStream_t *streams, const cublasHandle_t *handles, const cudaEvent_t *events,
	const std::size_t &batch_size, const std::size_t &rows, const std::size_t &cols,
	const std::size_t *roi_row_begin, const std::size_t *roi_row_end, const std::size_t *roi_col_begin, const std::size_t *roi_col_end,

	const float *x_grid, const std::size_t &x_grid_rows, const std::size_t &x_grid_cols, const std::size_t &x_grid_stride,	
	const float *y_grid, const std::size_t &y_grid_rows, const std::size_t &y_grid_cols, const std::size_t &y_grid_stride,
	const float  **batched_location_probability, const std::size_t &batched_location_probability_rows, const std::size_t &batched_location_probability_cols, const std::size_t &batched_location_probability_strides,
	int **batched_argmax
)
{
	/*if (batched_temp_storage_bytes.empty())
	{
		batched_temp_storage_bytes.resize(batch_size);
		batched_temp_storage.resize(batch_size);
		batched_argmax.resize(batch_size);
		for (std::size_t batch = 0; batch < batch_size; batch++)
		{
			if (batched_temp_storage_bytes[batch] == 0)
			{
				checkCudaErrors(cudaMalloc(&batched_argmax[batch], sizeof(cub::KeyValuePair<int, float>)));
				cub::DeviceReduce::ArgMax(
					batched_temp_storage[batch], batched_temp_storage_bytes[batch],
					batched_location_probability[batch],
					(cub::KeyValuePair<int, float> *)batched_argmax[batch],
					batched_location_probability_rows * batched_location_probability_strides,
					streams[0]);
				checkCudaErrors(cudaMalloc(&batched_temp_storage[batch], batched_temp_storage_bytes[batch]));
			}
		}
	}
	for (std::size_t batch = 0; batch < batch_size; batch++)
	{
		cub::DeviceReduce::ArgMax(
			batched_temp_storage[batch], batched_temp_storage_bytes[batch],
			batched_location_probability[batch],
			(cub::KeyValuePair<int, float> *)batched_argmax[batch],
			batched_location_probability_rows * batched_location_probability_strides,
			streams[0]);
	}*/
	
	std::set<int> spawn;
	for (int batch = 0; batch < batch_size; batch++)
	{

		auto k = batch % TRN::GPU::Context::STREAM_NUMBER;

		auto x = batched_location_probability[batch];
		auto result = batched_argmax[batch];
		auto h = handles[k];
		auto n = batched_location_probability_rows *batched_location_probability_strides;
		checkCudaErrors(cublasIsamax(h, n, x, 1, result));
		spawn.insert(k);
	}

	spawn.erase(0);

	//checkCudaErrors(cudaDeviceSynchronize());

	for (auto k : spawn)
	{
		//checkCudaErrors(cudaStreamSynchronize(streams[k]));
		checkCudaErrors(cudaEventRecord(events[k], streams[k]));
		checkCudaErrors(cudaStreamWaitEvent(streams[0], events[k], 0));

	}

}





template  void update_model<true, true, Widrow_Hoff>(
	const std::size_t &batch_size,
	const std::size_t &mini_batch_size,
	unsigned long &seed,
	const cudaStream_t *streams,
	const cublasHandle_t *handles,
	const cudaEvent_t *events,
	const Widrow_Hoff &parameter,
	const std::size_t &stimulus_stride, const std::size_t &reservoir_stride, const std::size_t &prediction_stride,
	const std::size_t &stimulus_size, const std::size_t &reservoir_size, const std::size_t &prediction_size,
	const float &leak_rate, const float &initial_state_scale,
	float **batched_incoming, const std::size_t &batched_incoming_rows, const std::size_t &batched_incoming_cols, const std::size_t &batched_incoming_strides,
	float **batched_expected, const std::size_t &batched_expected_rows, const std::size_t &batched_expected_cols, const std::size_t &batched_expected_strides,
	float **batched_w_ffwd, const std::size_t &batched_w_ffwd_rows, const std::size_t &batched_w_ffwd_cols, const std::size_t &batched_w_ffwd_strides,
	float **batched_u_ffwd, const std::size_t &batched_u_ffwd_rows, const std::size_t &batched_u_ffwd_cols, const std::size_t &batched_u_ffwd_strides,
	float **batched_x_res, const std::size_t &batched_x_res_rows, const std::size_t &batched_x_res_cols, const std::size_t &batched_x_res_strides,
	float **batched_w_rec, const std::size_t &batched_w_rec_rows, const std::size_t &batched_w_rec_cols, const std::size_t &batched_w_rec_strides,
	float **batched_u, const std::size_t &batched_u_rows, const std::size_t &batched_u_cols, const std::size_t &batched_u_strides,
	float **batched_p, const std::size_t &batched_p_rows, const std::size_t &batched_p_cols, const std::size_t &batched_p_strides,
	float **batched_x_ro, const std::size_t &batched_x_ro_rows, const std::size_t &batched_x_ro_cols, const std::size_t &batched_x_ro_strides,
	float **batched_w_ro, const std::size_t &batched_w_ro_rows, const std::size_t &batched_w_ro_cols, const std::size_t &batched_w_ro_strides,
	float ***bundled_pre, const std::size_t &bundled_pre_rows, const std::size_t &bundled_pre_cols, const std::size_t &bundled_pre_strides,
	float **batched_post, const std::size_t &batched_post_rows, const std::size_t &batched_post_cols, const std::size_t &batched_post_strides,
	float ***bundled_post, const std::size_t &bundled_post_rows, const std::size_t &bundled_post_cols, const std::size_t &bundled_post_strides,
	const int *offsets, const int *durations, const std::size_t &repetitions, const std::size_t &total_duration,
	float *states_samples, const std::size_t &states_rows, const std::size_t &states_cols, const std::size_t &states_stride,
	const float *one, const float *zero
	);
template void update_model< true, false, Widrow_Hoff >(
	const std::size_t &batch_size,
	const std::size_t &mini_batch_size,
	unsigned long &seed,
	const cudaStream_t *streams,
	const cublasHandle_t *handles,
	const cudaEvent_t *events,
	const Widrow_Hoff &parameter,
	const std::size_t &stimulus_stride, const std::size_t &reservoir_stride, const std::size_t &prediction_stride,
	const std::size_t &stimulus_size, const std::size_t &reservoir_size, const std::size_t &prediction_size,
	const float &leak_rate, const float &initial_state_scale,
	float **batched_incoming, const std::size_t &batched_incoming_rows, const std::size_t &batched_incoming_cols, const std::size_t &batched_incoming_strides,
	float **batched_expected, const std::size_t &batched_expected_rows, const std::size_t &batched_expected_cols, const std::size_t &batched_expected_strides,
	float **batched_w_ffwd, const std::size_t &batched_w_ffwd_rows, const std::size_t &batched_w_ffwd_cols, const std::size_t &batched_w_ffwd_strides,
	float **batched_u_ffwd, const std::size_t &batched_u_ffwd_rows, const std::size_t &batched_u_ffwd_cols, const std::size_t &batched_u_ffwd_strides,
	float **batched_x_res, const std::size_t &batched_x_res_rows, const std::size_t &batched_x_res_cols, const std::size_t &batched_x_res_strides,
	float **batched_w_rec, const std::size_t &batched_w_rec_rows, const std::size_t &batched_w_rec_cols, const std::size_t &batched_w_rec_strides,
	float **batched_u, const std::size_t &batched_u_rows, const std::size_t &batched_u_cols, const std::size_t &batched_u_strides,
	float **batched_p, const std::size_t &batched_p_rows, const std::size_t &batched_p_cols, const std::size_t &batched_p_strides,
	float **batched_x_ro, const std::size_t &batched_x_ro_rows, const std::size_t &batched_x_ro_cols, const std::size_t &batched_x_ro_strides,
	float **batched_w_ro, const std::size_t &batched_w_ro_rows, const std::size_t &batched_w_ro_cols, const std::size_t &batched_w_ro_strides,
	float ***bundled_pre, const std::size_t &bundled_pre_rows, const std::size_t &bundled_pre_cols, const std::size_t &bundled_pre_strides,
	float **batched_post, const std::size_t &batched_post_rows, const std::size_t &batched_post_cols, const std::size_t &batched_post_strides,
	float ***bundled_post, const std::size_t &bundled_post_rows, const std::size_t &bundled_post_cols, const std::size_t &bundled_post_strides,
	const int *offsets, const int *durations, const std::size_t &repetitions, const std::size_t &total_duration,
	float *states_samples, const std::size_t &states_rows, const std::size_t &states_cols, const std::size_t &states_stride,
	const float *one, const float *zero
	);
template void update_model< false, true, Widrow_Hoff>(
	const std::size_t &batch_size,
	const std::size_t &mini_batch_size,
	unsigned long &seed,
	const cudaStream_t *streams,
	const cublasHandle_t *handles,
	const cudaEvent_t *events,
	const Widrow_Hoff &parameter,
	const std::size_t &stimulus_stride, const std::size_t &reservoir_stride, const std::size_t &prediction_stride,
	const std::size_t &stimulus_size, const std::size_t &reservoir_size, const std::size_t &prediction_size,
	const float &leak_rate, const float &initial_state_scale,
	float **batched_incoming, const std::size_t &batched_incoming_rows, const std::size_t &batched_incoming_cols, const std::size_t &batched_incoming_strides,
	float **batched_expected, const std::size_t &batched_expected_rows, const std::size_t &batched_expected_cols, const std::size_t &batched_expected_strides,
	float **batched_w_ffwd, const std::size_t &batched_w_ffwd_rows, const std::size_t &batched_w_ffwd_cols, const std::size_t &batched_w_ffwd_strides,
	float **batched_u_ffwd, const std::size_t &batched_u_ffwd_rows, const std::size_t &batched_u_ffwd_cols, const std::size_t &batched_u_ffwd_strides,
	float **batched_x_res, const std::size_t &batched_x_res_rows, const std::size_t &batched_x_res_cols, const std::size_t &batched_x_res_strides,
	float **batched_w_rec, const std::size_t &batched_w_rec_rows, const std::size_t &batched_w_rec_cols, const std::size_t &batched_w_rec_strides,
	float **batched_u, const std::size_t &batched_u_rows, const std::size_t &batched_u_cols, const std::size_t &batched_u_strides,
	float **batched_p, const std::size_t &batched_p_rows, const std::size_t &batched_p_cols, const std::size_t &batched_p_strides,
	float **batched_x_ro, const std::size_t &batched_x_ro_rows, const std::size_t &batched_x_ro_cols, const std::size_t &batched_x_ro_strides,
	float **batched_w_ro, const std::size_t &batched_w_ro_rows, const std::size_t &batched_w_ro_cols, const std::size_t &batched_w_ro_strides,
	float ***bundled_pre, const std::size_t &bundled_pre_rows, const std::size_t &bundled_pre_cols, const std::size_t &bundled_pre_strides,
	float **batched_post, const std::size_t &batched_post_rows, const std::size_t &batched_post_cols, const std::size_t &batched_post_strides,
	float ***bundled_post, const std::size_t &bundled_post_rows, const std::size_t &bundled_post_cols, const std::size_t &bundled_post_strides,
	const int *offsets, const int *durations, const std::size_t &repetitions, const std::size_t &total_duration,
	float *states_samples, const std::size_t &states_rows, const std::size_t &states_cols, const std::size_t &states_stride,
	const float *one, const float *zero
	);
template void update_model< false, false, Widrow_Hoff>(
	const std::size_t &batch_size,
	const std::size_t &mini_batch_size,
	unsigned long &seed,
	const cudaStream_t *streams,
	const cublasHandle_t *handles,
	const cudaEvent_t *events,
	const Widrow_Hoff &parameter,
	const std::size_t &stimulus_stride, const std::size_t &reservoir_stride, const std::size_t &prediction_stride,
	const std::size_t &stimulus_size, const std::size_t &reservoir_size, const std::size_t &prediction_size,
	const float &leak_rate, const float &initial_state_scale,
	float **batched_incoming, const std::size_t &batched_incoming_rows, const std::size_t &batched_incoming_cols, const std::size_t &batched_incoming_strides,
	float **batched_expected, const std::size_t &batched_expected_rows, const std::size_t &batched_expected_cols, const std::size_t &batched_expected_strides,
	float **batched_w_ffwd, const std::size_t &batched_w_ffwd_rows, const std::size_t &batched_w_ffwd_cols, const std::size_t &batched_w_ffwd_strides,
	float **batched_u_ffwd, const std::size_t &batched_u_ffwd_rows, const std::size_t &batched_u_ffwd_cols, const std::size_t &batched_u_ffwd_strides,
	float **batched_x_res, const std::size_t &batched_x_res_rows, const std::size_t &batched_x_res_cols, const std::size_t &batched_x_res_strides,
	float **batched_w_rec, const std::size_t &batched_w_rec_rows, const std::size_t &batched_w_rec_cols, const std::size_t &batched_w_rec_strides,
	float **batched_u, const std::size_t &batched_u_rows, const std::size_t &batched_u_cols, const std::size_t &batched_u_strides,
	float **batched_p, const std::size_t &batched_p_rows, const std::size_t &batched_p_cols, const std::size_t &batched_p_strides,
	float **batched_x_ro, const std::size_t &batched_x_ro_rows, const std::size_t &batched_x_ro_cols, const std::size_t &batched_x_ro_strides,
	float **batched_w_ro, const std::size_t &batched_w_ro_rows, const std::size_t &batched_w_ro_cols, const std::size_t &batched_w_ro_strides,
	float ***bundled_pre, const std::size_t &bundled_pre_rows, const std::size_t &bundled_pre_cols, const std::size_t &bundled_pre_strides,
	float **batched_post, const std::size_t &batched_post_rows, const std::size_t &batched_post_cols, const std::size_t &batched_post_strides,
	float ***bundled_post, const std::size_t &bundled_post_rows, const std::size_t &bundled_post_cols, const std::size_t &bundled_post_strides,
	const int *offsets, const int *durations, const std::size_t &repetitions, const std::size_t &total_duration,
	float *states_samples, const std::size_t &states_rows, const std::size_t &states_cols, const std::size_t &states_stride,
	const float *one, const float *zero
	);
template void update_model<true, true, Nothing>(
	const std::size_t &batch_size,
	const std::size_t &mini_batch_size,
	unsigned long &seed,
	const cudaStream_t *streams,
	const cublasHandle_t *handles,
	const cudaEvent_t *events,
	const Nothing &parameter,
	const std::size_t &stimulus_stride, const std::size_t &reservoir_stride, const std::size_t &prediction_stride,
	const std::size_t &stimulus_size, const std::size_t &reservoir_size, const std::size_t &prediction_size,
	const float &leak_rate, const float &initial_state_scale,
	float **batched_incoming, const std::size_t &batched_incoming_rows, const std::size_t &batched_incoming_cols, const std::size_t &batched_incoming_strides,
	float **batched_expected, const std::size_t &batched_expected_rows, const std::size_t &batched_expected_cols, const std::size_t &batched_expected_strides,
	float **batched_w_ffwd, const std::size_t &batched_w_ffwd_rows, const std::size_t &batched_w_ffwd_cols, const std::size_t &batched_w_ffwd_strides,
	float **batched_u_ffwd, const std::size_t &batched_u_ffwd_rows, const std::size_t &batched_u_ffwd_cols, const std::size_t &batched_u_ffwd_strides,
	float **batched_x_res, const std::size_t &batched_x_res_rows, const std::size_t &batched_x_res_cols, const std::size_t &batched_x_res_strides,
	float **batched_w_rec, const std::size_t &batched_w_rec_rows, const std::size_t &batched_w_rec_cols, const std::size_t &batched_w_rec_strides,
	float **batched_u, const std::size_t &batched_u_rows, const std::size_t &batched_u_cols, const std::size_t &batched_u_strides,
	float **batched_p, const std::size_t &batched_p_rows, const std::size_t &batched_p_cols, const std::size_t &batched_p_strides,
	float **batched_x_ro, const std::size_t &batched_x_ro_rows, const std::size_t &batched_x_ro_cols, const std::size_t &batched_x_ro_strides,
	float **batched_w_ro, const std::size_t &batched_w_ro_rows, const std::size_t &batched_w_ro_cols, const std::size_t &batched_w_ro_strides,
	float ***bundled_pre, const std::size_t &bundled_pre_rows, const std::size_t &bundled_pre_cols, const std::size_t &bundled_pre_strides,
	float **batched_post, const std::size_t &batched_post_rows, const std::size_t &batched_post_cols, const std::size_t &batched_post_strides,
	float ***bundled_post, const std::size_t &bundled_post_rows, const std::size_t &bundled_post_cols, const std::size_t &bundled_post_strides,
	const int *offsets, const int *durations, const std::size_t &repetitions, const std::size_t &total_duration,
	float *states_samples, const std::size_t &states_rows, const std::size_t &states_cols, const std::size_t &states_stride,
	const float *one, const float *zero
	);
template void update_model< true, false, Nothing >(
	const std::size_t &batch_size,
	const std::size_t &mini_batch_size,
	unsigned long &seed,
	const cudaStream_t *streams,
	const cublasHandle_t *handles,
	const cudaEvent_t *events,
	const Nothing &parameter,
	const std::size_t &stimulus_stride, const std::size_t &reservoir_stride, const std::size_t &prediction_stride,
	const std::size_t &stimulus_size, const std::size_t &reservoir_size, const std::size_t &prediction_size,
	const float &leak_rate, const float &initial_state_scale,
	float **batched_incoming, const std::size_t &batched_incoming_rows, const std::size_t &batched_incoming_cols, const std::size_t &batched_incoming_strides,
	float **batched_expected, const std::size_t &batched_expected_rows, const std::size_t &batched_expected_cols, const std::size_t &batched_expected_strides,
	float **batched_w_ffwd, const std::size_t &batched_w_ffwd_rows, const std::size_t &batched_w_ffwd_cols, const std::size_t &batched_w_ffwd_strides,
	float **batched_u_ffwd, const std::size_t &batched_u_ffwd_rows, const std::size_t &batched_u_ffwd_cols, const std::size_t &batched_u_ffwd_strides,
	float **batched_x_res, const std::size_t &batched_x_res_rows, const std::size_t &batched_x_res_cols, const std::size_t &batched_x_res_strides,
	float **batched_w_rec, const std::size_t &batched_w_rec_rows, const std::size_t &batched_w_rec_cols, const std::size_t &batched_w_rec_strides,
	float **batched_u, const std::size_t &batched_u_rows, const std::size_t &batched_u_cols, const std::size_t &batched_u_strides,
	float **batched_p, const std::size_t &batched_p_rows, const std::size_t &batched_p_cols, const std::size_t &batched_p_strides,
	float **batched_x_ro, const std::size_t &batched_x_ro_rows, const std::size_t &batched_x_ro_cols, const std::size_t &batched_x_ro_strides,
	float **batched_w_ro, const std::size_t &batched_w_ro_rows, const std::size_t &batched_w_ro_cols, const std::size_t &batched_w_ro_strides,
	float ***bundled_pre, const std::size_t &bundled_pre_rows, const std::size_t &bundled_pre_cols, const std::size_t &bundled_pre_strides,
	float **batched_post, const std::size_t &batched_post_rows, const std::size_t &batched_post_cols, const std::size_t &batched_post_strides,
	float ***bundled_post, const std::size_t &bundled_post_rows, const std::size_t &bundled_post_cols, const std::size_t &bundled_post_strides,
	const int *offsets, const int *durations, const std::size_t &repetitions, const std::size_t &total_duration,
	float *states_samples, const std::size_t &states_rows, const std::size_t &states_cols, const std::size_t &states_stride,
	const float *one, const float *zero
	);
template void update_model< false, true, Nothing>(
	const std::size_t &batch_size,
	const std::size_t &mini_batch_size,
	unsigned long &seed,
	const cudaStream_t *streams,
	const cublasHandle_t *handles,
	const cudaEvent_t *events,
	const Nothing &parameter,
	const std::size_t &stimulus_stride, const std::size_t &reservoir_stride, const std::size_t &prediction_stride,
	const std::size_t &stimulus_size, const std::size_t &reservoir_size, const std::size_t &prediction_size,
	const float &leak_rate, const float &initial_state_scale,
	float **batched_incoming, const std::size_t &batched_incoming_rows, const std::size_t &batched_incoming_cols, const std::size_t &batched_incoming_strides,
	float **batched_expected, const std::size_t &batched_expected_rows, const std::size_t &batched_expected_cols, const std::size_t &batched_expected_strides,
	float **batched_w_ffwd, const std::size_t &batched_w_ffwd_rows, const std::size_t &batched_w_ffwd_cols, const std::size_t &batched_w_ffwd_strides,
	float **batched_u_ffwd, const std::size_t &batched_u_ffwd_rows, const std::size_t &batched_u_ffwd_cols, const std::size_t &batched_u_ffwd_strides,
	float **batched_x_res, const std::size_t &batched_x_res_rows, const std::size_t &batched_x_res_cols, const std::size_t &batched_x_res_strides,
	float **batched_w_rec, const std::size_t &batched_w_rec_rows, const std::size_t &batched_w_rec_cols, const std::size_t &batched_w_rec_strides,
	float **batched_u, const std::size_t &batched_u_rows, const std::size_t &batched_u_cols, const std::size_t &batched_u_strides,
	float **batched_p, const std::size_t &batched_p_rows, const std::size_t &batched_p_cols, const std::size_t &batched_p_strides,
	float **batched_x_ro, const std::size_t &batched_x_ro_rows, const std::size_t &batched_x_ro_cols, const std::size_t &batched_x_ro_strides,
	float **batched_w_ro, const std::size_t &batched_w_ro_rows, const std::size_t &batched_w_ro_cols, const std::size_t &batched_w_ro_strides,
	float ***bundled_pre, const std::size_t &bundled_pre_rows, const std::size_t &bundled_pre_cols, const std::size_t &bundled_pre_strides,
	float **batched_post, const std::size_t &batched_post_rows, const std::size_t &batched_post_cols, const std::size_t &batched_post_strides,
	float ***bundled_post, const std::size_t &bundled_post_rows, const std::size_t &bundled_post_cols, const std::size_t &bundled_post_strides,
	const int *offsets, const int *durations, const std::size_t &repetitions, const std::size_t &total_duration,
	float *states_samples, const std::size_t &states_rows, const std::size_t &states_cols, const std::size_t &states_stride,
	const float *one, const float *zero
	);
template void update_model< false, false, Nothing>(
	const std::size_t &batch_size,
	const std::size_t &mini_batch_size,
	unsigned long &seed,
	const cudaStream_t *streams,
	const cublasHandle_t *handles,
	const cudaEvent_t *events,
	const Nothing &parameter,
	const std::size_t &stimulus_stride, const std::size_t &reservoir_stride, const std::size_t &prediction_stride,
	const std::size_t &stimulus_size, const std::size_t &reservoir_size, const std::size_t &prediction_size,
	const float &leak_rate, const float &initial_state_scale,
	float **batched_incoming, const std::size_t &batched_incoming_rows, const std::size_t &batched_incoming_cols, const std::size_t &batched_incoming_strides,
	float **batched_expected, const std::size_t &batched_expected_rows, const std::size_t &batched_expected_cols, const std::size_t &batched_expected_strides,
	float **batched_w_ffwd, const std::size_t &batched_w_ffwd_rows, const std::size_t &batched_w_ffwd_cols, const std::size_t &batched_w_ffwd_strides,
	float **batched_u_ffwd, const std::size_t &batched_u_ffwd_rows, const std::size_t &batched_u_ffwd_cols, const std::size_t &batched_u_ffwd_strides,
	float **batched_x_res, const std::size_t &batched_x_res_rows, const std::size_t &batched_x_res_cols, const std::size_t &batched_x_res_strides,
	float **batched_w_rec, const std::size_t &batched_w_rec_rows, const std::size_t &batched_w_rec_cols, const std::size_t &batched_w_rec_strides, 
	float **batched_u, const std::size_t &batched_u_rows, const std::size_t &batched_u_cols, const std::size_t &batched_u_strides,
	float **batched_p, const std::size_t &batched_p_rows, const std::size_t &batched_p_cols, const std::size_t &batched_p_strides,
	float **batched_x_ro, const std::size_t &batched_x_ro_rows, const std::size_t &batched_x_ro_cols, const std::size_t &batched_x_ro_strides,
	float **batched_w_ro, const std::size_t &batched_w_ro_rows, const std::size_t &batched_w_ro_cols, const std::size_t &batched_w_ro_strides,
	float ***bundled_pre, const std::size_t &bundled_pre_rows, const std::size_t &bundled_pre_cols, const std::size_t &bundled_pre_strides,
	float **batched_post, const std::size_t &batched_post_rows, const std::size_t &batched_post_cols, const std::size_t &batched_post_strides,
	float ***bundled_post, const std::size_t &bundled_post_rows, const std::size_t &bundled_post_cols, const std::size_t &bundled_post_strides,
	const int *offsets, const int *durations, const std::size_t &repetitions, const std::size_t &total_duration,
	float *states_samples, const std::size_t &states_rows, const std::size_t &states_cols, const std::size_t &states_stride,
	const float *one, const float *zero
	);

