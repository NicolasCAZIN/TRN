/*#include "cuda_runtime.h"
#include <cublas.h>
#include "device_launch_parameters.h"
#include <helper_cuda.h>
#include <helper_math.h>
#include <functional>
#include <chrono>
#include <iostream>
#include <vector>
#include <memory>
#include <cub/cub.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_reduce.cuh>

#include <armadillo>


#define GRID_X 32
#define GRID_Y 32

#define REPEAT 100UL
#define batch_size 10UL
#define ROWS 1024UL
#define COLS 1024UL
#define PLACE_CELLS 256UL
#define ALGORITHM cub::BlockReduceAlgorithm::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY
struct Matrix
{

	Matrix(const std::size_t &cols) : Matrix(1, cols)
	{

	}
	Matrix(const std::size_t &rows, const std::size_t &cols) : rows(rows), cols(cols)
	{
		checkCudaErrors(cudaMallocPitch((void **)&data, &pitch, cols * sizeof(float), rows));
		stride = pitch / sizeof(float);

	}
	~Matrix()
	{
			checkCudaErrors(cudaFree(data));
	}
	float *data;
	size_t rows;
	size_t cols;
	size_t stride;
	 size_t pitch;
};

struct Batch
{
	std::vector<std::shared_ptr<Matrix>> matrices;

	float **data = NULL;
	std::size_t stride;
	Batch(const std::size_t &batch_size, const std::size_t &cols) : Batch(batch_size, 1, cols)
	{
	}
	Batch(const std::size_t &batch_size, const std::size_t &rows, const std::size_t &cols) :
		matrices(batch_size)
	{
		std::vector<float *> pointers(batch_size);
		for (std::size_t batch = 0; batch < batch_size; batch++)
		{
			matrices[batch] = std::make_shared<Matrix>(rows, cols);
			pointers[batch] = matrices[batch]->data;

		}
		stride = matrices[0]->stride;
		checkCudaErrors(cudaMalloc(&data, batch_size * sizeof(float *)));
		cudaMemcpy(data, pointers.data(), sizeof(float *) *batch_size, cudaMemcpyKind::cudaMemcpyHostToDevice);
	}
	~Batch()
	{
		checkCudaErrors(cudaFree(data));
	}
};

#define GB(size) ((size) >> 30)

const std::size_t gb_size = 0x1 << 30;
const std::size_t map_size = batch_size * ROWS * COLS * PLACE_CELLS * sizeof(float);
const float gb_map_size = map_size / gb_size;

static const std::size_t addition_flops = 1;
static const std::size_t multiplication_flops = 1;
static const std::size_t substraction_flops = 1;
static const std::size_t exponential_flops = 50;


static const size_t firing_rate_map_length = ROWS * COLS * PLACE_CELLS;
static const size_t naive_flops = batch_size * REPEAT * firing_rate_map_length * (substraction_flops + 2 * multiplication_flops +  exponential_flops);
static const size_t premult_flops = batch_size * REPEAT * firing_rate_map_length * (3 * multiplication_flops  + 2 * addition_flops + exponential_flops);
static const size_t model_flops = batch_size * REPEAT * firing_rate_map_length * (3 * multiplication_flops + 2 * addition_flops + exponential_flops);

static const float naive_gflops = naive_flops * 1e-9f;
static const float premult_gflops = premult_flops * 1e-9f;
static const float model_gflops = model_flops * 1e-9f;
// float4 / unrolled // dynamic parallelism / stream / batched


static float __constant__ y_min;
static float __constant__ y_max;
static float __constant__ y_range;
static float __constant__ x_min;
static float __constant__ x_max;
static float __constant__ x_range;
static float __constant__ x_center[PLACE_CELLS];
static float __constant__ y_center[PLACE_CELLS];
static float __constant__ inv_width2[PLACE_CELLS];



void compute_coefficients(cublasHandle_t &handle, const Matrix &firing_rate_map, const float &sigma)
{
	static const std::size_t taylor_series_order = 8;
	static const std::size_t coefficient_orderr = taylor_series_order * 2 + 1;
	std::vector<float> SA(coefficient_orderr);
	std::vector<float> s(coefficient_orderr);
	for (std::size_t k = 0; k < coefficient_orderr; k++)
	{

	}


	C(pc, 1) = (1 / 1)*(SA(pc, 0 + 1) / sigma(0 + 1)) - (1 / 1)*(SA(pc, 2 + 1) / sigma(2 + 1)) + (1 / 2)*(SA(pc, 4 + 1) / sigma(4 + 1)) - (1 / 6)*(SA(pc, 6 + 1) / sigma(6 + 1)) + (1 / 24)*(SA(pc, 8 + 1) / sigma(8 + 1)) - (1 / 120)*(SA(pc, 10 + 1) / sigma(10 + 1)) + (1 / 720)*(SA(pc, 12 + 1) / sigma(12 + 1)) - (1 / 5040)*(SA(pc, 14 + 1) / sigma(14 + 1));
	C(pc, 2) = (2 / 1)*(SA(pc, 1 + 1) / sigma(2 + 1)) - (2 / 1)*(SA(pc, 3 + 1) / sigma(4 + 1)) + (1 / 1)*(SA(pc, 5 + 1) / sigma(6 + 1)) - (1 / 3)*(SA(pc, 7 + 1) / sigma(8 + 1)) + (1 / 12)*(SA(pc, 9 + 1) / sigma(10 + 1)) - (1 / 60)*(SA(pc, 11 + 1) / sigma(12 + 1)) + (1 / 360)*(SA(pc, 13 + 1) / sigma(14 + 1));
	C(pc, 3) = -(1 / 1)*(SA(pc, 0 + 1) / sigma(2 + 1)) + (3 / 1)*(SA(pc, 2 + 1) / sigma(4 + 1)) - (5 / 2)*(SA(pc, 4 + 1) / sigma(6 + 1)) + (7 / 6)*(SA(pc, 6 + 1) / sigma(8 + 1)) - (3 / 8)*(SA(pc, 8 + 1) / sigma(10 + 1)) + (11 / 120)*(SA(pc, 10 + 1) / sigma(12 + 1)) - (13 / 720)*(SA(pc, 12 + 1) / sigma(14 + 1));
	C(pc, 4) = -(2 / 1)*(SA(pc, 1 + 1) / sigma(4 + 1)) + (10 / 3)*(SA(pc, 3 + 1) / sigma(6 + 1)) - (7 / 3)*(SA(pc, 5 + 1) / sigma(8 + 1)) + (1 / 1)*(SA(pc, 7 + 1) / sigma(10 + 1)) - (11 / 36)*(SA(pc, 9 + 1) / sigma(12 + 1)) + (13 / 180)*(SA(pc, 11 + 1) / sigma(14 + 1));
	C(pc, 5) = (1 / 2)*(SA(pc, 0 + 1) / sigma(4 + 1)) - (5 / 2)*(SA(pc, 2 + 1) / sigma(6 + 1)) + (35 / 12)*(SA(pc, 4 + 1) / sigma(8 + 1)) - (7 / 4)*(SA(pc, 6 + 1) / sigma(10 + 1)) + (11 / 16)*(SA(pc, 8 + 1) / sigma(12 + 1)) - (143 / 720)*(SA(pc, 10 + 1) / sigma(14 + 1));
	C(pc, 6) = (1 / 1)*(SA(pc, 1 + 1) / sigma(6 + 1)) - (7 / 3)*(SA(pc, 3 + 1) / sigma(8 + 1)) + (21 / 10)*(SA(pc, 5 + 1) / sigma(10 + 1)) - (11 / 10)*(SA(pc, 7 + 1) / sigma(12 + 1)) + (143 / 360)*(SA(pc, 9 + 1) / sigma(14 + 1));
	C(pc, 7) = -(1 / 6)*(SA(pc, 0 + 1) / sigma(6 + 1)) + (7 / 6)*(SA(pc, 2 + 1) / sigma(8 + 1)) - (7 / 4)*(SA(pc, 4 + 1) / sigma(10 + 1)) + (77 / 60)*(SA(pc, 6 + 1) / sigma(12 + 1)) - (143 / 240)*(SA(pc, 8 + 1) / sigma(14 + 1));
	C(pc, 8) = -(1 / 3)*(SA(pc, 1 + 1) / sigma(8 + 1)) + (1 / 1)*(SA(pc, 3 + 1) / sigma(10 + 1)) - (11 / 10)*(SA(pc, 5 + 1) / sigma(12 + 1)) + (143 / 210)*(SA(pc, 7 + 1) / sigma(14 + 1));
	C(pc, 9) = (1 / 24)*(SA(pc, 0 + 1) / sigma(8 + 1)) - (3 / 8)*(SA(pc, 2 + 1) / sigma(10 + 1)) + (11 / 16)*(SA(pc, 4 + 1) / sigma(12 + 1)) - (143 / 240)*(SA(pc, 6 + 1) / sigma(14 + 1));
	C(pc, 10) = (1 / 12)*(SA(pc, 1 + 1) / sigma(10 + 1)) - (11 / 36)*(SA(pc, 3 + 1) / sigma(12 + 1)) + (143 / 360)*(SA(pc, 5 + 1) / sigma(14 + 1));
	C(pc, 11) = -(1 / 120)*(SA(pc, 0 + 1) / sigma(10 + 1)) + (11 / 120)*(SA(pc, 2 + 1) / sigma(12 + 1)) - (143 / 720)*(SA(pc, 4 + 1) / sigma(14 + 1));
	C(pc, 12) = -(1 / 60)*(SA(pc, 1 + 1) / sigma(12 + 1)) + (13 / 180)*(SA(pc, 3 + 1) / sigma(14 + 1));
	C(pc, 13) = (1 / 720)*(SA(pc, 0 + 1) / sigma(12 + 1)) - (13 / 720)*(SA(pc, 2 + 1) / sigma(14 + 1));
	C(pc, 14) = (1 / 360)*(SA(pc, 1 + 1) / sigma(14 + 1));
	C(pc, 15) = -(1 / 5040)*(SA(pc, 0 + 1) / sigma(14 + 1));

}

typedef cub::KeyValuePair<int, float> Hypothesis;

__global__
void naive_fr_model(

	const float  ** __restrict__ activation, const int activation_stride,
	const float _sigma2_inv,
	float  ** __restrict__ coordinates)
{
	for (int batch = blockIdx.z * blockDim.z + threadIdx.z; batch < batch_size; batch += gridDim.z * blockDim.z)
	{
		for (int row = blockIdx.y * blockDim.y + threadIdx.y; row < ROWS; row += gridDim.y * blockDim.y)
		{
			for (int col = blockIdx.x * blockDim.x + threadIdx.x; col < COLS; col += gridDim.x * blockDim.x)
			{
				float __shared__ x[GRID_X];
				float __shared__ y[GRID_Y];

				Hypothesis  __shared__ h[GRID_Y * GRID_X];
				float __shared__ a[PLACE_CELLS];

				if (blockIdx.z == 0)
				{
					if (blockIdx.y == 0)
					{
						x[threadIdx.x] = (col / (COLS - 1)) * x_range + x_min;
					}
					if (blockIdx.x == 0)
					{
						y[threadIdx.y] = (row / (ROWS - 1)) * y_range + y_min;
					}
				}
				if (blockIdx.y == 0 && blockIdx.x == 0)
				{
#pragma unroll PLACE_CELLS
					for (int place_cell = 0; place_cell < PLACE_CELLS; place_cell++)
					{
						a[place_cell] = activation[batch][place_cell];
					}
				}

				__syncthreads();
				float sum = 0.0f;

#pragma unroll PLACE_CELLS
				for (int place_cell = 0; place_cell < PLACE_CELLS; place_cell++)
				{
					float dx = x[threadIdx.x] - x_center[place_cell];
					float dy = y[threadIdx.y] - y_center[place_cell];
					float f = expf(inv_width2[place_cell] * (fmaf(dx, dx, dy * dy)));
					float value = a[place_cell] - f;
					sum += expf(_sigma2_inv * value * value);
				}
				if (blockIdx.z == 0)
				{
					auto idx = threadIdx.y * GRID_X + threadIdx.x;
					h[idx].key = idx;
					h[idx].value = sum;
				}
				__syncthreads();
				// Specialize BlockReduce type for our thread block
			
			}
		}
	}
}

static float sf_model()
{
	auto t0 = std::chrono::high_resolution_clock::now();
	cudaDeviceSynchronize();
	auto seconds = std::chrono::duration_cast<std::chrono::duration<float>>(std::chrono::high_resolution_clock::now() - t0).count();

	return model_gflops / seconds;
}


static void bench(const std::function<float()> &functor, const std::string &label)
{
	auto throughput = functor();
	std::cout << label << " troughput : " << throughput << " Gflops/s" << std::endl;
}

int main()
{

	checkCudaErrors(cudaSetDevice(0));
	bench(sf_model, "sf_model");
	//bench(sf_naive_fr_contiguous_pc_coalesced, "sf_naive_fr_contiguous_pc_coalesced");
	//bench(sf_naive_fr_contiguous_pc_coalesced_batched, "sf_naive_fr_contiguous_pc_coalesced_batched");
	//bench(sf_naive_fr_contiguous_pc_coalesced_batched_float4, "sf_naive_fr_contiguous_pc_coalesced_batched_float4");
	//bench(sf_naive_fr_contiguous_pc_coalesced_batched_float8, "sf_naive_fr_contiguous_pc_coalesced_batched_float8");


	checkCudaErrors(cudaDeviceReset());
	return 0;
}
*/