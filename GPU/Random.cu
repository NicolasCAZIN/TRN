#include "stdafx.h"
#include <cuda.h>
#include <curand_kernel.h>
/*#include <cublasXt.h>*/
//#include <cublas_api.h>

// helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>

#include <helper_functions.h>
#include <helper_cuda.h>
#include <helper_math.h>

template <const int factor = 1>
__global__
static void batched_random_uniform_kernel(
	const unsigned long seed,
	const float offset,
	const float scale,
	const int batch_size, const int rows, const int cols,
	float ** __restrict__ x, const int x_stride
)
{
	for (int batch = blockIdx.z * blockDim.z + threadIdx.z; batch < batch_size; batch += gridDim.z * blockDim.z)
	{
		float *X = x[batch];
		for (int col = blockIdx.y * blockDim.y + threadIdx.y; col < cols; col += gridDim.y * blockDim.y)
		{
#pragma unroll factor
			for (int row = blockIdx.x * blockDim.x + threadIdx.x; row < rows; row += gridDim.x * blockDim.x)
			{
				curandStatePhilox4_32_10_t s;
				curand_init(seed + col * rows + row + batch * rows * cols, 0, 0, &s);
				reinterpret_cast<float4 *>(&X[col * x_stride])[row] = curand_uniform4(&s) * scale + offset;
			}
		}
	}
}

template <const int factor = 1>
__global__
static void batched_random_uniform_sparse_kernel(
	const unsigned long seed,
	const float offset,
	const float scale,
	const float sparsity,
	const int batch_size, const int rows, const int cols,
	float ** __restrict__ x, const int x_stride
)
{
	for (int batch = blockIdx.z * blockDim.z + threadIdx.z; batch < batch_size; batch += gridDim.z * blockDim.z)
	{
		float *X = x[batch];
		for (int col = blockIdx.y * blockDim.y + threadIdx.y; col < cols; col += gridDim.y * blockDim.y)
		{
#pragma unroll factor
			for (int row = blockIdx.x * blockDim.x + threadIdx.x; row < rows; row += gridDim.x * blockDim.x)
			{
				curandStatePhilox4_32_10_t s;
				// seed a random number generator
				curand_init(seed + col * rows + row + batch * rows * cols, 0, 0, &s);

				auto dice = curand_uniform4(&s);
				auto value = curand_uniform4(&s) * scale + offset;

				value.x = dice.x < sparsity ? 0.0f : value.x;
				value.y = dice.y < sparsity ? 0.0f : value.y;
				value.z = dice.z < sparsity ? 0.0f : value.z;
				value.w = dice.w < sparsity ? 0.0f : value.w;
				reinterpret_cast<float4 *>(&X[col * x_stride])[row] = value;
			}
		}
	}
}
void random_uniform(const cudaStream_t &stream,
	const unsigned long &seed,
	const float &a, const float &b, const float &sparsity,
	const std::size_t &batch_size, const std::size_t &rows, const std::size_t &cols,
	float **x, const std::size_t &x_stride)
{
	auto scale = b - a;
	auto offset = a;

	auto Rows = rows / 4;
	dim3 grid, block;
	block.x = 32;
	grid.x = (Rows + block.x - 1) / block.x;

	block.y = 32;
	grid.y = (cols + block.y - 1) / block.y;

	block.z = 1;
	grid.z = (batch_size + block.z - 1) / block.z;


	if (Rows % 8 == 0)
	{
		if (sparsity > 0.0f)
		{
			batched_random_uniform_sparse_kernel <8> << < grid, block, 0, stream >> > (
				seed,
				offset, scale, sparsity,
				batch_size, Rows, cols, x, x_stride);
		}
		else
		{
			batched_random_uniform_kernel <8> << < grid, block, 0, stream >> > (
				seed,
				offset, scale,
				batch_size, Rows, cols, x, x_stride);
		}
	}
	else if (Rows % 4 == 0)
	{
		if (sparsity > 0.0f)
		{
			batched_random_uniform_sparse_kernel <4> << < grid, block, 0, stream >> > (
				seed,
				offset, scale, sparsity,
				batch_size, Rows, cols, x, x_stride);
		}
		else
		{
			batched_random_uniform_kernel <4> << < grid, block, 0, stream >> > (
				seed,
				offset, scale,
				batch_size, Rows, cols, x, x_stride);
		}
	}
	else if (Rows % 2 == 0)
	{
		if (sparsity > 0.0f)
		{
			batched_random_uniform_sparse_kernel <2> << < grid, block, 0, stream >> > (
				seed,
				offset, scale, sparsity,
				batch_size, Rows, cols, x, x_stride);
		}
		else
		{
			batched_random_uniform_kernel <2> << < grid, block, 0, stream >> > (
				seed,
				offset, scale,
				batch_size, Rows, cols, x, x_stride);
		}
	}
	else
	{
		if (sparsity > 0.0f)
		{
			batched_random_uniform_sparse_kernel <1> << < grid, block, 0, stream >> > (
				seed,
				offset, scale, sparsity,
				batch_size, Rows, cols, x, x_stride);
		}
		else
		{
			batched_random_uniform_kernel <1> << < grid, block, 0, stream >> > (
					seed,
					offset, scale,
					batch_size, Rows, cols, x, x_stride);
		}
	}

	checkCudaErrors(cudaGetLastError());
}

template <const int factor = 1>
__global__
static void batched_random_gaussian_kernel(
	const unsigned long seed,
	const float mu,
	const float sigma,
	const int batch_size, const int rows, const int cols,
	float ** __restrict__ x, const int x_stride
)
{
	for (int batch = blockIdx.z * blockDim.z + threadIdx.z; batch < batch_size; batch += gridDim.z * blockDim.z)
	{
		float *X = x[batch];
		for (int col = blockIdx.y * blockDim.y + threadIdx.y; col < cols; col += gridDim.y * blockDim.y)
		{
#pragma unroll factor
			for (int row = blockIdx.x * blockDim.x + threadIdx.x; row < rows; row += gridDim.x * blockDim.x)
			{
				curandStatePhilox4_32_10_t s;
				curand_init(seed + col * rows + row + batch * rows * cols, 0, 0, &s);
				reinterpret_cast<float4 *>(&X[col * x_stride])[row] = curand_normal4(&s) * sigma + mu;
			}
		}
	}
}
void random_gaussian(const cudaStream_t &stream,
	const unsigned long &seed,
	const float &mu, const float &sigma,
	const std::size_t &batch_size, const std::size_t &rows, const std::size_t &cols,
	float **x, const std::size_t &x_stride)
{
	dim3 grid, block;

	auto Rows = rows / 4;
	block.x = 32;
	grid.x = (Rows + block.x - 1) / block.x;

	block.y = 32;
	grid.y = (cols + block.y - 1) / block.y;

	block.z = 1;
	grid.z = (batch_size + block.z - 1) / block.z;

	if (Rows % 8 == 0)
	{
		batched_random_gaussian_kernel <8> << < grid, block, 0, stream >> > (
			seed,
			mu, sigma,
			batch_size, Rows, cols, x, x_stride);
	}
	else if (Rows % 4 == 0)
	{
		batched_random_gaussian_kernel <4> << < grid, block, 0, stream >> > (
			seed,
			mu, sigma,
			batch_size, Rows, cols, x, x_stride);

	}
	else if (Rows % 2 == 0)
	{
		batched_random_gaussian_kernel <2> << < grid, block, 0, stream >> > (
			seed,
			mu, sigma,
			batch_size, Rows, cols, x, x_stride);
	}
	else
	{
		batched_random_gaussian_kernel <1> << < grid, block, 0, stream >> > (
			seed,
			mu, sigma,
			batch_size, Rows, cols, x, x_stride);
	}


	checkCudaErrors(cudaGetLastError());
}


