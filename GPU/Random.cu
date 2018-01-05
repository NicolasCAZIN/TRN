#include "stdafx.h"
#include <cuda.h>
#include <curand_kernel.h>
/*#include <cublasXt.h>*/
//#include <cublas_api.h>

// helper functions and utilities to work with CUDA
#include "Driver.cuh"

#include <helper_math.h>

__global__
static void batched_random_uniform_kernel(
	const unsigned long seed,
	const float offset,
	const float scale,
	const int batch_size, const int rows, const int cols,
	float ** __restrict__ x, const int x_stride, bool blank_diagonal
)
{
	for (int batch = blockIdx.z * blockDim.z + threadIdx.z; batch < batch_size; batch += gridDim.z * blockDim.z)
	{
		float *X = x[batch];
		for (int row = blockIdx.y * blockDim.y + threadIdx.y; row < rows; row += gridDim.y * blockDim.y)
		{
			for (int col = blockIdx.x * blockDim.x + threadIdx.x; col < cols >> 2; col += gridDim.x * blockDim.x)
			{
				curandStatePhilox4_32_10_t s;
				curand_init(seed + col * rows + row + batch * rows * cols, 0, 0, &s);
				float4 r = curand_uniform4(&s) * scale + offset;
				if (blank_diagonal && (row >> 0x2) == col)
				{
					((float *)&r.x)[row & 0x3] = 0.0f;
				}
				reinterpret_cast<float4 *>(&X[row * x_stride])[col] = r;
			}
		}
	}
}

__global__
static void batched_random_uniform_sparse_kernel(
	const unsigned long seed,
	const float offset,
	const float scale,
	const float sparsity,
	const int batch_size, const int rows, const int cols,
	float ** __restrict__ x, const int x_stride, bool blank_diagonal
)
{
	for (int batch = blockIdx.z * blockDim.z + threadIdx.z; batch < batch_size; batch += gridDim.z * blockDim.z)
	{
		float *X = x[batch];
		for (int row = blockIdx.y * blockDim.y + threadIdx.y; row < rows; row += gridDim.y * blockDim.y)
		{
			for (int col = blockIdx.x * blockDim.x + threadIdx.x; col < cols >> 2; col += gridDim.x * blockDim.x)
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
				if (blank_diagonal && (row >> 0x2) == col)
				{
					((float *)&value.x)[row & 0x3] = 0.0f;
				}
				reinterpret_cast<float4 *>(&X[row * x_stride])[col] = value;
			}
		}
	}
}
void random_uniform(const cudaStream_t &stream,
	const unsigned long &seed,
	const float &a, const float &b, const float &sparsity,
	const std::size_t &batch_size, const std::size_t &rows, const std::size_t &cols,
	float **x, const std::size_t &x_stride, const bool &blank_diagonal)
{
	auto scale = b - a;
	auto offset = a;

	dim3 grid, block;
	block.x = 32;
	block.y = 32;
	block.z = 1;
	grid.x = (cols / 4 + block.x - 1) / block.x;
	grid.y = (rows + block.y - 1) / block.y;
	grid.z = (batch_size + block.z - 1) / block.z;


	if (sparsity > 0.0f)
	{
		batched_random_uniform_sparse_kernel << < grid, block, 0, stream >> > (
			seed,
			offset, scale, sparsity,
			batch_size, rows, cols, x, x_stride, blank_diagonal);
	}
	else
	{
		batched_random_uniform_kernel << < grid, block, 0, stream >> > (
			seed,
			offset, scale,
			batch_size, rows, cols, x, x_stride, blank_diagonal);
	}

	checkCudaErrors(cudaGetLastError());
}

__global__
static void batched_random_gaussian_kernel(
	const unsigned long seed,
	const float mu,
	const float sigma,
	const int batch_size, const int rows, const int cols,
	float ** __restrict__ x, const int x_stride, bool blank_diagonal
)
{
	for (int batch = blockIdx.z * blockDim.z + threadIdx.z; batch < batch_size; batch += gridDim.z * blockDim.z)
	{
		float *X = x[batch];
		for (int row = blockIdx.y * blockDim.y + threadIdx.y; row < rows; row += gridDim.y * blockDim.y)
		{
			for (int col = blockIdx.x * blockDim.x + threadIdx.x; col < cols >> 2; col += gridDim.x * blockDim.x)
			{
				curandStatePhilox4_32_10_t s;
				curand_init(seed + col * rows + row + batch * rows * cols, 0, 0, &s);
				float4 r = curand_normal4(&s) * sigma + mu;
				if (blank_diagonal && (row >> 0x2) == col)
				{
					((float *)&r.x)[row & 0x3] = 0.0f;
				}
				reinterpret_cast<float4 *>(&X[row * x_stride])[col] = r;
			}
		}
	}
}__global__
static void batched_random_gaussian_sparse_kernel(
	const unsigned long seed,
	const float mu,
	const float sigma,
	const float sparsity,
	const int batch_size, const int rows, const int cols,
	float ** __restrict__ x, const int x_stride, bool blank_diagonal
)
{
	for (int batch = blockIdx.z * blockDim.z + threadIdx.z; batch < batch_size; batch += gridDim.z * blockDim.z)
	{
		float *X = x[batch];
		for (int row = blockIdx.y * blockDim.y + threadIdx.y; row < rows; row += gridDim.y * blockDim.y)
		{
			for (int col = blockIdx.x * blockDim.x + threadIdx.x; col < cols >> 2; col += gridDim.x * blockDim.x)
			{
				curandStatePhilox4_32_10_t s;
				// seed a random number generator
				curand_init(seed + col * rows + row + batch * rows * cols, 0, 0, &s);

				auto dice = curand_uniform4(&s);
				auto value = curand_normal4(&s) * sigma + mu;

				value.x = dice.x < sparsity ? 0.0f : value.x;
				value.y = dice.y < sparsity ? 0.0f : value.y;
				value.z = dice.z < sparsity ? 0.0f : value.z;
				value.w = dice.w < sparsity ? 0.0f : value.w;
				if (blank_diagonal && (row >> 0x2) == col)
				{
					((float *)&value.x)[row & 0x3] = 0.0f;
				}
				reinterpret_cast<float4 *>(&X[row * x_stride])[col] = value;
			}
		}
	}
}
void random_gaussian(const cudaStream_t &stream,
	const unsigned long &seed,
	const float &mu, const float &sigma, const float &sparsity,
	const std::size_t &batch_size, const std::size_t &rows, const std::size_t &cols,
	float **x, const std::size_t &x_stride, const bool &blank_diagonal)
{
	dim3 grid, block;

	block.x = 32;
	block.y = 32;
	block.z = 1;
	grid.x = (cols / 4 + block.x - 1) / block.x;
	grid.y = (rows + block.y - 1) / block.y;
	grid.z = (batch_size + block.z - 1) / block.z;

	if (sparsity > 0.0f)
	{
		batched_random_gaussian_sparse_kernel << < grid, block, 0, stream >> > (
			seed,
			mu, sigma, sparsity,
			batch_size, rows, cols, x, x_stride, blank_diagonal);
	}
	else
	{
		batched_random_gaussian_kernel << < grid, block, 0, stream >> > (
			seed,
			mu, sigma,
			batch_size, rows, cols, x, x_stride, blank_diagonal);
	}
	checkCudaErrors(cudaGetLastError());
}


