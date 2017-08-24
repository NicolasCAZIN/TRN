/**
* Copyright 2016 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/

#include <vector>
#include <iostream>
#include <ctime>
#include <cublas.h>
#include <helper_cuda.h>

static const std::size_t repetitions = 1000;
static const std::size_t reservoir_size = 1024;
static const std::size_t batch_size = 100;
static const std::size_t stimulus_size = 256;

static const std::size_t flops = repetitions * reservoir_size * stimulus_size * 2 * batch_size;

static float cublas_batched_sgemm()
{
	float **X = NULL;
	float **Y = NULL;
	float **W = NULL;
	std::vector<float *> _X(batch_size);
	std::vector<float *> _Y(batch_size);
	std::vector<float *> _W(batch_size);
	std::size_t X_rows = 1;
	std::size_t X_cols = reservoir_size + stimulus_size;
	std::size_t X_pitch;

	std::size_t Y_rows = 1;
	std::size_t Y_cols = reservoir_size;
	std::size_t Y_pitch;

	std::size_t W_rows = reservoir_size;
	std::size_t W_cols = reservoir_size + stimulus_size;
	std::size_t W_pitch;

	checkCudaErrors(cudaMalloc(&X, batch_size * sizeof(float *)));
	checkCudaErrors(cudaMalloc(&Y, batch_size * sizeof(float *)));
	checkCudaErrors(cudaMalloc(&W, batch_size * sizeof(float *)));

	for (std::size_t batch = 0; batch < batch_size; batch++)
	{
		checkCudaErrors(cudaMallocPitch((void **)&_X[batch], &X_pitch, X_rows, X_cols));
		checkCudaErrors(cudaMallocPitch((void **)&_Y[batch], &Y_pitch, Y_rows, Y_cols));
		checkCudaErrors(cudaMallocPitch((void **)&_W[batch], &W_pitch, W_rows, W_cols));
	}

	checkCudaErrors(cudaMemcpy(X, _X.data(), sizeof(float *) * batch_size, cudaMemcpyKind::cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(Y, _Y.data(), sizeof(float *) * batch_size, cudaMemcpyKind::cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(W, _W.data(), sizeof(float *) * batch_size, cudaMemcpyKind::cudaMemcpyHostToDevice));

	auto t0 = std::clock();
	for (int repetition = 0; repetition < repetitions; repetition++)
	{

	}

	auto t1 = std::clock();
	float seconds = (t1 - t0) / CLOCKS_PER_SEC;

	checkCudaErrors(cudaFree(X));
	checkCudaErrors(cudaFree(Y));
	checkCudaErrors(cudaFree(W));


	return (flops / seconds) * 1.0f-9;
}

int main(int argc, char *argv[])
{

	std::cout << "cublas_batched_sgemm : " << cublas_batched_sgemm() << "Glops/s" << std::endl;
}