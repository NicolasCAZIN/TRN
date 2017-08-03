#pragma once

#include "Context.h"

class TRN::GPU::Context::Handle
{
public:
	int device;
	std::size_t stride_alignement;
	cudaStream_t stream;
	cublasHandle_t handle;
	curandGenerator_t generator;
	std::string name;
	std::size_t max_block_size;
};

//template class GPU_EXPORT std::unique_ptr<TRN::GPU::Context::Handle>;