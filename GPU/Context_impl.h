#pragma once

#include "Context.h"

class TRN::GPU::Context::Handle
{
public:
	int device;
	std::size_t stride_alignement;
	std::vector<cudaStream_t> streams;
	std::vector<cublasHandle_t> handles;
	std::vector<cudaEvent_t> events;
	std::string name;
	std::size_t max_block_size;
};

//template class GPU_EXPORT std::unique_ptr<TRN::GPU::Context::Handle>;