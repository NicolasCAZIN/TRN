#pragma once

#include "Batch.h"

class TRN::Core::Batch::Handle
{
public :
	bool uploaded;
	std::size_t size;

	float **dev_elements;
	std::vector<bool> filled;
	std::vector<std::shared_ptr<TRN::Core::Matrix>> matrices;
	std::vector<std::size_t> host_strides;
	std::vector<std::size_t> host_rows;
	std::vector<std::size_t> host_cols;
	std::vector<float *> host_elements;
};
