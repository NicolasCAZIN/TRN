#pragma once

#include "Bundle.h"

class TRN::Core::Bundle::Handle
{
public:
	bool uploaded;
	std::size_t size;

	float ***dev_elements;
	std::vector<bool> filled;
	std::vector<std::shared_ptr<TRN::Core::Batch>> batches;
	std::vector<std::size_t *> host_strides;
	std::vector<std::size_t *> host_rows;
	std::vector<std::size_t *> host_cols;
	std::vector<std::size_t> host_matrices;
	std::vector<float **> host_elements;
};
