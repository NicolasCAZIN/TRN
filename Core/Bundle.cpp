#include "stdafx.h"
#include "Bundle_impl.h"

TRN::Core::Bundle::Bundle(const std::shared_ptr<TRN::Backend::Driver> &driver, const std::size_t &size) :
	TRN::Helper::Bridge<TRN::Backend::Driver>(driver),
	handle(std::make_unique<Handle>())
{
	handle->uploaded = false;
	handle->size = size;
	handle->filled.resize(size);
	handle->batches.resize(size);
	handle->host_strides.resize(size);
	handle->host_rows.resize(size);
	handle->host_cols.resize(size);
	handle->host_elements.resize(size);

	implementor->get_memory()->allocate((void **)&handle->dev_elements, sizeof(float **), size);

	std::fill(handle->filled.begin(), handle->filled.end(), false);
}
TRN::Core::Bundle::~Bundle()
{
	handle.reset();
}


void TRN::Core::Bundle::to(std::vector<float> &elements, std::size_t &batches, std::vector<std::size_t> &matrices, std::vector<std::vector<std::size_t>> &rows, std::vector<std::vector<std::size_t>> &cols)
{
	batches = handle->size;
	matrices.resize(batches);
	rows.resize(batches);
	cols.resize(batches);

	for (std::size_t k = 0; k < handle->size; k++)
	{
		std::vector<float> temp_values;
		std::size_t temp_matrices;
		std::vector<std::size_t> temp_rows, temp_cols;

		handle->batches[k]->to(temp_values, temp_matrices, temp_rows, temp_cols);
		elements.insert(elements.end(), temp_values.begin(), temp_values.end());
		rows[k] = temp_rows;
		cols[k] = temp_cols;
		matrices[k] = temp_matrices;
	}
}


float *** TRN::Core::Bundle::get_elements(const bool &host)
{
	if (host)
	{
		return handle->host_elements.data();
	}
	else
	{
		if (!handle->uploaded)
		{
			upload();
		}
		return handle->dev_elements;
	}
}

const std::size_t **TRN::Core::Bundle::get_rows()
{
	if (!handle->uploaded)
		upload();
	return (const std::size_t **)handle->host_rows.data();
}


const std::size_t **TRN::Core::Bundle::get_cols()
{
	if (!handle->uploaded)
		upload();
	return (const std::size_t **)handle->host_cols.data();
}


const std::size_t **TRN::Core::Bundle::get_strides()
{
	if (!handle->uploaded)
		upload();
	return (const std::size_t **)handle->host_strides.data();
}


const std::size_t *TRN::Core::Bundle::get_rows(const std::size_t &index)
{
	if (!handle->uploaded)
		upload();
	return (const std::size_t *)handle->host_rows[index];
}


const std::size_t *TRN::Core::Bundle::get_cols(const std::size_t &index)
{
	if (!handle->uploaded)
		upload();
	return (const std::size_t *)handle->host_cols[index];
}
const std::size_t *TRN::Core::Bundle::get_strides(const std::size_t &index)
{
	if (!handle->uploaded)
		upload();
	return (const std::size_t *)handle->host_strides[index];
}

std::shared_ptr<TRN::Core::Batch> TRN::Core::Bundle::get_batches(const std::size_t &index)
{
	if (!handle->uploaded)
		upload();
	if (!handle->batches[index])
	{
		throw std::runtime_error("batch #" + std::to_string(index) + " is not present");
	}
	return handle->batches[index];
}

void TRN::Core::Bundle::update(const std::size_t &index, const std::shared_ptr<TRN::Core::Batch> &batch)
{
	if (index >= handle->size)
		throw std::runtime_error("Batch was allocated with a size of " + std::to_string(handle->size));
	handle->batches[index] = batch;
	handle->host_elements[index] = batch->get_elements();
	handle->host_rows[index] = (std::size_t *)batch->get_rows();
	handle->host_cols[index] = (std::size_t *)batch->get_cols();
	handle->host_strides[index] = (std::size_t *)batch->get_strides();

	handle->filled[index] = true;
	handle->uploaded = false;
}

void TRN::Core::Bundle::upload()
{
	if (std::any_of(handle->filled.begin(), handle->filled.end(), [](const bool &filled) {return filled == false; }))
		throw std::runtime_error("All indexes of the batch have not been filled");
	implementor->get_memory()->upload(handle->host_elements.data(), handle->dev_elements, sizeof(float **), handle->size);

	handle->uploaded = true;
}

std::shared_ptr<TRN::Core::Bundle> TRN::Core::Bundle::create(const std::shared_ptr<TRN::Backend::Driver> &driver, const std::size_t &size)
{
	return std::make_shared<TRN::Core::Bundle>(driver, size);
}