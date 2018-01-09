#include "stdafx.h"
#include "Batch_impl.h"

TRN::Core::Batch::Batch(const std::shared_ptr<TRN::Backend::Driver> &driver, const std::size_t &size) :
	TRN::Helper::Bridge<TRN::Backend::Driver>(driver),
	handle(std::make_unique<Handle>())
{

	handle->uploaded = false;
	handle->size = size;
	handle->filled.resize(size);
	handle->matrices.resize(size);
	handle->host_strides.resize(size);
	handle->host_rows.resize(size);
	handle->host_cols.resize(size);
	handle->host_elements.resize(size);
	implementor->get_memory()->allocate((void **)&handle->dev_elements, sizeof(float *), size);

	std::fill(handle->filled.begin(), handle->filled.end(), false);
}


TRN::Core::Batch::~Batch()
{

	implementor->get_memory()->deallocate(handle->dev_elements);
	handle.reset();
}
	
void TRN::Core::Batch::from(const TRN::Core::Batch &batch)
{
	for (std::size_t k = 0; k < batch.handle->matrices.size(); k++)
		handle->matrices[k]->from(*batch.handle->matrices[k]);
}

void TRN::Core::Batch::to(std::vector<float> &elements, std::size_t &matrices, std::vector<std::size_t> &rows, std::vector<std::size_t> &cols)
{
	matrices = handle->size;
	rows = handle->host_rows;
	cols = handle->host_cols;

	for (std::size_t k = 0; k < handle->size; k++)
	{
		std::vector<float> temp_values;
		std::size_t temp_rows, temp_cols;

		handle->matrices[k]->to(temp_values, temp_rows, temp_cols);

		elements.insert(elements.end(), temp_values.begin(), temp_values.end());
	}		
}

const std::size_t TRN::Core::Batch::get_size()
{
	return handle->size;
}
float ** TRN::Core::Batch::get_elements(const bool &host)
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

const std::size_t *TRN::Core::Batch::get_rows()
{
	if (!handle->uploaded)
		upload();
	return handle->host_rows.data();
}


const std::size_t *TRN::Core::Batch::get_cols()
{
	if (!handle->uploaded)
		upload();
	return handle->host_cols.data();
}


const std::size_t *TRN::Core::Batch::get_strides()
{
	if (!handle->uploaded)
		upload();
	return handle->host_strides.data();
}


const std::size_t TRN::Core::Batch::get_rows(const std::size_t &index)
{
	if (!handle->uploaded)
		upload();
	return handle->host_rows[index];
}


const std::size_t TRN::Core::Batch::get_cols(const std::size_t &index)
{
	if (!handle->uploaded)
		upload();
	return handle->host_cols[index];
}
const std::size_t TRN::Core::Batch::get_strides(const std::size_t &index)
{
	if (!handle->uploaded)
		upload();
	return handle->host_strides[index];
}

std::shared_ptr<TRN::Core::Matrix> TRN::Core::Batch::get_matrices(const std::size_t &index)
{
	if (!handle->uploaded)
		upload();
	if (!handle->matrices[index])
	{
		handle->matrices[index] = TRN::Core::Matrix::create(implementor, handle->host_elements[index], handle->host_rows[index], handle->host_cols[index], handle->host_strides[index]);
	}
	return handle->matrices[index];
}

void TRN::Core::Batch::update(const std::size_t &index, const std::shared_ptr<TRN::Core::Matrix> &matrix)
{
	if (index >= handle->size)
		throw std::runtime_error("Batch was allocated with a size of " + std::to_string(handle->size));
	handle->matrices[index] = matrix;
	update(index, matrix->get_elements(), matrix->get_rows(), matrix->get_cols(), matrix->get_stride());
}
void TRN::Core::Batch::update(const std::size_t &index, float *elements, const std::size_t &rows, const std::size_t &cols, const std::size_t &stride)
{
	if (index >= handle->size)
		throw std::runtime_error("Batch was allocated with a size of " + std::to_string(handle->size));
	handle->host_elements[index] = elements;
	handle->host_rows[index] = rows;
	handle->host_cols[index] = cols;
	handle->host_strides[index] = stride;

	handle->filled[index] = true;
	handle->uploaded = false;
}

void TRN::Core::Batch::upload()
{
	if (std::any_of(handle->filled.begin(), handle->filled.end(), [](const bool &filled) {return filled == false; }))
		throw std::runtime_error("All indexes of the batch have not been filled");
	implementor->get_memory()->upload(handle->host_elements.data(), handle->dev_elements, sizeof(float *), handle->size);

	handle->uploaded = true;
}

std::shared_ptr<TRN::Core::Batch> TRN::Core::Batch::create(const std::shared_ptr<TRN::Backend::Driver> &driver, const std::size_t &size)
{
	return std::make_shared<TRN::Core::Batch>(driver, size);
}