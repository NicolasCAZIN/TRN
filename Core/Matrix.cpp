#include "stdafx.h"
#include "Matrix_impl.h"
#include "Helper/Bridge.h"

const bool TRN::Core::Matrix::DEFAULT_BLANK = false;
const std::size_t TRN::Core::Matrix::DEFAULT_ROWS = 0;
const std::size_t TRN::Core::Matrix::DEFAULT_COLS = 0;

TRN::Core::Matrix::Matrix(const std::shared_ptr<TRN::Backend::Driver> &driver, const std::size_t &rows, const std::size_t &cols, const bool &blank) :
	TRN::Helper::Bridge<TRN::Backend::Driver>(driver),
	handle(std::make_unique<TRN::Core::Matrix::Handle>())
{
	handle->rows = rows;
	handle->cols = cols;

	if (rows > 0 && cols > 0)
	{
		handle->ownership = true;

		implementor->get_memory()->allocate((void **)&handle->elements, handle->stride, sizeof(float), handle->cols, handle->rows);
		if (blank)
		{
			implementor->get_memory()->blank(handle->elements, sizeof(float), handle->cols, handle->rows, handle->stride);
		}
	}
	else
	{
		handle->ownership = false;
		handle->elements = NULL;
	}
}

TRN::Core::Matrix::Matrix(const std::shared_ptr<TRN::Backend::Driver> &driver, const float *dev_elements, const std::size_t &rows, const std::size_t &cols, const std::size_t &stride) :
	TRN::Helper::Bridge<TRN::Backend::Driver>(driver),
	handle(std::make_unique<TRN::Core::Matrix::Handle>())
{
	if (rows == 0 || cols == 0)
		throw std::runtime_error("Matrix dimensions should not be zero");
	handle->rows = rows;
	handle->cols = cols;
	handle->stride = stride;

	handle->elements = (float *)dev_elements;
	handle->ownership = false;
}

TRN::Core::Matrix::Matrix(const std::shared_ptr<TRN::Backend::Driver> &driver, const std::vector<float> &elements, const std::size_t &rows, const std::size_t &cols) :
	TRN::Helper::Bridge<TRN::Backend::Driver>(driver),
	handle(std::make_unique<TRN::Core::Matrix::Handle>())
{
	if (rows == 0 || cols == 0)
		throw std::runtime_error("Matrix dimensions should not be zero");
	handle->rows = rows;
	handle->cols = cols;
	assert(std::any_of(elements.begin(), elements.end(), isnan<float>) == false);
	handle->ownership = true;

	implementor->get_memory()->allocate((void **)&handle->elements, handle->stride, sizeof(float), handle->cols, handle->rows);
	implementor->get_memory()->blank(handle->elements, sizeof(float), handle->cols, handle->rows, handle->stride);
	implementor->get_memory()->upload(elements.data(), handle->elements, sizeof(float) , handle->cols, handle->rows, handle->stride);
}

TRN::Core::Matrix::Matrix(const std::shared_ptr<TRN::Backend::Driver> &driver, const std::shared_ptr<TRN::Core::Matrix> &matrix, const std::size_t &row, const std::size_t &col, const std::size_t &rows, const std::size_t &cols) :
	TRN::Helper::Bridge<TRN::Backend::Driver>(driver),
	handle(std::make_unique<TRN::Core::Matrix::Handle>())
{
	handle->rows = (rows == TRN::Core::Matrix::DEFAULT_ROWS ? matrix->get_rows() - row: rows);
	handle->cols = (cols == TRN::Core::Matrix::DEFAULT_COLS ? matrix->get_cols() - col : cols);


	if (handle->rows > 0 && handle->cols > 0)
	{
		handle->elements = &matrix->get_elements()[row * matrix->get_stride() + col];
		handle->stride = matrix->get_stride();
	}
	else
	{
		handle->elements = NULL;
		handle->stride = 0;
	}

	handle->ownership = false;
}

TRN::Core::Matrix::~Matrix()
{
	if (handle->ownership && handle->rows > 0 &&  handle->cols > 0)
	{
		implementor->get_memory()->deallocate(handle->elements);
		handle->elements = NULL;
	}

	handle.reset();
}


void TRN::Core::Matrix::to(TRN::Core::Matrix &matrix) 
{
	if (matrix.get_cols() != handle->cols)
		throw std::logic_error("Source and destination matrices must have the same width");
	if (matrix.get_rows() != handle->rows)
		throw std::logic_error("Source and destination matrices must have the same height");
	
	implementor->get_memory()->copy(handle->elements, matrix.get_elements(), sizeof(float), handle->cols, handle->rows, handle->stride, matrix.get_stride());
}

void TRN::Core::Matrix::to(std::vector<float> &elements, std::size_t &rows, std::size_t &cols)
{
	rows = handle->rows;
	cols = handle->cols;
	
	elements.resize(rows * cols);

	implementor->get_memory()->download(elements.data(), handle->elements, sizeof(float), handle->cols, handle->rows, handle->stride, false);
	assert(std::any_of(elements.begin(), elements.end(), isnan<float>) == false);
}




void TRN::Core::Matrix::from(const TRN::Core::Matrix &matrix) const
{
	if (matrix.get_cols() != handle->cols)
		throw std::logic_error("Source and destination matrices must have the same width");
	if (matrix.get_rows() != handle->rows)
		throw std::logic_error("Source and destination matrices must have the same height");

	implementor->get_memory()->copy(matrix.get_elements(), handle->elements, sizeof(float), handle->cols, handle->rows,  matrix.get_stride(), handle->stride);
}


void TRN::Core::Matrix::from(const std::vector<float> &elements, const std::size_t &rows, const std::size_t &cols) const
{
	if (rows != handle->rows)
		throw std::runtime_error("invalid row number");
	if (cols != handle->cols)
		throw std::runtime_error("invalid col number");
	assert(std::any_of(elements.begin(), elements.end(), isnan<float>) == false);
	implementor->get_memory()->upload(elements.data(), handle->elements, sizeof(float), handle->cols, handle->rows, handle->stride, false);
}


float *TRN::Core::Matrix::get_elements() const
{
	return handle->elements;
}
const std::size_t &TRN::Core::Matrix::get_rows() const
{
	return handle->rows;
}
const std::size_t &TRN::Core::Matrix::get_cols() const
{
	return handle->cols;
}

const std::size_t &TRN::Core::Matrix::get_stride() const
{
	return handle->stride;
}

std::shared_ptr<TRN::Core::Matrix> TRN::Core::Matrix::create(const std::shared_ptr<TRN::Backend::Driver> &driver, const std::size_t &rows, const std::size_t &cols, const bool &blank)
{
	return std::make_shared<TRN::Core::Matrix>(driver, rows, cols, blank);
}

std::shared_ptr<TRN::Core::Matrix> TRN::Core::Matrix::create(const std::shared_ptr<TRN::Backend::Driver> &driver, const float *dev_elements, const std::size_t &rows, const std::size_t &cols, const std::size_t &stride)
{
	return std::make_shared<TRN::Core::Matrix>(driver, dev_elements, rows, cols, stride);
}

std::shared_ptr<TRN::Core::Matrix> TRN::Core::Matrix::create(const std::shared_ptr<TRN::Backend::Driver> &driver, const std::vector<float> &elements, const std::size_t &rows, const std::size_t &cols)
{
	return std::make_shared<TRN::Core::Matrix>(driver, elements, rows, cols);
}
//#include <iostream>
std::shared_ptr<TRN::Core::Matrix> TRN::Core::Matrix::create(const std::shared_ptr<TRN::Backend::Driver> &driver,const std::shared_ptr<TRN::Core::Matrix> &matrix, const std::size_t &row, const std::size_t &col, const std::size_t &rows, const std::size_t &cols)
{
	/*INFORMATION "row " << row;
	INFORMATION ", col " << col;
	INFORMATION ", rows " << rows;
	INFORMATION ", cols " << cols ;*/
	return std::make_shared<TRN::Core::Matrix>(driver, matrix, row, col, rows, cols);
}
