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
	if (driver->get_memory()->is_column_major())
	{
		handle->width = handle->rows;
		handle->height= handle->cols;

	}
	else
	{
		handle->width = handle->cols;
		handle->height = handle->rows;
	}
	if (rows > 0 && cols > 0)
	{
		handle->ownership = true;

		implementor->get_memory()->allocate((void **)&handle->elements, handle->stride, sizeof(float), handle->width, handle->height);
		if (blank)
		{
			implementor->get_memory()->blank(handle->elements, sizeof(float), handle->width, handle->height, handle->stride);
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
	if (driver->get_memory()->is_column_major())
	{
		handle->width = handle->rows;
		handle->height = handle->cols;
	}
	else
	{
		handle->width = handle->cols;
		handle->height = handle->rows;
	}

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

	if (driver->get_memory()->is_column_major())
	{
		handle->width = handle->rows;
		handle->height = handle->cols;
	}
	else
	{
		handle->width = handle->cols;
		handle->height = handle->rows;
	}

	handle->ownership = true;

	implementor->get_memory()->allocate((void **)&handle->elements, handle->stride, sizeof(float), handle->width, handle->height);
	implementor->get_memory()->blank(handle->elements, sizeof(float), handle->width, handle->height, handle->stride);

	//checkCudaErrors(cudaMemcpy2DAsync(dst, dst_stride * sizeof(float), src, src_stride * sizeof(float), sizeof(float), size, cudaMemcpyKind::cudaMemcpyDeviceToDevice, stream));
	std::vector<float> to_upload = to_device_storage(elements, rows, cols);

	implementor->get_memory()->upload(to_upload.data(), handle->elements, sizeof(float) , handle->width, handle->height, handle->stride);
	//implementor->synchronize();
}

TRN::Core::Matrix::Matrix(const std::shared_ptr<TRN::Backend::Driver> &driver, const std::shared_ptr<TRN::Core::Matrix> &matrix, const std::size_t &row, const std::size_t &col, const std::size_t &rows, const std::size_t &cols) :
	TRN::Helper::Bridge<TRN::Backend::Driver>(driver),
	handle(std::make_unique<TRN::Core::Matrix::Handle>())
{
	handle->rows = (rows == TRN::Core::Matrix::DEFAULT_ROWS ? matrix->get_rows() - row: rows);
	handle->cols = (cols == TRN::Core::Matrix::DEFAULT_COLS ? matrix->get_cols() - col : cols);


	if (implementor->get_memory()->is_column_major())
	{
		handle->width = handle->rows;
		handle->height = handle->cols;
	}
	else
	{
		handle->width = handle->cols;
		handle->height = handle->rows;
	}

	if (handle->rows > 0 && handle->cols > 0)
	{
		if (implementor->get_memory()->is_column_major())
		{
			handle->elements = &matrix->get_elements()[col * matrix->get_stride() + row];
		}
		else
		{
			handle->elements = &matrix->get_elements()[row * matrix->get_stride() + col];
		}
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
	}

	handle.reset();
}

static const int column_major[] = { 0,1 };
static const int row_major[] = { 1,0 };
static const bool ascending[] = { false,false };

std::vector<float> TRN::Core::Matrix::to_device_storage(const std::vector<float> &host_storage, const std::size_t &rows, const std::size_t &cols)
{
	std::vector<float> device_storage(host_storage.size());
	boost::multi_array_ref<float, 2> dst(
		(float *)device_storage.data(),
		boost::extents[rows][cols],
		boost::general_storage_order<2>(implementor->get_memory()->is_column_major() ? column_major : row_major, ascending)
	);

	boost::multi_array_ref<float, 2> src(
		(float *)host_storage.data(),
		boost::extents[rows][cols],
		boost::general_storage_order<2>(row_major, ascending));


	std::copy(src.begin(), src.end(), dst.begin());
	return device_storage;
}
std::vector<float> TRN::Core::Matrix::to_host_storage(const std::vector<float> &device_storage, const std::size_t &rows, const std::size_t &cols)
{
	std::vector<float> host_storage(device_storage.size());
	boost::multi_array_ref<float, 2> src(
		(float *)device_storage.data(),
		boost::extents[rows][cols],
		boost::general_storage_order<2>(implementor->get_memory()->is_column_major() ? column_major : row_major, ascending)
	);

	boost::multi_array_ref<float, 2> dst(host_storage.data(),
		boost::extents[rows][cols], boost::general_storage_order<2>(row_major, ascending));

	
	std::copy(src.begin(), src.end(), dst.begin());
	return host_storage;
}


void TRN::Core::Matrix::to(TRN::Core::Matrix &matrix) 
{
	if (matrix.get_cols() != handle->cols)
		throw std::logic_error("Source and destination matrices must have the same width");
	if (matrix.get_rows() != handle->rows)
		throw std::logic_error("Source and destination matrices must have the same height");
	
	implementor->get_memory()->copy(handle->elements, matrix.get_elements(), sizeof(float), handle->width, handle->height, handle->stride, matrix.get_stride());
}

void TRN::Core::Matrix::to(std::vector<float> &elements, std::size_t &rows, std::size_t &cols)
{
	rows = handle->rows;
	cols = handle->cols;
	
	std::vector<float> to_download(rows * cols);
	implementor->get_memory()->download(to_download.data(), handle->elements, sizeof(float), handle->width, handle->height, handle->stride, false);
	elements = to_host_storage(to_download, rows, cols);
}

void TRN::Core::Matrix::from(const TRN::Core::Matrix &matrix) 
{
	if (matrix.get_cols() != handle->cols)
		throw std::logic_error("Source and destination matrices must have the same width");
	if (matrix.get_rows() != handle->rows)
		throw std::logic_error("Source and destination matrices must have the same height");

	implementor->get_memory()->copy(matrix.get_elements(), handle->elements, sizeof(float), handle->width, handle->height,  matrix.get_stride(), handle->stride);
}


void TRN::Core::Matrix::from(const std::vector<float> &elements, const std::size_t &rows, const std::size_t &cols)
{
	if (rows != handle->rows)
		throw std::runtime_error("invalid row number");
	if (cols != handle->cols)
		throw std::runtime_error("invalid col number");


	std::vector<float> to_upload = to_device_storage(elements, rows, cols);
	implementor->get_memory()->upload(to_upload.data(), handle->elements, sizeof(float), handle->width, handle->height, handle->stride, false);
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
const std::size_t &TRN::Core::Matrix::get_width() const
{
	return handle->width;
}
const std::size_t &TRN::Core::Matrix::get_height() const
{
	return handle->height;
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
	/*std::cout << "row " << row;
	std::cout << ", col " << col;
	std::cout << ", rows " << rows;
	std::cout << ", cols " << cols << std::endl;*/
	return std::make_shared<TRN::Core::Matrix>(driver, matrix, row, col, rows, cols);
}
