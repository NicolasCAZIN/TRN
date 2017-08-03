#pragma once

#include "core_global.h"
#include "Helper/Bridge.h"
#include "Backend/Driver.h"

namespace TRN
{
	namespace Core
	{
		
		class CORE_EXPORT Matrix :
			public TRN::Helper::Bridge <TRN::Backend::Driver>
		{
		private :
			static const bool DEFAULT_BLANK;
			static const std::size_t DEFAULT_ROWS;
			static const std::size_t DEFAULT_COLS;
	

		private :
			class Handle;
			mutable std::unique_ptr<Handle> handle;
		
		public:
			Matrix(const std::shared_ptr<TRN::Backend::Driver> &driver, const std::size_t &rows = DEFAULT_ROWS, const std::size_t &cols = DEFAULT_COLS, const bool &blank = DEFAULT_BLANK);
			Matrix(const std::shared_ptr<TRN::Backend::Driver> &driver, const std::vector<float> &elements, const std::size_t &rows, const std::size_t &cols);
			Matrix(const std::shared_ptr<TRN::Backend::Driver> &driver, const float *dev_elements, const std::size_t &rows, const std::size_t &cols, const std::size_t &stride);
			Matrix(const std::shared_ptr<TRN::Backend::Driver> &driver, const std::shared_ptr<TRN::Core::Matrix> &matrix, const std::size_t &row, const std::size_t &col, const std::size_t &rows = DEFAULT_ROWS, const std::size_t &cols = DEFAULT_COLS);
			virtual ~Matrix();

		public :
			void from(const TRN::Core::Matrix &matrix);
			void from(const std::vector<float> &elements, const std::size_t &rows, const std::size_t &cols);
			void to(std::vector<float> &elements, std::size_t &rows, std::size_t &cols);
			void to(TRN::Core::Matrix &matrix);

		public:
			float *get_elements() const;
	
			const std::size_t &get_rows() const;
			const std::size_t &get_cols() const;
			const std::size_t &get_width() const;
			const std::size_t &get_height() const;
			const std::size_t &get_stride() const;

		private :
			std::vector<float> to_device_storage(const std::vector<float> &host, const std::size_t &rows, const std::size_t &cols);
			std::vector<float> to_host_storage(const std::vector<float> &device, const std::size_t &rows, const std::size_t &cols);
		public :
			static std::shared_ptr<Matrix> create(const std::shared_ptr<TRN::Backend::Driver> &driver, const std::size_t &rows = DEFAULT_ROWS, const std::size_t &cols = DEFAULT_COLS, const bool &blank = DEFAULT_BLANK);
			static std::shared_ptr<TRN::Core::Matrix> create(const std::shared_ptr<TRN::Backend::Driver> &driver, const float *dev_elements, const std::size_t &rows, const std::size_t &cols, const std::size_t &stride);
			static std::shared_ptr<Matrix> create(const std::shared_ptr<TRN::Backend::Driver> &driver, const std::vector<float> &elements, const std::size_t &rows, const std::size_t &cols);
			static std::shared_ptr<Matrix> create(const std::shared_ptr<TRN::Backend::Driver> &driver, const std::shared_ptr<TRN::Core::Matrix> &matrix, const std::size_t &row, const std::size_t &col, const std::size_t &rows = DEFAULT_ROWS, const std::size_t &cols = DEFAULT_COLS);
		};
	};
};
