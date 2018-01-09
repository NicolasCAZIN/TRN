#pragma once

#include "core_global.h"
#include "Core/Matrix.h"

namespace TRN
{
	namespace Core
	{
		class CORE_EXPORT Batch : public TRN::Helper::Bridge<TRN::Backend::Driver>
		{
		private :
			class Handle;
			std::unique_ptr<Handle> handle;

		public :
			Batch(const std::shared_ptr<TRN::Backend::Driver> &driver, const std::size_t &size);
			~Batch();

		public :
			float **get_elements(const bool &host = false);
			const std::size_t get_size();
			const std::size_t *get_strides();
			const std::size_t *get_rows();
			const std::size_t *get_cols();
			const std::size_t get_strides(const std::size_t &index);
			const std::size_t get_rows(const std::size_t &index);
			const std::size_t get_cols(const std::size_t &index);
			std::shared_ptr<TRN::Core::Matrix> get_matrices(const std::size_t &index);
			void update(const std::size_t &index, const std::shared_ptr<TRN::Core::Matrix> &matrix);
			void update(const std::size_t &index, float *elements, const std::size_t &rows, const std::size_t &cols, const std::size_t &stride);

		public :
			void from(const TRN::Core::Batch &batch);
			void to(std::vector<float> &elements, std::size_t &matrices, std::vector<std::size_t> &rows, std::vector<std::size_t> &cols);
		private :
			void upload();

		public :
			static std::shared_ptr<Batch> create(const std::shared_ptr<TRN::Backend::Driver> &driver, const std::size_t &size);
		};
	};
};
