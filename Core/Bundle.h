#pragma once

#include "core_global.h"
#include "Core/Batch.h"

namespace TRN
{
	namespace Core
	{
		class CORE_EXPORT Bundle : public TRN::Helper::Bridge<TRN::Backend::Driver>
		{
		private:
			class Handle;
			std::unique_ptr<Handle> handle;

		public:
			Bundle(const std::shared_ptr<TRN::Backend::Driver> &driver, const std::size_t &size);
			~Bundle();

		public:
			float ***get_elements(const bool &host = false);
			const std::size_t **get_strides();
			const std::size_t **get_rows();
			const std::size_t **get_cols();
			const std::size_t *get_strides(const std::size_t &index);
			const std::size_t *get_rows(const std::size_t &index);
			const std::size_t *get_cols(const std::size_t &index);
			std::shared_ptr<TRN::Core::Batch> get_batches(const std::size_t &index);
			void update(const std::size_t &index, const std::shared_ptr<TRN::Core::Batch> &batch);

		public:
			void to(std::vector<float> &elements, std::size_t &batches, std::vector<std::size_t> &matrices, std::vector<std::vector<std::size_t>> &rows, std::vector<std::vector<std::size_t>> &cols);
		private:
			void upload();

		public:
			static std::shared_ptr<Bundle> create(const std::shared_ptr<TRN::Backend::Driver> &driver, const std::size_t &size);
		};
	};
};
