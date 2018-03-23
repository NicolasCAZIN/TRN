#pragma once

#include "initializer_global.h"
#include "Core/Initializer.h"

namespace TRN
{
	namespace Initializer
	{
		class INITIALIZER_EXPORT Custom : public TRN::Core::Initializer
		{
		private:
			class Handle;
			std::unique_ptr<Handle> handle;

		public:
			Custom(const std::shared_ptr<TRN::Backend::Driver> &driver, 
				const std::function<void(const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &request,
				std::function<void(const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &reply);
			virtual ~Custom();

		public:
			virtual void initialize(unsigned long &seed, std::shared_ptr<TRN::Core::Batch> &batch, const bool &blank_diagonal) override;

		public:
			static std::shared_ptr<Custom> create(const std::shared_ptr<TRN::Backend::Driver> &driver, 
				const std::function<void(const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &request,
				std::function<void(const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &reply);
		};
	};
};

