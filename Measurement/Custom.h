#pragma once

#include "measurement_global.h"
#include "Core/Measurement.h"

namespace TRN
{
	namespace Measurement
	{
		class MEASUREMENT_EXPORT Custom :
			public TRN::Core::Measurement::Implementation
		{
		private:
			class Handle;
			std::unique_ptr<Handle> handle;
		public:
			Custom(const std::shared_ptr<TRN::Backend::Driver> &driver, const std::function<void(const unsigned long long &evaluation_id, const std::vector<float> &primed, const std::vector<float> &predicted, const std::vector<float> &expected, const std::size_t &preamble, const std::size_t &pages, const std::size_t &rows, const std::size_t &cols)> &functor);

		public:
			virtual void compute(const unsigned long long &evaluation_id, const std::shared_ptr<TRN::Core::Matrix> &primed, const std::shared_ptr<TRN::Core::Batch> &predicted, const std::shared_ptr<TRN::Core::Matrix> &expected, const std::shared_ptr<TRN::Core::Matrix> &error) override;

		public:
			static std::shared_ptr <TRN::Measurement::Custom> create(const std::shared_ptr<TRN::Backend::Driver> &driver, const std::function<void(const unsigned long long &evaluation_id, const std::vector<float> &primed, const std::vector<float> &predicted, const std::vector<float> &expected, const std::size_t &preamble, const std::size_t &pages, const std::size_t &rows, const std::size_t &cols)> &functor);
		};

	};
};
