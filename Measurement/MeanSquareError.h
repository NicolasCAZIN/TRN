#pragma once

#include "measurement_global.h"
#include "Core/Measurement.h"

namespace TRN
{
	namespace Measurement
	{
		class MEASUREMENT_EXPORT MeanSquareError :
			public TRN::Core::Measurement::Implementation
		{
		private :
			class Handle;
			std::unique_ptr<Handle> handle;

		public :
			MeanSquareError(const std::shared_ptr<TRN::Backend::Driver> &driver, const std::function<void(const unsigned long long &evaluation_id, const std::vector<float> &values, const std::size_t &rows, const std::size_t &cols)> &functor);

		public:
			virtual void compute(const unsigned long long &evaluation_id, const std::shared_ptr<TRN::Core::Matrix> &primed, const std::shared_ptr<TRN::Core::Batch> &predicted, const std::shared_ptr<TRN::Core::Matrix> &expected, const std::shared_ptr<TRN::Core::Matrix> &error) override;

		public :
			static std::shared_ptr <TRN::Measurement::MeanSquareError> create(const std::shared_ptr<TRN::Backend::Driver> &driver, const std::function<void(const unsigned long long &evaluation_id, const std::vector<float> &values, const std::size_t &rows, const std::size_t &cols)> &functor);
		};

	};
};
