#pragma once

#include "measurement_global.h"
#include "Core/Measurement.h"

namespace TRN
{
	namespace Measurement
	{
		class MEASUREMENT_EXPORT Position :
			public TRN::Core::Measurement::Abstraction
		{
		public:
			Position(const std::shared_ptr<TRN::Core::Measurement::Implementation> &compute, const std::size_t &batch_size);

		public:
			void update(const TRN::Core::Message::Payload<TRN::Core::Message::PREDICTION> &payload) override;
			void update(const TRN::Core::Message::Payload<TRN::Core::Message::POSITION> &payload) override;
			void update(const TRN::Core::Message::Payload<TRN::Core::Message::TARGET_SEQUENCE> &payload) override;
			void update(const TRN::Core::Message::Payload<TRN::Core::Message::TARGET_TRAJECTORY> &payload) override;

		public :
			static std::shared_ptr<TRN::Measurement::Position> create(const std::shared_ptr<TRN::Core::Measurement::Implementation> &compute, const std::size_t &batch_size);
		};
	};
};
