#pragma once

#include "core_global.h"
#include "Helper/Observer.h"
#include "Message.h"
#include "Batch.h"

namespace TRN
{
	namespace Core
	{
		namespace Measurement
		{
			class CORE_EXPORT Implementation :
				public TRN::Helper::Bridge<TRN::Backend::Driver>
			{
			protected:
				Implementation(const std::shared_ptr<TRN::Backend::Driver> &driver);
				virtual ~Implementation() noexcept(false);

			public:
				virtual void compute(const unsigned long long &evaluation_id, const std::shared_ptr<TRN::Core::Matrix> &primed, const std::shared_ptr<TRN::Core::Batch> &predicted, const std::shared_ptr<TRN::Core::Matrix> &expected, const std::shared_ptr<TRN::Core::Matrix> &error) = 0;
			};

			class CORE_EXPORT Abstraction :
				public TRN::Helper::Bridge<TRN::Core::Measurement::Implementation>,
				public TRN::Helper::Observer<TRN::Core::Message::Payload<TRN::Core::Message::TEST>>,
				public TRN::Helper::Observer<TRN::Core::Message::Payload<TRN::Core::Message::PREDICTION>>,
				public TRN::Helper::Observer<TRN::Core::Message::Payload<TRN::Core::Message::POSITION>>,
				public TRN::Helper::Observer<TRN::Core::Message::Payload<TRN::Core::Message::TARGET_SEQUENCE>>,
				public TRN::Helper::Observer<TRN::Core::Message::Payload<TRN::Core::Message::TARGET_TRAJECTORY>>
			{
			protected:
				class Handle;
				std::unique_ptr<Handle> handle;

			protected :
				Abstraction(const std::shared_ptr<TRN::Core::Measurement::Implementation> &compute, const std::size_t &batch_size);
				virtual ~Abstraction() noexcept(false);
			protected :
				virtual void update(const TRN::Core::Message::Payload<TRN::Core::Message::TEST> &payload) override;
			protected:
	
				void set_expected(const std::shared_ptr<TRN::Core::Matrix> &expected);
				void on_update(const unsigned long long &evaluation_id, const std::shared_ptr<TRN::Core::Batch> &predicted);

			};
		};
	};
};