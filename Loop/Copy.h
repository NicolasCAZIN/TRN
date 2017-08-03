#pragma once

#include "loop_global.h"
#include "Core/Loop.h"

namespace TRN
{
	namespace Loop
	{
		class LOOP_EXPORT Copy : public TRN::Core::Loop
		{
		public:
			Copy(const std::shared_ptr<TRN::Backend::Driver> &driver, const std::size_t &batch_size, const std::size_t &stimulus_size);

		public :
			virtual void update(const TRN::Core::Message::Payload<TRN::Core::Message::PREDICTION> &payload) override;

			virtual void visit(std::shared_ptr<TRN::Core::Message::Payload<TRN::Core::Message::FLOPS>> &payload) override;
		public :
			static std::shared_ptr<Copy> create(const std::shared_ptr<TRN::Backend::Driver> &driver, const std::size_t &batch_size, const std::size_t &stimulus_size);
		};
	};
};

