#pragma once

#include "scheduler_global.h"
#include "Core/Scheduler.h"

namespace TRN
{
	namespace Scheduler
	{
		class SCHEDULER_EXPORT Tiled : public TRN::Core::Scheduler
		{
		private :
			class Handle;
			std::unique_ptr<Handle> handle;

		public:
			Tiled(const std::shared_ptr<TRN::Backend::Driver> &driver, const unsigned int &epochs);
			~Tiled();

		public :
			virtual void update(const TRN::Core::Message::Payload<TRN::Core::Message::SET> &payload) override;

		public:
			static std::shared_ptr<Tiled> create(const std::shared_ptr<TRN::Backend::Driver> &driver, const unsigned int &epochs);
		};

	};
};
