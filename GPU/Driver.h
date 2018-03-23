#pragma once

#include "gpu_global.h"
#include "Backend/Driver.h"
#include "Context.h"

namespace TRN
{
	namespace GPU
	{

		class GPU_EXPORT Driver : public TRN::Backend::Driver
		{
		private :
			class Handle;
			std::unique_ptr<Handle> handle;

		public:
			Driver( const int &device);
			virtual ~Driver();
		private :
			Driver(const std::shared_ptr<TRN::GPU::Context> context);
		public:
			virtual void synchronize() override;
			virtual std::string name() override;
			virtual int index() override;
			virtual void toggle() override;
			virtual void dispose() override;
		public:
			static std::shared_ptr<Driver> create( const int &device);
		};

		std::list<std::pair<int, std::string>> GPU_EXPORT enumerate_devices();

	};
};