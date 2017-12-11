#pragma once

#include "cpu_global.h"
#include "Backend/Driver.h"
#include "Implementation.h"

namespace TRN
{
	namespace CPU
	{ 
		template<TRN::CPU::Implementation>
		class CPU_EXPORT Driver : public TRN::Backend::Driver
		{
		public :
			Driver();
		public:
			virtual void dispose() override;
			virtual void synchronize() override;
			virtual std::string name() override;
			virtual int index() override;
			virtual void toggle() override;

		public :
			static std::shared_ptr<Driver> create();
		};
		std::list<std::pair<int, std::string>> CPU_EXPORT enumerate_devices();
	};
};

