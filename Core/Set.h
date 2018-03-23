#pragma once

#include "core_global.h"
#include "Matrix.h"
#include "Scheduling.h"

namespace TRN
{
	namespace Core
	{
		class CORE_EXPORT Set
		{
		private :
			class Handle;
			std::unique_ptr<Handle> handle;

		public :
			Set(const std::shared_ptr<TRN::Backend::Driver> &driver, const std::vector<std::shared_ptr<TRN::Core::Matrix>> &sequences);
			~Set();

		public :
			const std::shared_ptr<TRN::Core::Matrix> &get_sequence();
			const std::shared_ptr<TRN::Core::Scheduling> &get_scheduling();

		public :
			static std::shared_ptr<TRN::Core::Set> create(const std::shared_ptr<TRN::Backend::Driver> &driver, const std::vector<std::shared_ptr<TRN::Core::Matrix>> &sequences);
		};
	};
};