#pragma once

#include "core_global.h"
#include "Helper/Bridge.h"
#include "Backend/Driver.h"

namespace TRN
{
	namespace Core
	{
		class CORE_EXPORT Scheduling :
			public TRN::Helper::Bridge<TRN::Backend::Driver>
		{
		private :
			class Handle;
			mutable std::unique_ptr<Handle> handle;

		public :
			Scheduling(const std::shared_ptr<TRN::Backend::Driver> &driver, const std::vector<unsigned int> &offsets, const std::vector<unsigned int> &durations);
			virtual ~Scheduling();

			unsigned int *get_offsets();
			unsigned int *get_durations();
			const std::size_t &get_repetitions();
			const std::size_t &get_total_duration();

			void to(std::vector<unsigned int> &offsets, std::vector<unsigned int> &durations);

		public :
			static std::shared_ptr<Scheduling> create(const std::shared_ptr<TRN::Backend::Driver> &driver, const std::vector<unsigned int> &offsets, const std::vector<unsigned int> &durations);
		};
	};
};