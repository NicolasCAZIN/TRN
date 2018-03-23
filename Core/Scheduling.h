#pragma once

#include "core_global.h"

namespace TRN
{
	namespace Core
	{
		class CORE_EXPORT Scheduling
		{
		private :
			class Handle;
			std::unique_ptr<Handle> handle;

		public :
			Scheduling(const std::vector<int> &offsets, const std::vector<int> &durations);
			Scheduling(const std::vector<std::vector<int>> &indices);
			virtual ~Scheduling();

		public :
			Scheduling &operator = (const Scheduling &scheduling);
		public :
			std::vector<int> get_offsets();
			std::vector<int> get_durations();

			void set_offsets(const std::vector<int> &offsets);
			void set_durations(const std::vector<int> &durations);
			std::size_t get_total_duration();

			void to(std::vector<int> &offsets, std::vector<int> &durations);
			void to(std::vector<std::vector<int>> &indices);
			void from(const std::vector<std::vector<int>> &indices);

		public :
			static std::shared_ptr<Scheduling> create(const std::vector<std::vector<int>> &indices);
			static std::shared_ptr<Scheduling> create(const std::vector<int> &offsets, const std::vector<int> &durations);
		};
	};
};