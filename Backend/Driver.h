#pragma once

#include "backend_global.h"

#include "Memory.h"
#include "Random.h"
#include "Algorithm.h"

namespace TRN
{
	namespace Backend
	{
		class BACKEND_EXPORT Driver
		{
		protected :
			class Handle;
			std::unique_ptr<Handle> handle;

		public :
			Driver(const std::shared_ptr<TRN::Backend::Memory> &memory, const std::shared_ptr<TRN::Backend::Random> &random, const std::shared_ptr<TRN::Backend::Algorithm> &algorithm);
			virtual ~Driver();

		public :
			const std::shared_ptr<TRN::Backend::Memory> &get_memory();
			const std::shared_ptr<TRN::Backend::Random> &get_random();
			const std::shared_ptr<TRN::Backend::Algorithm> &get_algorithm();
		
			virtual void dispose() = 0;
			virtual void synchronize() = 0;
			virtual void toggle() = 0;
			virtual std::string name() = 0;
			virtual int index() = 0;
		};
	};
};
