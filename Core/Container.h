#pragma once

#include "core_global.h"

#include "Set.h"

namespace TRN
{
	namespace Core
	{
		class CORE_EXPORT Container
		{
		public :
			virtual const std::shared_ptr<TRN::Core::Set> retrieve_set(const std::string &label, const std::string &tag) = 0;
			virtual const std::shared_ptr<TRN::Core::Matrix> retrieve_sequence(const std::string &label, const std::string &tag) = 0;
		};
	};
};
