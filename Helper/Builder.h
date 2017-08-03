#pragma once

namespace TRN
{
	namespace Helper
	{
		template <class Built>
		class Builder
		{
		public:
			virtual const std::shared_ptr<Built> build() = 0;
		};
	};
};
