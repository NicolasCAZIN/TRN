#pragma once

namespace TRN
{
	namespace Helper
	{
		template <class Decorated>
		class Decorator : public Decorated
		{
		protected:
			mutable std::shared_ptr<Decorated> decorated;

		protected:
			Decorator(const std::shared_ptr<Decorated> decorated) : decorated(decorated) {}
			virtual ~Decorator()
			{
				decorated.reset();
			}
		};
	};
};
