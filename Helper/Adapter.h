#pragma once

namespace TRN
{
	namespace Helper
	{
		template <class Interface, class Adaptee>
		class Adapter : public Interface
		{
		protected:
			mutable std::shared_ptr<Adaptee> adaptee;

		protected:
			Adapter(const std::shared_ptr<Adaptee> adaptee) : adaptee(adaptee) {}
		};
	};
};