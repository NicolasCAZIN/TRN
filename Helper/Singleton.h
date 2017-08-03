#pragma once

namespace TRN
{
	namespace Helper
	{
		template <class Type>
		class Singleton
		{
		protected:
			Singleton() {}

		public:
			static std::shared_ptr<Type> instance()
			{
				static auto instance = std::make_shared<Type>();

				return instance;
			}
		};
	}
}
