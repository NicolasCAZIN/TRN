#pragma once

#include "Singleton.h"
#include <map>
namespace TRN
{
	namespace Helper
	{
		template<typename BaseType, typename ... Args>
		class Factory : public Singleton<TRN::Helper::Factory<BaseType, Args...>>
		{
		private:

			class Interface
			{
			public:
				virtual std::shared_ptr<BaseType> create(Args &&...args) = 0;
			};

			template<typename Type>
			class Creator : public Interface
			{
			public:
				virtual std::shared_ptr<BaseType> create(Args &&...args)
				{
					return std::shared_ptr<BaseType>(new Type(std::forward<Args>(args)...));
				}
			};


		private:
			std::map<std::string, std::unique_ptr<Interface > > creators;

		public:
			template<typename Type>
			void reg(const std::string &type)
			{
				creators[type].reset(new Helper::Factory<BaseType, Args...>::Creator<Type>);
			}

			std::shared_ptr<BaseType> create(const std::string &type, Args &&...args)
			{
				return creators[type]->create(std::forward<Args>(args)...);
			}
		};
		template<typename BaseType, typename Type, typename...Args>
		class Register
		{
		public:
			Register(const std::string &type)
			{
				Factory<BaseType, Args...>::instance()->template reg<Type>(type);
			}
		};
	};
}