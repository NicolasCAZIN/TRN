#pragma once

namespace TRN
{
	namespace Helper
	{
		template <class Implementor>
		class Bridge
		{
		protected :
			const std::shared_ptr<Implementor> implementor;

		public :
			Bridge(const std::shared_ptr<Implementor> implementor) : implementor(implementor ? implementor : throw std::invalid_argument("Implementor is not initialized")) {}
			virtual ~Bridge() {}
			std::shared_ptr<Implementor> get_implementor() const { return implementor; }

		};
	};
};