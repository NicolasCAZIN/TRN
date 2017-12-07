#pragma once

namespace TRN
{
	namespace Helper
	{
		template <class Implementor, template <typename> class Pointer = std::shared_ptr >
		class Bridge
		{
		protected :
			const Pointer<Implementor> implementor;

		public :
			Bridge(const Pointer<Implementor> implementor) : implementor(implementor) {}
	
			Pointer<Implementor> get_implementor() const { return implementor; }

		};
	};
};