#pragma once

namespace TRN
{
	namespace Helper
	{
		template<class Interface>
		class Delegate : public Interface
		{
		public :
			virtual ~Delegate()
			{
	
			}
		};

		template<class Interface>
		class Delegator
		{
		public :
			Delegator()
			{
			}

			virtual ~Delegator()
			{
				delegate.reset();
			}

		protected:
			mutable std::weak_ptr<Interface> delegate;

		public:
	
			void set_delegate(const std::shared_ptr<Interface> &delegate) const
			{
				this->delegate = delegate;
			}
		};
	};
}
