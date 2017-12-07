#pragma once

namespace TRN
{
	namespace Helper
	{


		template <typename Subject>
		class Observer
		{

		private: 
	
		public:
			Observer(){}
			//friend class TRN::Helper::Observable<Subject>;
		public:
			virtual void update(const Subject &subject) = 0;
		};


		template <typename Subject>
		class Observable
		{
		private:
		/*	using Signal = typename boost::signals2::signal<void(const Subject &subject)>;
			using Slot = typename Signal::slot_type;
			Signal sig;*/
			std::list<std::weak_ptr<TRN::Helper::Observer<Subject>>> observers;
			
	
		public:
			void attach(const std::shared_ptr<TRN::Helper::Observer<Subject>> &observer) 
			{
				observers.push_back(observer);
				//Slot f();
				//connections.push_back(sig.connect(std::bind(&TRN::Helper::Observer<Subject>::update, observer, std::placeholders::_1)));
				
			}

	
		protected :
			void notify(const Subject subject)
			{
				for (auto observer : observers)
				{
					auto locked = observer.lock();
					if (locked)
					{
						locked->update(subject);
					}
				}
			//	sig(subject);
			}
		};
	};
};
