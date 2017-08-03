#pragma once

namespace TRN
{
	namespace Helper
	{
		// strategie d'evaluation de reseaux
		// strategie d'entrainement de reseaux
		template <class Behavior>
		class Strategy
		{
		protected:
			mutable std::shared_ptr<Behavior> behavior;

		public:
			void set_behavior(const std::shared_ptr<Behavior> behavior)
			{
				this->behavior = behavior;
			}
		};
	};
};