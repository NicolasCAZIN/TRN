#pragma once

namespace TRN
{
	namespace Helper
	{
		template <typename Type>
		class Visitor
		{
		public:
			virtual void visit(std::shared_ptr<Type> &visitable) = 0;
		};
		template<typename Type>
		class Visitable
		{
		private:
			mutable std::shared_ptr<Type> target;

		public:
			Visitable(const std::shared_ptr<Type> &target) : target(target)
			{
			}

		public:
			void accept(const std::shared_ptr<Visitor<Type>> &visitor)
			{
				visitor->visit(target);
			}
		};
	};
};
