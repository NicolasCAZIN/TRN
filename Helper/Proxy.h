#pragma once

namespace TRN
{
	namespace Helper
	{
		template <class Subject>
		class Proxy : public Subject
		{
		private:
			mutable std::shared_ptr<Subject> subject;
		};
	};
};
