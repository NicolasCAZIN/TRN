#pragma once

#include "mutator_global.h"
#include "Core/Mutator.h"

namespace TRN
{
	namespace Mutator
	{
		class MUTATOR_EXPORT Punch : public TRN::Core::Mutator
		{
		private:
			class Handle;
			std::unique_ptr<Handle> handle;

		public:
			Punch(const unsigned long &seed, const float &rate, const std::size_t &size, const std::size_t &counter);
			virtual ~Punch();

		public:
			virtual void update(const TRN::Core::Message::Payload<TRN::Core::Message::SCHEDULING> &payload) override;

		public :
			static std::shared_ptr<Punch> create(const unsigned long &seed, const float &rate, const std::size_t &size, const std::size_t &counter);
		};
	};
};


