#pragma once

#include "mutator_global.h"
#include "Core/Mutator.h"

namespace TRN
{
	namespace Mutator
	{
		class MUTATOR_EXPORT Shuffle : public TRN::Core::Mutator
		{
		public:
			virtual void update(const TRN::Core::Message::Payload<TRN::Core::Message::SCHEDULING> &payload) override;

		public:
			static std::shared_ptr<Shuffle> create();

		};
	};
};


