#pragma once

#include "mutator_global.h"
#include "Core/Mutator.h"

namespace TRN
{
	namespace Mutator
	{
		class MUTATOR_EXPORT Shuffle : public TRN::Core::Mutator
		{
		private :
			class Handle;
			std::unique_ptr<Handle> handle;
		public :
			Shuffle(const unsigned long &seed);
			virtual ~Shuffle();


		public:
			virtual void update(const TRN::Core::Message::Payload<TRN::Core::Message::SCHEDULING> &payload) override;

		public:
			static std::shared_ptr<Shuffle> create(const unsigned long &seed);

		};
	};
};


