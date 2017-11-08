#pragma once

#include "core_global.h"
#include "Batch.h"

namespace TRN
{
	namespace Core
	{
		class CORE_EXPORT Initializer : 
			public TRN::Helper::Bridge<TRN::Backend::Driver>
		{
		public :
			static const bool DEFAULT_BLANK_DIAGONAL;

		protected :
			Initializer(const std::shared_ptr<TRN::Backend::Driver> &driver);

		public :
			virtual ~Initializer();

		public:
			virtual void initialize(unsigned long &seed, std::shared_ptr<TRN::Core::Batch> &batch, const bool &blank_diagonal = DEFAULT_BLANK_DIAGONAL) = 0;
		};
	};
};

