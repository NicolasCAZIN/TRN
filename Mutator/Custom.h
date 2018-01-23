#pragma once

#include "mutator_global.h"
#include "Core/Mutator.h"

namespace TRN
{
	namespace Mutator
	{
		class MUTATOR_EXPORT Custom : public TRN::Core::Mutator
		{
		private:
			class Handle;
			std::unique_ptr<Handle> handle;

		public:
			Custom(const unsigned long &seed, const std::function<void(const unsigned long long &evaluation_id, const unsigned long &seed, const std::vector<int> &offsets, const std::vector<int> &durations)> &request,
				std::function<void(const unsigned long long &evaluation_id, const std::vector<int> &offsets, const std::vector<int> &durations)> &reply);
			virtual ~Custom();

		public:
			virtual void update(const TRN::Core::Message::Payload<TRN::Core::Message::SCHEDULING> &payload) override;

		public :
			static std::shared_ptr<Custom> create(const unsigned long &seed, const std::function<void(const unsigned long long &evaluation_id, const unsigned long &seed, const std::vector<int> &offsets, const std::vector<int> &durations)> &request, std::function<void(const std::size_t &trial, const std::vector<int> &offsets, const std::vector<int> &durations)> &reply);
		};
	};
};


