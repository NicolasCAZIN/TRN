#pragma once

#include "scheduler_global.h"
#include "Core/Scheduler.h"

namespace TRN
{
	namespace Scheduler
	{
		class SCHEDULER_EXPORT Snippets : public TRN::Core::Scheduler
		{
		private :
			class Handle;
			std::unique_ptr<Handle> handle;

		public:
			Snippets(const unsigned long &seed, const unsigned int &snippets_size, const unsigned int &time_budget,
				const float &learn_reverse_rate, const float &generate_reverse_rate,
				const float &learning_rate,
				const float &discount, const std::string &tag = "");
			virtual ~Snippets();

		private :
			std::vector<int> draw_snippet(const std::vector<float> &V, const std::vector<float> &R, const std::vector<int> &sequence_offset, const std::vector<int> &set_offset, const std::vector<int> &batch_durations, const std::vector<int> &sequence_number, const float &reverse_rate);

		public :
			virtual void update(const TRN::Core::Message::Payload<TRN::Core::Message::SET> &payload) override;
			virtual void reset() override;
		public :
			static std::shared_ptr<Snippets> create(const unsigned long &seed, const unsigned int &snippets_size, const unsigned int &time_budget,
				const float &learn_reverse_rate, const float &generate_reverse_rate,
				const float &learning_rate,
				const float &discount, const std::string &tag = "");
		};

	};
};
