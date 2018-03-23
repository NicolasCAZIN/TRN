#pragma once

#include "loop_global.h"
#include "Core/Loop.h"

#include "Helper/Observer.h"

namespace TRN
{
	namespace Loop
	{
		class LOOP_EXPORT Custom : 
			public TRN::Core::Loop
		{
		private:
			class Handle;
			std::unique_ptr<Handle> handle;

		public:
			Custom(const std::shared_ptr<TRN::Backend::Driver> &driver, const std::size_t &batch_size, const std::size_t &stimulus_size,
				const std::function<void(const unsigned long long &evaluation_id, const std::vector<float> &prediction, const std::size_t &rows, const std::size_t &cols)> &prediction,
				std::function<void(const unsigned long long &evaluation_id, const std::vector<float> &stimulus, const std::size_t &rows, const std::size_t &cols)> &stimulus);
			virtual ~Custom();

		public:
			virtual void update(const TRN::Core::Message::Payload<TRN::Core::Message::PREDICTION> &payload) override;
			virtual void visit(std::shared_ptr<TRN::Core::Message::Payload<TRN::Core::Message::FLOPS>> &payload) override;
		public:
			static std::shared_ptr<Custom> create(const std::shared_ptr<TRN::Backend::Driver> &driver, const std::size_t &batch_size, const std::size_t &stimulus_size,
				const std::function<void(const unsigned long long &evaluation_id, const std::vector<float> &prediction, const std::size_t &rows, const std::size_t &cols)> &prediction,
				std::function<void(const unsigned long long &evaluation_id, const std::vector<float> &stimulus, const std::size_t &rows, const std::size_t &cols)> &stimulus);
		};
	};
};

