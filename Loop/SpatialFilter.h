#pragma once

#include "loop_global.h"
#include "Core/Loop.h"
#include "Core/Decoder.h"
#include "Core/Encoder.h"

namespace TRN
{
	namespace Loop
	{
		class LOOP_EXPORT SpatialFilter : 
			public TRN::Core::Loop,
			public TRN::Helper::Observer<TRN::Core::Message::Payload<TRN::Core::Message::STIMULUS>>,
			public TRN::Helper::Observer<TRN::Core::Message::Payload<TRN::Core::Message::POSITION>>
		{
		private :
			class Handle;
			std::unique_ptr<Handle> handle;

		public:
			SpatialFilter(const std::shared_ptr<TRN::Backend::Driver> &driver, const std::size_t &batch_size, const std::size_t &stimulus_size, 
				/*const std::function<void(const unsigned long long &evaluation_id, const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols)> &predicted_position,
				std::function<void(const unsigned long long &evaluation_id, const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols)> &estimated_position,
				const std::function<void(const unsigned long long &evaluation_id, const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols)> &predicted_stimulus,
				std::function<void(const unsigned long long &evaluation_id, const std::vector<float> &stimulus, const std::size_t &rows, const std::size_t &cols)> &perceived_stimulus,*/
				const std::shared_ptr<TRN::Core::Encoder> &encoder,
				const std::shared_ptr<TRN::Core::Decoder> &decoder,
				const std::string &tag);
			virtual ~SpatialFilter();


		public:
			virtual void update(const TRN::Core::Message::Payload<TRN::Core::Message::TEST> &payload) override;
			virtual void update(const TRN::Core::Message::Payload<TRN::Core::Message::PREDICTION> &payload) override;
			virtual void update(const TRN::Core::Message::Payload<TRN::Core::Message::STIMULUS> &payload) override;
			virtual void update(const TRN::Core::Message::Payload<TRN::Core::Message::POSITION> &payload) override;
			virtual void visit(std::shared_ptr<TRN::Core::Message::Payload<TRN::Core::Message::FLOPS>> &payload) override;

		private :
			void location_at(const std::size_t &t, std::shared_ptr<TRN::Core::Matrix> &location);

		public :
			static std::shared_ptr<SpatialFilter> create(const std::shared_ptr<TRN::Backend::Driver> &driver, const std::size_t &batch_size, const std::size_t &stimulus_size,
				/*const std::function<void(const unsigned long long &evaluation_id, const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols)> &predicted_position,
				std::function<void(const unsigned long long &evaluation_id, const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols)> &estimated_position,
				const std::function<void(const unsigned long long &evaluation_id, const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols)> &predicted_stimulus,
				std::function<void(const unsigned long long &evaluation_id, const std::vector<float> &stimulus, const std::size_t &rows, const std::size_t &cols)> &perceived_stimulus,*/
				const std::shared_ptr<TRN::Core::Encoder> &encoder,
				const std::shared_ptr<TRN::Core::Decoder> &decoder,

				const std::string &tag);
		};
	};
};

