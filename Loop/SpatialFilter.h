#pragma once

#include "loop_global.h"
#include "Core/Loop.h"

namespace TRN
{
	namespace Loop
	{
		class LOOP_EXPORT SpatialFilter : 
			public TRN::Core::Loop
		{
		private :
			class Handle;
			mutable std::unique_ptr<Handle> handle;

		public:
			SpatialFilter(const std::shared_ptr<TRN::Backend::Driver> &driver, const std::size_t &batch_size, const std::size_t &stimulus_size, const unsigned long &seed,
				const std::function<void(const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols)> &predicted_position,
				std::function<void(const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols)> &estimated_position,
				const std::function<void(const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols)> &predicted_stimulus,
				std::function<void(const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &stimulus, const std::size_t &rows, const std::size_t &cols)> &perceived_stimulus,
				const std::size_t &rows, const std::size_t &cols,
				const std::pair<float, float> &x, const std::pair<float, float> &y,
				const std::shared_ptr<TRN::Core::Matrix> &firing_rate_map,
				const float &sigma, 
				const float &radius,
				const float &scale,
				const std::string &tag);
			virtual ~SpatialFilter();

		private :


		public:
			virtual void update(const TRN::Core::Message::Payload<TRN::Core::Message::TEST> &payload) override;
			virtual void update(const TRN::Core::Message::Payload<TRN::Core::Message::PREDICTION> &payload) override;
			virtual void visit(std::shared_ptr<TRN::Core::Message::Payload<TRN::Core::Message::FLOPS>> &payload) override;

		public:
			static std::shared_ptr<SpatialFilter> create(const std::shared_ptr<TRN::Backend::Driver> &driver, const std::size_t &batch_size, const std::size_t &stimulus_size, const unsigned long &seed,
				const std::function<void(const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols)> &predicted_position,
				std::function<void(const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols)> &estimated_position,
				const std::function<void(const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols)> &predicted_stimulus,
				std::function<void(const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &stimulus, const std::size_t &rows, const std::size_t &cols)> &perceived_stimulus,
				const std::size_t &rows, const std::size_t &cols,
				const std::pair<float, float> &x, const std::pair<float, float> &y,
				const std::shared_ptr<TRN::Core::Matrix> &firing_rate_map,
				const float &sigma, 
				const float &radius,
				const float &scale,
				const std::string &tag);
		};
	};
};

