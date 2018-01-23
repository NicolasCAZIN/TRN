#pragma once

#include "model_global.h"

#include "Simulator/Basic.h"
#include "Simulator/States.h"
#include "Simulator/Weights.h"
#include "Simulator/Performances.h"
#include "Simulator/Scheduling.h"

namespace TRN
{
	namespace Model
	{
		namespace Simulator
		{
			namespace Basic
			{
				std::shared_ptr<TRN::Simulator::Basic> MODEL_EXPORT create(const std::function<void()> &trained, const std::function<void()> &primed, const std::function<void()> &tested);
			};
			namespace States
			{
				std::shared_ptr<TRN::Simulator::States> MODEL_EXPORT create(const std::shared_ptr<TRN::Core::Simulator> &decorated,
					const std::function<void(const unsigned long long &evaluation_id, const std::string &phase, const std::string &label, const std::size_t &batch,  const std::vector<float> &data, const std::size_t &rows, const std::size_t &cols)> &functor,
					const bool &train, const bool &prime, const bool &test);
			};
			namespace Weights
			{
				std::shared_ptr<TRN::Simulator::Weights> MODEL_EXPORT create(const std::shared_ptr<TRN::Core::Simulator> &decorated,
					const std::function<void(const unsigned long long &evaluation_id, const std::string &phase, const std::string &label, const std::size_t &batch, const std::vector<float> &data, const std::size_t &rows, const std::size_t &cols)> &functor,
					const bool &train, const bool &initialization);
			};
			namespace Performances
			{
				std::shared_ptr<TRN::Simulator::Performances> MODEL_EXPORT create(const std::shared_ptr<TRN::Core::Simulator> &decorated,
					const std::function<void(const unsigned long long &evaluation_id,  const std::string &phase, const float &cycles_per_second, const float &gflops_per_second)> &functor,
					const bool &train, const bool &prime, const bool &test);
			};
			namespace Scheduling
			{
				std::shared_ptr<TRN::Simulator::Scheduling> MODEL_EXPORT create(const std::shared_ptr<TRN::Core::Simulator> &decorated,
					const std::function<void(const unsigned long long &evaluation_id,  const std::vector<int> &offsets, const std::vector<int> &durations)> &functor) ;
			};
		};
	};
};