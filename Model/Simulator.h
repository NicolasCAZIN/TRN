#pragma once

#include "model_global.h"

#include "Simulator/Basic.h"
#include "Simulator/States.h"
#include "Simulator/Weights.h"
#include "Simulator/Performances.h"


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
					const std::function<void(const std::string &phase, const std::string &label, const std::vector<float> &data, const std::size_t &rows, const std::size_t &cols)> &functor,
					const bool &train, const bool &prime, const bool &test);
			};
			namespace Weights
			{
				std::shared_ptr<TRN::Simulator::Weights> MODEL_EXPORT create(const std::shared_ptr<TRN::Core::Simulator> &decorated,
					const std::function<void(const std::string &phase, const std::string &label, const std::vector<float> &data, const std::size_t &rows, const std::size_t &cols)> &functor,
					const bool &train, const bool &initialization);
			};
			namespace Performances
			{
				std::shared_ptr<TRN::Simulator::Performances> MODEL_EXPORT create(const std::shared_ptr<TRN::Core::Simulator> &decorated,
					const std::function<void(const std::string &phase, const size_t &batch_size, const size_t &cycles, const float &gflops, const float &cycles_per_second)> &functor,
					const bool &train, const bool &prime, const bool &test);
			};
		};
	};
};