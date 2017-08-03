/*#pragma once

#include "cpu_global.h"
#include "Core/Reservoir.h"

namespace CPU
{
	//template<bool gather_states, bool gather_weights>
	class CPU_EXPORT Reservoir : public Core::Reservoir
	{
	private :
		std::function<float(void)> random_generator;
		float *c;
		float *p;

		std::shared_ptr<Data::States> states;

	public :
		Reservoir(int stimulus_size, int readout_size, int reservoir_size,
			float leak_rate, float learning_rate, float initial_state_scale, int seed,
			std::shared_ptr<Core::Initializer> feedforward,
			std::shared_ptr<Core::Initializer> recurrent,
			std::shared_ptr<Core::Initializer> feedback,
			std::shared_ptr<Core::Initializer> readout
			);
		virtual ~Reservoir();

	public :
		void train(std::shared_ptr<Core::Dataset> dataset) Q_DECL_OVERRIDE;

	public :
		void visit(std::shared_ptr<Data::States> states) Q_DECL_OVERRIDE;
	};
};*/