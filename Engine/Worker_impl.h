#pragma once

#include "Worker.h"
#include "Core/Simulator.h"



class TRN::Engine::Worker::Handle
{
public:
	std::weak_ptr<TRN::Engine::Communicator> communicator;
	std::thread receiver;
	std::mutex functors;
	std::shared_ptr<TRN::Backend::Driver> driver;
	std::map<unsigned int, std::shared_ptr<TRN::Core::Simulator>> simulators;
	
	int rank;

	std::map<unsigned int, std::function<void(const std::vector<float> &stimulus, const std::size_t &rows, const std::size_t &cols)>> perceived_stimulus;
	std::map<unsigned int, std::function<void(const std::vector<float> &estimated_position, const std::size_t &rows, const std::size_t &cols)>> estimated_position;
	std::map<unsigned int, std::function<void(const std::vector<unsigned int> &offsets, const std::vector<unsigned int> &durations)>> scheduler;
	std::map<unsigned int, std::function<void(const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)>> feedforward_weights;
	std::map<unsigned int, std::function<void(const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)>> feedback_weights;
	std::map<unsigned int, std::function<void(const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)>> readout;
	std::map<unsigned int, std::function<void(const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)>> recurrent;
};
