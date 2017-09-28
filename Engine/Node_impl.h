#pragma once

#include "Node.h"

class TRN::Engine::Node::Handle
{
public :
	//std::string name;
	bool configured_required;
	std::size_t remaining_initializations;
	std::map<unsigned int, std::function<void(const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &stimulus, const std::size_t &rows, const std::size_t &cols)>> perceived_stimulus;
	std::map<unsigned int, std::function<void(const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &estimated_position, const std::size_t &rows, const std::size_t &cols)>> estimated_position;
	std::map<unsigned int, std::function<void(const std::size_t &trial, const std::vector<int> &offsets, const std::vector<int> &durations)>> scheduler;
	std::map<unsigned int, std::function<void(const std::size_t &trial, const std::vector<int> &offsets, const std::vector<int> &durations)>> mutator;
	std::map<unsigned int, std::function<void(const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)>> feedforward_weights;
	std::map<unsigned int, std::function<void(const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)>> feedback_weights;
	std::map<unsigned int, std::function<void(const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)>> readout;
	std::map<unsigned int, std::function<void(const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)>> recurrent;
	int rank;
};