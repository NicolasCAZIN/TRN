#pragma once

#include "Node.h"
#include "Cache.h"
class TRN::Engine::Node::Handle
{
public :
	//std::string name;

	bool disposed;
	bool configured_required;
	std::shared_ptr<TRN::Engine::Cache> cache;
	std::size_t remaining_initializations;
	std::map<unsigned long long, std::function<void(const unsigned long long &evaluation_id, const std::vector<float> &stimulus, const std::size_t &rows, const std::size_t &cols)>> perceived_stimulus;
	std::map<unsigned long long, std::function<void(const unsigned long long &evaluation_id, const std::vector<float> &estimated_position, const std::size_t &rows, const std::size_t &cols)>> estimated_position;
	std::map<unsigned long long, std::function<void(const unsigned long long &evaluation_id, const std::vector<int> &offsets, const std::vector<int> &durations)>> scheduler;
	std::map<unsigned long long, std::function<void(const unsigned long long &evaluation_id, const std::vector<int> &offsets, const std::vector<int> &durations)>> mutator;
	std::map<unsigned long long, std::function<void(const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)>> feedforward_weights;
	std::map<unsigned long long, std::function<void(const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)>> feedback_weights;
	std::map<unsigned long long, std::function<void(const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)>> readout;
	std::map<unsigned long long, std::function<void(const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)>> recurrent;
	int rank;

};