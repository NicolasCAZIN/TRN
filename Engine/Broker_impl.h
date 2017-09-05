#pragma once

#include "Broker.h"
#include "Manager.h"
#include "Communicator.h"
#include "Helper/Queue.h"
class TRN::Engine::Broker::Handle 
{
public :


public :
	size_t count;
	std::shared_ptr<TRN::Engine::Manager> manager;
	std::shared_ptr<TRN::Engine::Communicator> communicator;
	
	std::mutex functors;
	std::mutex ack;
	//std::condition_variable transaction_pending;
	
	std::map<int, std::function<void()>> on_ack;
	std::thread receive;
	std::function<void(const int &rank, const std::string &host, const unsigned int &index, const std::string &name)> processor;
	std::function<void(const unsigned int &id, const int &rank)> on_allocation;
	std::function<void(const unsigned int &id, const int &rank)> on_deallocation;

	std::map<unsigned int, std::function<void(const std::vector<float> &values,  const std::size_t &rows, const  std::size_t &cols)>> measurement_readout_mean_square_error;
	std::map<unsigned int, std::function<void(const std::vector<float> &values,  const std::size_t &rows, const  std::size_t &cols)>> measurement_readout_frechet_distance;
	std::map<unsigned int, std::function<void(const std::vector<float> &predicted, const std::vector<float> &expected, const std::size_t &pages,  const std::size_t &rows, const  std::size_t &cols)>> measurement_readout_custom;
	std::map<unsigned int, std::function<void(const std::vector<float> &values,  const std::size_t &rows, const  std::size_t &cols)>> measurement_position_mean_square_error;
	std::map<unsigned int, std::function<void(const std::vector<float> &values,  const std::size_t &rows, const  std::size_t &cols)>> measurement_position_frechet_distance;
	std::map<unsigned int, std::function<void(const std::vector<float> &predicted, const std::vector<float> &expected, const std::size_t &pages, const std::size_t &rows, const  std::size_t &cols)>> measurement_position_custom;

	std::map<unsigned int, std::function<void(const std::string &phase, const size_t &batch_size, const size_t &cycles, const float &gflops, const float &seconds)>> performances;
	std::map<unsigned int, std::function<void(const std::string &phase, const std::string &label, const std::vector<float> &samples, const std::size_t &rows, const std::size_t &cols)>> states;
	std::map<unsigned int, std::function<void(const std::string &phase, const std::string &label, const std::vector<float> &samples, const std::size_t &rows, const std::size_t &cols)>> weights;
	std::map<unsigned int, std::function<void(const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols)>> predicted_position;
	std::map<unsigned int, std::function<void(const std::vector<float> &stimulus, const std::size_t &rows, const std::size_t &cols)>> predicted_stimulus;

	std::map<unsigned int, std::function<void(const std::vector<int> &offsets, const std::vector<int> &durations)> > mutator;
	std::map<unsigned int, std::function<void(const std::vector<float> &elements, const std::size_t &rows, const std::size_t &cols, const std::vector<int> &offsets, const std::vector<int> &durations)> > scheduler;
	std::map<unsigned int, std::function<void(const std::vector<int> &offsets, const std::vector<int> &durations)> > scheduling;

	std::map<unsigned int, std::function<void(const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const  std::size_t &cols)>> feedforward;
	std::map<unsigned int, std::function<void(const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const  std::size_t &cols)>> feedback;
	std::map<unsigned int, std::function<void(const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const  std::size_t &cols)>> readout;
	std::map<unsigned int, std::function<void(const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)>> recurrent;
};
