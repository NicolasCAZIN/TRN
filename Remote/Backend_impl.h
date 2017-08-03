#pragma once

#include "Backend.h"
#include "Network/Manager.h"
class TRN::Remote::Backend::Handle
{
public:
	std::map<int, std::function<void(const std::string &phase, const float &cycles_per_second)>> performances;
	std::map<int, std::function<void(const std::string &label, const std::vector<float> &samples, const std::size_t &rows, const std::size_t &cols)>> states;
	std::map<int, std::function<void(const std::string &label, const std::vector<float> &samples, const std::size_t &rows, const std::size_t &cols)>> weights;
	std::map<int, std::function<void(const std::vector<float> &position)>> predicted_position;
	std::map<int, std::function<void(const std::vector<float> &stimulus)>> predicted_stimulus;

	std::map<int, std::function<void(const std::vector<float> &elements, const std::size_t &rows, const std::size_t &cols, const std::vector<int> &offsets, const std::vector<int> &durations)> > scheduler;
	std::map<int, std::function<void(const std::size_t &rows, const  std::size_t &cols)>> feedforward;
	std::map<int, std::function<void(const std::size_t &rows, const  std::size_t &cols)>> feedback;
	std::map<int, std::function<void(const std::size_t &rows, const  std::size_t &cols)>> readout;
	std::map<int, std::function<void(const std::size_t &rows, const std::size_t &cols)>> recurrent;

	bool stopped;
	std::thread run_thread;


	Handle() :
		stopped(false)
	
	{

	}
};

