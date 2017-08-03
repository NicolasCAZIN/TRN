#pragma once

#include "Connection.h"

class TRN::Remote::Connection::Handle
{
public :
	std::map<int, std::function<void(const std::string &phase, const float &cycles_per_second)>> performances;
	std::map<int, std::function<void(const std::string &label, const std::vector<float> &samples, const std::size_t &rows, const std::size_t &cols)>> states;
	std::map<int, std::function<void(const std::string &label, const std::vector<float> &samples, const std::size_t &rows, const std::size_t &cols)>> weights;
	std::map<int, std::function<void(const std::vector<float> &prediction)>> loop;

	std::map<int, std::function<void(const std::vector<float> &incoming, const std::vector<float> &expected, const std::vector<float> &reward, const std::size_t &observations)>> scheduler;
	std::map<int, std::function<void(const std::size_t &rows, const  std::size_t &cols)>> feedforward;
	std::map<int, std::function<void(const std::size_t &rows, const  std::size_t &cols)>> feedback;
	std::map<int, std::function<void(const std::size_t &rows, const  std::size_t &cols)>> readout;
	std::map<int, std::function<void(const std::size_t &rows, const std::size_t &cols)>> recurrent;
};
