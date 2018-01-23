#pragma once

#include "Frontend.h"

class TRN::Engine::Frontend::Handle
{
public:
	std::function<void()> on_completed;
	std::function<void(const unsigned long long &simulation_id, const std::size_t &counter, const bool &success, const std::string &cause)> on_ack;
	std::function<void(const int &rank, const std::string &host, const unsigned int &index, const std::string &name)> on_processor;
	std::function<void(const unsigned long long &simulation_id, const int &rank)> on_allocated;
	std::function<void(const unsigned long long &simulation_id, const int &rank)> on_deallocated;
	std::function<void(const int &rank)> on_quit;
	std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id)> on_trained;
	std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id)> on_primed;
	std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id)> on_tested;
	std::function<void(const unsigned long long &simulation_id)> on_configured;
	std::function<void(const std::string &message) > on_error;
	std::function<void(const std::string &message) > on_information;
	std::function<void(const std::string &message) > on_warning;

	std::map<unsigned long long, std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols)>> on_measurement_readout_mean_square_error;
	std::map<unsigned long long, std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols)>> on_measurement_readout_frechet_distance;
	std::map<unsigned long long, std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &primed, const std::vector<float> &predicted, const std::vector<float> &expected, const std::size_t &preamble, const std::size_t &pages, const std::size_t &rows, const  std::size_t &cols)>> on_measurement_readout_custom;
	std::map<unsigned long long, std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols)>> on_measurement_position_mean_square_error;
	std::map<unsigned long long, std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols)>> on_measurement_position_frechet_distance;
	std::map<unsigned long long, std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &primed, const std::vector<float> &predicted, const std::vector<float> &expected, const std::size_t &preamble, const std::size_t &pages, const std::size_t &rows, const  std::size_t &cols)>> on_measurement_position_custom;
	std::map<unsigned long long, std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::string &phase, const float &cycles_per_second, const float &gflops_per_second)>> on_performances;
	std::map<unsigned long long, std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::string &phase, const std::string &label, const std::size_t &batch, const std::vector<float> &samples, const std::size_t &rows, const std::size_t &cols)>> on_states;
	std::map<unsigned long long, std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::string &phase, const std::string &label, const std::size_t &batch,  const std::vector<float> &samples, const std::size_t &rows, const std::size_t &cols)>> on_weights;
	std::map<unsigned long long, std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols)>> on_position;
	std::map<unsigned long long, std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &stimulus, const std::size_t &rows, const std::size_t &cols)>> on_stimulus;

	std::map<unsigned long long, std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const unsigned long &seed, const std::vector<int> &offsets, const std::vector<int> &durations)>> on_mutator;
	std::map<unsigned long long, std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const unsigned long &seed, const std::vector<float> &elements, const std::size_t &rows, const std::size_t &cols, const std::vector<int> &offsets, const std::vector<int> &durations)>> on_scheduler;
	std::map<unsigned long long, std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<int> &offsets, const std::vector<int> &durations)>> on_scheduling;

	std::map<unsigned long long, std::function<void(const unsigned long long &simulation_id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const  std::size_t &cols)>> on_feedforward;
	std::map<unsigned long long, std::function<void(const unsigned long long &simulation_id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const  std::size_t &cols)>> on_feedback;
	std::map<unsigned long long, std::function<void(const unsigned long long &simulation_id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const  std::size_t &cols)>> on_readout;
	std::map<unsigned long long, std::function<void(const unsigned long long &simulation_id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> >on_recurrent;
};
