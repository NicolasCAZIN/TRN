#pragma once

#include "matfile_global.h"
#include "TRN4CPP/Callbacks.h"

class MATFILE_EXPORT Callbacks :public TRN4CPP::Plugin::Callbacks::Interface
{
private :
	class Handle;
	std::unique_ptr<Handle> handle;
private :
	void update();
	void save(const std::size_t &version, mxArray *deep_copy);

public:
	virtual void initialize(const std::map<std::string, std::string> &arguments) override;
	virtual void uninitialize() override;

	virtual	void callback_measurement_readout_raw(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &primed, const std::vector<float> &predicted, const std::vector<float> &expected, const std::size_t &preamble, const std::size_t &pages, const std::size_t &rows, const  std::size_t &cols) override;
	virtual void callback_measurement_position_raw(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &primed, const std::vector<float> &predicted, const std::vector<float> &expected, const std::size_t &preamble, const std::size_t &pages, const std::size_t &rows, const  std::size_t &cols) override;
	virtual void callback_measurement_readout_mean_square_error(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols) override;
	virtual void callback_measurement_readout_frechet_distance(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols) override;
	virtual void callback_measurement_position_mean_square_error(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols) override;
	virtual void callback_measurement_position_frechet_distance(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols) override;

	virtual void callback_performances(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::string &phase, const float &cycles_per_second, const float &gflops_per_second) override;
	virtual void callback_states(const unsigned long long &id, const std::string &phase, const std::string &label, const std::size_t &batch, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &samples, const std::size_t &rows, const std::size_t &cols) override;
	virtual void callback_weights(const unsigned long long &id, const std::string &phase, const std::string &label, const std::size_t &batch, const std::size_t &trial, const std::vector<float> &samples, const std::size_t &rows, const std::size_t &cols) override;
	virtual void callback_scheduling(const unsigned long long &id, const std::size_t &trial, const std::vector<int> &offsets, const std::vector<int> &durations) override;
};

extern "C" BOOST_SYMBOL_EXPORT Callbacks plugin_callbacks;
Callbacks plugin_callbacks;