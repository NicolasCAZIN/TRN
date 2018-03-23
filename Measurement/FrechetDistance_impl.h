#pragma once

#include "FrechetDistance.h"



class TRN::Measurement::FrechetDistance::Handle
{
public:
	std::function<void(const unsigned long long &evaluation_id, const std::vector<float> &values, const std::size_t &rows, const std::size_t &cols)> functor;

	std::function<void(const std::size_t &batch_c_1, const std::size_t &batch_c_2,
		const std::size_t &stride_c_1, const std::size_t &stride_c_2,
		const std::vector<float> &c_1, const std::vector<float> &c_2,
		const std::size_t &n_1,
		const std::size_t &n_2,
		const std::size_t &n_d,
		std::vector<float> &dfd)> compute_frechet_distance;



};
