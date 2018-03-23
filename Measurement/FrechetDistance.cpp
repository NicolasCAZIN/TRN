#include "stdafx.h"
#include "FrechetDistance_impl.h"
#include "Helper/Logger.h"


enum Aggregator
{
	Max,
	Sum,
	Avg
};
enum Norm
{
	LINF = -1,
	L1 = 1,
	L2 = 2
};


void operator >> (const std::string &str, Aggregator &aggregator)
{
	if (str.empty())
	{
		WARNING_LOGGER << "Frechet distance aggregator unspecified. Using MAX";
		aggregator = Aggregator::Sum;
	}
	else
	{
		auto upper_str = boost::to_upper_copy(str);
		if (upper_str == "MAX")
			aggregator = Aggregator::Max;
		else if (upper_str == "SUM")
			aggregator = Aggregator::Sum;
		else if (upper_str == "AVG")
			aggregator = Aggregator::Avg;
		else
			throw std::invalid_argument("Unexpected aggregator type " + str);
	}
}
void operator >> (const std::string &str, Norm &norm)
{
	if (str.empty())
	{
		WARNING_LOGGER << "Frechet distance norm unspecified. Using L2 norm";
		norm = Norm::L2;
	}
	else
	{
		auto upper_str = boost::to_upper_copy(str);

		if (upper_str == "1")
			norm = Norm::L1;
		else if (upper_str == "2")
			norm = Norm::L2;
		else if (upper_str == "INF")
			norm = Norm::LINF;
		else
			throw std::invalid_argument("Unexpected norm type " + str);
	}
}

template <Aggregator aggregator>
static inline float finish(const float &frechet_distance, const int &count)
{
	return frechet_distance;
}
template <>
static inline float finish<Aggregator::Avg>(const float &frechet_distance, const int &count)
{
	return frechet_distance / (float)count;
}

template <Aggregator aggregator>
static inline float aggregate(const float &d1, const float &d2)
{
	return d1 + d2;
}
template <>
static inline float aggregate<Aggregator::Max>(const float &d1, const float &d2)
{
	return std::fmaxf(d1, d2);
}



template<Norm norm>
static inline float compute_norm(const float *c_1, const float *c_2, const std::size_t &n_d)
{
}
template<>
static inline float compute_norm<Norm::L1>(const float *c_1, const float *c_2, const std::size_t &n_d)
{
	float sum = 0.0f;
	for (std::size_t k = 0; k < n_d; k++)
	{
		float d = c_1[k] - c_2[k];
		sum += std::fabsf(d);
	}
	return sum;
}
template<>
static inline float compute_norm<Norm::L2>(const float *c_1, const float *c_2, const std::size_t &n_d)
{
	float sum = 0.0f;
	for (std::size_t k = 0; k < n_d; k++)
	{
		float d = c_1[k] - c_2[k];
		sum += d * d;
	}
	return std::sqrtf(sum);
}
template<>
static inline float compute_norm<Norm::LINF>(const float *c_1, const float *c_2, const std::size_t &n_d)
{
	float m = 0.0f;
	for (std::size_t k = 0; k < n_d; k++)
	{
		float d = c_1[k] - c_2[k];
		m = std::fmaxf(std::fabsf(d), m);
	}
	return m;
}




template<Norm norm = Norm::L2>
static inline void compute_norm(const std::vector<float> &c_1, const std::vector<float> &c_2, const std::size_t &n_1, const std::size_t &n_2, const std::size_t &n_d, std::vector<float> &dist)
{
	for (std::size_t i = 0; i < n_1; i++)
	{
		for (std::size_t  j = 0; j < n_2; j++)
		{
			dist[i * n_2 + j] = compute_norm<norm>(&c_1[i * n_d], &c_2[j * n_d], n_d);
		}
	}
}


static inline float evaluate_nrm(const std::vector<float> &dist, const std::size_t &n_1, const std::size_t &n_2, const std::size_t &n_d, const std::size_t &i, const std::size_t &j)
{
	return dist[(i)* n_2 + (j)];
}

template <Aggregator aggregator>
static inline float recursive_c(const std::vector<float> &dist, const std::size_t &n_1, const std::size_t &n_2, const std::size_t &n_d, const std::size_t i, const std::size_t j, std::vector<float> &ca, int &count)
{
	float *ca_ij; /* Pointer to `ca(i, j)`, just to simplify notation */

				  /*
				  * Target the shortcut to the (i, j)-th entry of the matrix `ca`
				  *
				  * Once again, notice the 1-offset.
				  */
	ca_ij = &ca[(i)*n_2 + (j)];

	/* This implements the algorithm from [1] */
	if (*ca_ij > -1.0f)
	{

		return *ca_ij;
	}
	else if ((i == 0) && (j == 0))
	{
		*ca_ij = evaluate_nrm(dist, n_1, n_2, n_d, 0, 0);
	}
	else if ((i > 0) && (j == 0))
	{
		*ca_ij = aggregate<aggregator>(recursive_c<aggregator>(dist, n_1, n_2, n_d, i - 1, 0, ca, count), evaluate_nrm(dist, n_1, n_2, n_d, i, 0));
		count++;
	}
	else if ((i == 0) && (j > 0))
	{
		*ca_ij = aggregate<aggregator>(recursive_c<aggregator>(dist, n_1, n_2, n_d, 0, j - 1, ca, count), evaluate_nrm(dist, n_1, n_2, n_d, 0, j));
		count++;
	}
	else if ((i > 0) && (j > 0))
	{
		*ca_ij = aggregate<aggregator>(
			std::fminf(std::fminf(
				recursive_c<aggregator>(dist, n_1, n_2, n_d, i - 1, j, ca, count),
				recursive_c<aggregator>(dist, n_1, n_2, n_d, i - 1, j - 1, ca, count)),
				recursive_c<aggregator>(dist, n_1, n_2, n_d, i, j - 1, ca, count)),
			evaluate_nrm(dist, n_1, n_2, n_d, i, j));
		count++;
	}
	else
	{

		*ca_ij = -std::numeric_limits<float>::infinity();
	}

	return *ca_ij;
}


template <Aggregator aggregator = Aggregator::Max>
static float discrete_frechet_distance(const std::vector<float> &dist, const std::size_t &n_1, const std::size_t &n_2, const std::size_t &n_d, std::vector<float> &ca)
{
	int count = 1;
	float fd = recursive_c<aggregator>(dist, n_1, n_2, n_d, n_1 - 1, n_2 - 1, ca, count);
	assert(count == ca.size());
	return finish<aggregator>(fd, count);
}

template <Norm norm = Norm::L2, Aggregator aggregator = Aggregator::Max>
static inline void compute_frechet_distance(
const std::size_t &batch_c_1, const std::size_t &batch_c_2,
		const std::size_t &stride_c_1, const std::size_t &stride_c_2,
		const std::vector<float> &c_1, const std::vector<float> &c_2,
		const std::size_t &n_1,
		const std::size_t &n_2,
		const std::size_t &n_d,
		std::vector<float> &dfd
)
{
	const auto K = batch_c_1 * batch_c_2;
	const auto size = n_1 * n_2;

	std::vector<std::pair<std::size_t, std::size_t>> pairs;
	auto symetrical = (c_1.data() == c_2.data());
	if (symetrical)
	{
		for (std::size_t b1 = 0; b1 < batch_c_1; b1++)
		{
			for (std::size_t b2 = b1 + 1; b2 < batch_c_2; b2++)
			{
				pairs.push_back(std::make_pair(b1, b2));
			}
		}
	}
	else
	{
		for (std::size_t b1 = 0; b1 < batch_c_1; b1++)
		{
			for (std::size_t b2 = 0; b2 < batch_c_2; b2++)
			{
				pairs.push_back(std::make_pair(b1, b2));
			}
		}
	}


#pragma omp parallel for
	for (int k = 0; k < pairs.size(); k++)
	{
		auto b1 = pairs[k].first;
		auto b2 = pairs[k].second;

		const float *c1b1_ptr = &c_1[b1 * stride_c_1];
		const float *c2b2_ptr = &c_2[b2 * stride_c_2];
		std::vector<float> c1b1(c1b1_ptr, c1b1_ptr + n_1 * n_d);
		std::vector<float> c2b2(c2b2_ptr, c2b2_ptr + n_2 * n_d);
		assert(c1b1.size() == n_1 * n_d);
		assert(c2b2.size() == n_2 * n_d);
		std::vector<float> dist(size, 0.0f);

		compute_norm<norm>(c1b1, c2b2, n_1, n_2, n_d, dist);

		std::vector<float> ca(size, -1.0f);
		auto d = discrete_frechet_distance<aggregator>(dist, n_1, n_2, n_d, ca);
		
		//cv::Mat penality(n_1, n_2, CV_32F, ca.data());
		dfd[b2 + batch_c_2 * b1] = d;
		if (symetrical)
			dfd[b1 + batch_c_2 * b2] = d;
	}
}

TRN::Measurement::FrechetDistance::FrechetDistance(const std::shared_ptr<TRN::Backend::Driver> &driver, const std::string &norm, const std::string &aggregator,
	const std::function<void(const unsigned long long &evaluation_id, const std::vector<float> &values, const std::size_t &rows, const std::size_t &cols)> &functor) :
	TRN::Core::Measurement::Implementation(driver),
	handle(std::make_unique<Handle>())
{
	handle->functor = functor;
	Norm _norm;
	Aggregator _aggregator;

	norm >> _norm;
	aggregator >> _aggregator;
	switch (_norm)
	{
	case Norm::L1:
		switch (_aggregator)
		{
		case Aggregator::Max:
			handle->compute_frechet_distance = compute_frechet_distance<Norm::L1, Aggregator::Max>;
			break;
		case Aggregator::Sum:
			handle->compute_frechet_distance = compute_frechet_distance<Norm::L1, Aggregator::Sum>;
			break;
		case Aggregator::Avg:
			handle->compute_frechet_distance = compute_frechet_distance<Norm::L1, Aggregator::Avg>;
			break;
		}
		break;
	case Norm::L2:
		switch (_aggregator)
		{
		case Aggregator::Max:
			handle->compute_frechet_distance = compute_frechet_distance<Norm::L2, Aggregator::Max>;
			break;
		case Aggregator::Sum:
			handle->compute_frechet_distance = compute_frechet_distance<Norm::L2, Aggregator::Sum>;
			break;
		case Aggregator::Avg:
			handle->compute_frechet_distance = compute_frechet_distance<Norm::L2, Aggregator::Avg>;
			break;
		}
		break;
	case Norm::LINF:
		switch (_aggregator)
		{
		case Aggregator::Max:
			handle->compute_frechet_distance = compute_frechet_distance<Norm::LINF, Aggregator::Max>;
			break;
		case Aggregator::Sum:
			handle->compute_frechet_distance = compute_frechet_distance<Norm::LINF, Aggregator::Sum>;
			break;
		case Aggregator::Avg:
			handle->compute_frechet_distance = compute_frechet_distance<Norm::LINF, Aggregator::Avg>;
			break;
		}
		break;
	}

}



void TRN::Measurement::FrechetDistance::compute(const unsigned long long &evaluation_id,
	const std::shared_ptr<TRN::Core::Matrix> &primed, 
	const std::shared_ptr<TRN::Core::Batch> &predicted, 
	const std::shared_ptr<TRN::Core::Matrix> &expected, 
	const std::shared_ptr<TRN::Core::Matrix> &error)
{
	DEBUG_LOGGER << "Using non vectorized fallback";
	std::vector<float> expected_values;
	std::size_t expected_rows;
	std::size_t expected_cols;
	expected->to(expected_values, expected_rows, expected_cols);

	std::vector<float> predicted_values;
	std::vector<std::size_t> predicted_rows;
	std::vector<std::size_t> predicted_cols;
	std::size_t batch_size;
	predicted->to(predicted_values, batch_size, predicted_rows, predicted_cols);

	std::vector<float> error_values;
	std::size_t error_rows;
	std::size_t error_cols;
	error->to(error_values, error_rows, error_cols);

	auto batch_c1 = 1;
	auto batch_c2 = batch_size;
	auto n_1 = expected_rows;
	auto n_2 = predicted_rows[0];
	auto n_d = expected_cols;

	auto stride_c1 = n_1 * n_d;
	auto stride_c2 = n_2 * n_d;
	auto c_1 = expected_values;
	auto c_2 = predicted_values;

	handle->compute_frechet_distance(batch_c1, batch_c2, stride_c1, stride_c2, c_1, c_2, n_1, n_2, n_d, error_values);

	handle->functor(evaluation_id, error_values, error_rows, error_cols);
}

std::shared_ptr<TRN::Measurement::FrechetDistance> TRN::Measurement::FrechetDistance::create(const std::shared_ptr<TRN::Backend::Driver> &driver, const std::string &norm, const std::string &aggregator, const std::function<void(const unsigned long long &evaluation_id, const std::vector<float> &values, const std::size_t &rows, const std::size_t &cols)> &functor)
{
	return std::make_shared<TRN::Measurement::FrechetDistance>(driver, norm, aggregator, functor);
}