#include "stdafx.h"
#include "Measurement.h"
#include "Measurement/Custom.h"
#include "Measurement/MeanSquareError.h"
#include "Measurement/FrechetDistance.h"
#include "Measurement/Position.h"
#include "Measurement/Sequence.h"

std::shared_ptr<TRN::Core::Measurement::Implementation> MODEL_EXPORT TRN::Model::Measurement::MeanSquareError::create(const std::shared_ptr<TRN::Backend::Driver> &driver, const std::function<void(const unsigned long long &evaluation_id, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols)> &functor)
{
	return TRN::Measurement::MeanSquareError::create(driver, functor);
}

std::shared_ptr<TRN::Core::Measurement::Implementation> MODEL_EXPORT TRN::Model::Measurement::FrechetDistance::create(const std::shared_ptr<TRN::Backend::Driver> &driver, const std::string &norm, const std::string &aggregator, const std::function<void(const unsigned long long &evaluation_id, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols)> &functor)
{
	return TRN::Measurement::FrechetDistance::create(driver, norm, aggregator, functor);
}

std::shared_ptr<TRN::Core::Measurement::Implementation> MODEL_EXPORT TRN::Model::Measurement::Custom::create(const std::shared_ptr<TRN::Backend::Driver> &driver, const std::function<void(const unsigned long long &evaluation_id, const std::vector<float> &primed, const std::vector<float> &predicted, const std::vector<float> &expected, const std::size_t &preamble, const std::size_t &pages, const std::size_t &rows, const  std::size_t &cols)> &functor)
{
	return TRN::Measurement::Custom::create(driver, functor);
}

std::shared_ptr<TRN::Core::Measurement::Abstraction> MODEL_EXPORT TRN::Model::Measurement::Position::create(const std::shared_ptr<TRN::Core::Measurement::Implementation> &implementation, const std::size_t &batch_size)
{
	return TRN::Measurement::Position::create(implementation, batch_size);
}

std::shared_ptr<TRN::Core::Measurement::Abstraction> MODEL_EXPORT TRN::Model::Measurement::Sequence::create(const std::shared_ptr<TRN::Core::Measurement::Implementation> &implementation, const std::size_t &batch_size)
{
	return TRN::Measurement::Sequence::create(implementation, batch_size);
}
