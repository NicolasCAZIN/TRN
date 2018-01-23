#include "stdafx.h"
#include "Simulator.h"


std::shared_ptr<TRN::Simulator::Basic> TRN::Model::Simulator::Basic::create(const std::function<void()> &trained, const std::function<void()> &primed, const std::function<void()> &tested)
{
	return TRN::Simulator::Basic::create(trained, primed, tested);
}



std::shared_ptr<TRN::Simulator::States>  TRN::Model::Simulator::States::create(const std::shared_ptr<TRN::Core::Simulator> &decorated,
	const std::function<void(const unsigned long long &evaluation_id, const std::string &phase, const std::string &label, const std::size_t &batch, const std::vector<float> &data, const std::size_t &rows, const std::size_t &cols)> &functor,
	const bool &train, const bool &prime, const bool &test)
{
	return TRN::Simulator::States::create(decorated, functor, train, prime, test);
}

std::shared_ptr<TRN::Simulator::Weights>  TRN::Model::Simulator::Weights::create(const std::shared_ptr<TRN::Core::Simulator> &decorated,
	const std::function<void(const unsigned long long &evaluation_id, const std::string &phase, const std::string &label,  const std::size_t &batch,const std::vector<float> &data, const std::size_t &rows, const std::size_t &cols)> &functor,
	const bool &train, const bool &initialization)
{
	return TRN::Simulator::Weights::create(decorated, functor, train, initialization);
}
std::shared_ptr<TRN::Simulator::Performances>  TRN::Model::Simulator::Performances::create(const std::shared_ptr<TRN::Core::Simulator> &decorated,
	const std::function<void(const unsigned long long &evaluation_id,  const std::string &phase, const float &cycles_per_second, const float &gflops_per_second)> &functor,
	const bool &train, const bool &prime, const bool &test)
{
	return TRN::Simulator::Performances::create(decorated, functor, train, prime, test);
}
std::shared_ptr<TRN::Simulator::Scheduling>  TRN::Model::Simulator::Scheduling::create(const std::shared_ptr<TRN::Core::Simulator> &decorated,
	const std::function<void(const unsigned long long &evaluation_id, const std::vector<int> &offsets, const std::vector<int> &durations)> &functor)
{
	return TRN::Simulator::Scheduling::create(decorated, functor);
}


