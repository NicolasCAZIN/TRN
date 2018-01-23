#include "stdafx.h"
#include "Mutator.h"

#include "Mutator/Shuffle.h"
#include "Mutator/Reverse.h"
#include "Mutator/Punch.h"
#include "Mutator/Custom.h"

std::shared_ptr<TRN::Core::Mutator> TRN::Model::Mutator::Shuffle::create(const unsigned long &seed)
{
	return TRN::Mutator::Shuffle::create(seed);
}

std::shared_ptr<TRN::Core::Mutator> TRN::Model::Mutator::Reverse::create(const unsigned long &seed,const float &rate, const std::size_t &size)
{
	return TRN::Mutator::Reverse::create(seed, rate, size);
}

std::shared_ptr<TRN::Core::Mutator> TRN::Model::Mutator::Punch::create(const unsigned long &seed,const float &rate, const std::size_t &size, const std::size_t &counter)
{
	return TRN::Mutator::Punch::create(seed, rate, size, counter);
}

std::shared_ptr<TRN::Core::Mutator> TRN::Model::Mutator::Custom::create(const unsigned long &seed,const std::function<void(const unsigned long long &evaluation_id, const unsigned long &seed, const std::vector<int> &offsets, const std::vector<int> &durations)> &request, std::function<void(const std::size_t &trial, const std::vector<int> &offsets, const std::vector<int> &durations)> &reply)
{
	return TRN::Mutator::Custom::create(seed, request, reply);
}
