#include "stdafx.h"
#include "Mutator.h"

#include "Mutator/Shuffle.h"
#include "Mutator/Reverse.h"
#include "Mutator/Punch.h"
#include "Mutator/Custom.h"

std::shared_ptr<TRN::Core::Mutator> TRN::Model::Mutator::Shuffle::create()
{
	return TRN::Mutator::Shuffle::create();
}

std::shared_ptr<TRN::Core::Mutator> TRN::Model::Mutator::Reverse::create(const float &rate, const std::size_t &size)
{
	return TRN::Mutator::Reverse::create(rate, size);
}

std::shared_ptr<TRN::Core::Mutator> TRN::Model::Mutator::Punch::create(const float &rate, const std::size_t &size, const std::size_t &number)
{
	return TRN::Mutator::Punch::create(rate, size, number);
}

std::shared_ptr<TRN::Core::Mutator> TRN::Model::Mutator::Custom::create(const std::function<void(const std::vector<int> &offsets, const std::vector<int> &durations)> &request, std::function<void(const std::vector<int> &offsets, const std::vector<int> &durations)> &reply)
{
	return TRN::Mutator::Custom::create(request, reply);
}
