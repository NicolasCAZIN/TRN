#include "stdafx.h"
#include "Driver_impl.h"



TRN::Backend::Driver::Driver(const std::shared_ptr<TRN::Backend::Memory> &memory, const std::shared_ptr<TRN::Backend::Random> &random, const std::shared_ptr<TRN::Backend::Algorithm> &algorithm) :
	handle(std::make_unique<Handle>())
{
	handle->memory = memory;
	handle->random = random;
	handle->algorithm = algorithm;
}
TRN::Backend::Driver::~Driver()
{
	handle->memory.reset();
	handle->random.reset();
	handle->algorithm.reset();
	handle.reset();
}

const std::shared_ptr<TRN::Backend::Memory> &TRN::Backend::Driver::get_memory()
{
	return handle->memory;
}
const std::shared_ptr<TRN::Backend::Random> &TRN::Backend::Driver::get_random()
{
	return handle->random;
}
const std::shared_ptr<TRN::Backend::Algorithm> &TRN::Backend::Driver::get_algorithm()
{
	return handle->algorithm;
}

