#include "stdafx.h"
#include "Initializer.h"

TRN::Core::Initializer::Initializer(const std::shared_ptr<TRN::Backend::Driver> &driver) :
	TRN::Helper::Bridge<TRN::Backend::Driver>(driver)
{
}

TRN::Core::Initializer::~Initializer()
{
}