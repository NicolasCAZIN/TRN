#include "stdafx.h"
#include "Initializer.h"

const bool TRN::Core::Initializer::DEFAULT_BLANK_DIAGONAL = false;

TRN::Core::Initializer::Initializer(const std::shared_ptr<TRN::Backend::Driver> &driver) :
	TRN::Helper::Bridge<TRN::Backend::Driver>(driver)
{
}

TRN::Core::Initializer::~Initializer()
{
}