#include "stdafx.h"
#include "Scheduler.h"


TRN::Core::Scheduler::Scheduler(const std::shared_ptr<TRN::Backend::Driver> &driver) :
	TRN::Helper::Bridge<TRN::Backend::Driver>(driver)
{
}

TRN::Core::Scheduler::~Scheduler()
{
}


