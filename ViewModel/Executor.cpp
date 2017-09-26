#include "stdafx.h"
#include "Executor.h"
#include "Engine/Blocking.h"
#include "Engine/NonBlocking.h"

std::shared_ptr<TRN::Engine::Executor> VIEWMODEL_EXPORT TRN::ViewModel::Executor::Blocking::create()
{
	return TRN::Engine::Blocking::create();
}

std::shared_ptr<TRN::Engine::Executor> VIEWMODEL_EXPORT TRN::ViewModel::Executor::NonBlocking::create()
{
	return TRN::Engine::NonBlocking::create();
}
