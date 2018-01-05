#pragma once

#include "Cache.h"



struct TRN::Engine::Cache::Handle 
{
	boost::interprocess::managed_windows_shared_memory segment; 
};
