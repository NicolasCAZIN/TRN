#include "stdafx.h"
#include "Helper/Logger.h"
#include "Cache.h"

BOOLEAN WINAPI DllMain(IN HINSTANCE hDllHandle, IN DWORD     nReason, IN LPVOID    Reserved)
{
	BOOLEAN bSuccess = TRUE;

	try
	{
		switch (nReason)
		{
			case DLL_PROCESS_ATTACH:
				TRN::Engine::Cache::initialize();
				break;

			case DLL_PROCESS_DETACH:
				TRN::Engine::Cache::uninitialize();
				break;
		}
	}
	catch (std::exception &e)
	{
		bSuccess = false;
		ERROR_LOGGER << e.what();
	}
	return bSuccess;
}