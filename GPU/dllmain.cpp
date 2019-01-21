#include "stdafx.h"
//#include "Basic.h"
#include "Helper/Logger.h"

BOOLEAN WINAPI DllMain(IN HINSTANCE hDllHandle, IN DWORD     nReason, IN LPVOID    Reserved)
{
	BOOLEAN bSuccess = TRUE;

	try
	{
		switch (nReason)
		{
			case DLL_PROCESS_ATTACH:
				//TRN4CPP::Engine::initialize();
				break;

			case DLL_PROCESS_DETACH:
				//TRN4CPP::Engine::uninitialize();
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