#include "stdafx.h"
#include "TRN/Factory.h"
#include "Builder.h"

BOOL APIENTRY DllMain(HMODULE hModule, DWORD  ul_reason_for_call, LPVOID lpReserved)
{
	switch (ul_reason_for_call)
	{
	case DLL_PROCESS_ATTACH:
		
		break;
	case DLL_THREAD_ATTACH:
		printf("DllMain, DLL_THREAD_ATTACH\n");
		break;
	case DLL_THREAD_DETACH:
		printf("DllMain, DLL_THREAD_DETACH\n");
		break;
	case DLL_PROCESS_DETACH:
		printf("DllMain, DLL_PROCESS_DETACH\n");
		break;
	default:
		printf("DllMain, ????\n");
		break;
	}
	return TRUE;
}