#include "stdafx.h"

#include "Convert.h"

#include "TRN4JAVA_Simplified.h"
#include "TRN4JAVA_Simplified_Simulation.h"

#include "TRN4CPP/Simplified.h"
#include "TRN4CPP/Sequences.h"
#include "Helper/Logger.h"

void Java_TRN4JAVA_Simplified_00024Simulation_compute(JNIEnv *env, jclass jclazz, jstring scenario_filename)
{
	TRACE_LOGGER;
	try
	{
		TRN4CPP::Simulation::compute(TRN4JAVA::Convert::to_string(env, scenario_filename));
	}
	catch (std::exception &e)
	{
		ERROR_LOGGER << e.what(); env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
	}
}
