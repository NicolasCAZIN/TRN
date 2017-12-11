#include "stdafx.h"

#include "Convert.h"

#include "TRN4JAVA_Simplified.h"
#include "TRN4JAVA_Simplified_Plugin.h"
#include "TRN4JAVA_Simplified_Simulation.h"

#include "TRN4CPP/Simplified.h"
#include "TRN4CPP/Sequences.h"
#include "Helper/Logger.h"

void Java_TRN4JAVA_Simplified_00024Plugin_initialize(JNIEnv *env, jclass jclazz, jstring library_path, jstring name, jobject arguments)
{
	TRACE_LOGGER;
	try
	{
		TRN4CPP::Plugin::Sequences::initialize(TRN4JAVA::Convert::to_string(env, library_path), TRN4JAVA::Convert::to_string(env, name), TRN4JAVA::Convert::to_map(env, arguments));
	}
	catch (std::exception &e)
	{
		ERROR_LOGGER << e.what(); env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
	}
}
void Java_TRN4JAVA_Simplified_00024Simulation_declare(JNIEnv *env, jclass jclazz, jstring label, jfloatArray elements, jlong rows, jlong cols, jstring tag)
{
	TRACE_LOGGER;
	try
	{
		TRN4CPP::Sequences::declare(TRN4JAVA::Convert::to_string(env, label), TRN4JAVA::Convert::to_string(env, tag), TRN4JAVA::Convert::to_float_vector(env, elements), (std::size_t)rows, (std::size_t)cols);
	}
	catch (std::exception &e)
	{
		ERROR_LOGGER << e.what(); env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
	}
}
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
