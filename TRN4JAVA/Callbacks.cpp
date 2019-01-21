#include "stdafx.h"

#include "Functor.h"
#include "Convert.h"

#include "TRN4JAVA_Callbacks.h"
#include "TRN4JAVA_Callbacks_Plugin.h"
#include "TRN4JAVA_Callbacks_Simulation.h"
#include "TRN4JAVA_Callbacks_Simulation_Measurement.h"
#include "TRN4JAVA_Callbacks_Simulation_Measurement_Position.h"
#include "TRN4JAVA_Callbacks_Simulation_Measurement_Position_FrechetDistance.h"
#include "TRN4JAVA_Callbacks_Simulation_Measurement_Position_MeanSquareError.h"
#include "TRN4JAVA_Callbacks_Simulation_Measurement_Position_Raw.h"
#include "TRN4JAVA_Callbacks_Simulation_Measurement_Processed.h"
#include "TRN4JAVA_Callbacks_Simulation_Measurement_Raw.h"
#include "TRN4JAVA_Callbacks_Simulation_Measurement_Readout.h"
#include "TRN4JAVA_Callbacks_Simulation_Measurement_Readout_FrechetDistance.h"
#include "TRN4JAVA_Callbacks_Simulation_Measurement_Readout_MeanSquareError.h"
#include "TRN4JAVA_Callbacks_Simulation_Measurement_Readout_Raw.h"
#include "TRN4JAVA_Callbacks_Simulation_Recording.h"
#include "TRN4JAVA_Callbacks_Simulation_Recording_Performances.h" 
#include "TRN4JAVA_Callbacks_Simulation_Recording_Scheduling.h"
#include "TRN4JAVA_Callbacks_Simulation_Recording_States.h"
#include "TRN4JAVA_Callbacks_Simulation_Recording_Weights.h"

#include "TRN4CPP/Callbacks.h"

void Java_TRN4JAVA_Callbacks_00024Plugin_initialize(JNIEnv *env, jclass jclazz, jstring library_path, jstring name, jobject arguments)
{
	TRACE_LOGGER;
	try
	{
		TRN4CPP::Plugin::Callbacks::initialize(TRN4JAVA::Convert::to_string(env, library_path), TRN4JAVA::Convert::to_string(env, name), TRN4JAVA::Convert::to_map(env, arguments));
	}
	catch (std::exception &e)
	{
		ERROR_LOGGER << e.what(); env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
	}
}
void Java_TRN4JAVA_Callbacks_00024Simulation_00024Measurement_00024Position_00024FrechetDistance_install(JNIEnv *env, jclass jclazz, jobject frechet_distance)
{
	TRACE_LOGGER;
	TRN4JAVA::Functor::install(env, frechet_distance, TRN4JAVA::Functor::PROCESSED_CALLBACK_SIGNATURE, TRN4JAVA::Functor::processed_global_ref, TRN4CPP::Simulation::Measurement::Position::FrechetDistance::install, TRN4JAVA::Functor::make_function(TRN4JAVA::Functor::measurement_matrix_callback));
}
void Java_TRN4JAVA_Callbacks_00024Simulation_00024Measurement_00024Position_00024MeanSquareError_install(JNIEnv *env, jclass jclazz, jobject mean_square_error)
{
	TRACE_LOGGER;
	TRN4JAVA::Functor::install(env, mean_square_error, TRN4JAVA::Functor::PROCESSED_CALLBACK_SIGNATURE, TRN4JAVA::Functor::processed_global_ref, TRN4CPP::Simulation::Measurement::Position::MeanSquareError::install, TRN4JAVA::Functor::make_function(TRN4JAVA::Functor::measurement_matrix_callback));
}
void Java_TRN4JAVA_Callbacks_00024Simulation_00024Measurement_00024Position_00024Raw_install(JNIEnv *env, jclass jclazz, jobject raw)
{
	TRACE_LOGGER;
	TRN4JAVA::Functor::install(env, raw, TRN4JAVA::Functor::RAW_CALLBACK_SIGNATURE, TRN4JAVA::Functor::raw_global_ref, TRN4CPP::Simulation::Measurement::Position::Raw::install, TRN4JAVA::Functor::make_function(TRN4JAVA::Functor::measurement_raw_callback));
}
void Java_TRN4JAVA_Callbacks_00024Simulation_00024Measurement_00024Readout_00024FrechetDistance_install(JNIEnv *env, jclass jclazz, jobject frechet_distance)
{
	TRACE_LOGGER;
	TRN4JAVA::Functor::install(env, frechet_distance, TRN4JAVA::Functor::PROCESSED_CALLBACK_SIGNATURE, TRN4JAVA::Functor::processed_global_ref, TRN4CPP::Simulation::Measurement::Readout::FrechetDistance::install, TRN4JAVA::Functor::make_function(TRN4JAVA::Functor::measurement_matrix_callback));
}
void Java_TRN4JAVA_Callbacks_00024Simulation_00024Measurement_00024Readout_00024MeanSquareError_install(JNIEnv *env, jclass jclazz, jobject mean_square_error)
{
	TRACE_LOGGER;
	TRN4JAVA::Functor::install(env, mean_square_error, TRN4JAVA::Functor::PROCESSED_CALLBACK_SIGNATURE, TRN4JAVA::Functor::processed_global_ref, TRN4CPP::Simulation::Measurement::Readout::MeanSquareError::install, TRN4JAVA::Functor::make_function(TRN4JAVA::Functor::measurement_matrix_callback));
}
void Java_TRN4JAVA_Callbacks_00024Simulation_00024Measurement_00024Readout_00024Raw_install(JNIEnv *env, jclass jclazz, jobject raw)
{
	TRACE_LOGGER;
	TRN4JAVA::Functor::install(env, raw, TRN4JAVA::Functor::RAW_CALLBACK_SIGNATURE, TRN4JAVA::Functor::raw_global_ref, TRN4CPP::Simulation::Measurement::Readout::Raw::install, TRN4JAVA::Functor::make_function(TRN4JAVA::Functor::measurement_raw_callback));
}
void Java_TRN4JAVA_Callbacks_00024Simulation_00024Recording_00024Performances_install(JNIEnv *env, jclass jclazz, jobject performances)
{
	TRACE_LOGGER;
	TRN4JAVA::Functor::install(env, performances, TRN4JAVA::Functor::PERFORMANCES_CALLBACK_SIGNATURE, TRN4JAVA::Functor::recording_global_ref, TRN4CPP::Simulation::Recording::Performances::install, TRN4JAVA::Functor::make_function(TRN4JAVA::Functor::recording_performances_callback));
}
void Java_TRN4JAVA_Callbacks_00024Simulation_00024Recording_00024Scheduling_install(JNIEnv *env, jclass jclazz, jobject scheduling)
{
	TRACE_LOGGER;
	TRN4JAVA::Functor::install(env, scheduling, TRN4JAVA::Functor::SCHEDULING_CALLBACK_SIGNATURE, TRN4JAVA::Functor::recording_global_ref, TRN4CPP::Simulation::Recording::Scheduling::install, TRN4JAVA::Functor::make_function(TRN4JAVA::Functor::recording_scheduling_callback));
}
void Java_TRN4JAVA_Callbacks_00024Simulation_00024Recording_00024States_install(JNIEnv *env, jclass jclazz, jobject states)
{
	TRACE_LOGGER;
	TRN4JAVA::Functor::install(env, states, TRN4JAVA::Functor::STATES_CALLBACK_SIGNATURE, TRN4JAVA::Functor::recording_global_ref, TRN4CPP::Simulation::Recording::States::install, TRN4JAVA::Functor::make_function(TRN4JAVA::Functor::recording_states_callback));
}
void Java_TRN4JAVA_Callbacks_00024Simulation_00024Recording_00024Weights_install(JNIEnv *env, jclass jclazz, jobject weights)
{
	TRACE_LOGGER;
	TRN4JAVA::Functor::install(env, weights, TRN4JAVA::Functor::WEIGHTS_CALLBACK_SIGNATURE, TRN4JAVA::Functor::recording_global_ref, TRN4CPP::Simulation::Recording::Weights::install, TRN4JAVA::Functor::make_function(TRN4JAVA::Functor::recording_weights_callback));
}