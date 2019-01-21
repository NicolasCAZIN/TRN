#include "stdafx.h"

#include "Functor.h"
#include "Convert.h"

#include "TRN4JAVA_Custom.h"
#include "TRN4JAVA_Custom_Plugin.h"
#include "TRN4JAVA_Custom_Simulation.h"
#include "TRN4JAVA_Custom_Simulation_Loop.h"
#include "TRN4JAVA_Custom_Simulation_Encoder.h"
#include "TRN4JAVA_Custom_Simulation_Reservoir.h"
#include "TRN4JAVA_Custom_Simulation_Reservoir_Feedforward.h"
#include "TRN4JAVA_Custom_Simulation_Reservoir_Readout.h"
#include "TRN4JAVA_Custom_Simulation_Reservoir_Recurrent.h"
#include "TRN4JAVA_Custom_Simulation_Reservoir_Weights.h"
#include "TRN4JAVA_Custom_Simulation_Scheduler.h"
#include "TRN4JAVA_Custom_Simulation_Scheduler_Mutator.h"

#include "TRN4CPP/Custom.h"
#include <set>

void Java_TRN4JAVA_Custom_00024Plugin_initialize(JNIEnv *env, jclass jclazz, jstring library_path, jstring name, jobject arguments)
{
	TRACE_LOGGER;
	try
	{
		TRN4CPP::Plugin::Custom::initialize(TRN4JAVA::Convert::to_string(env, library_path), TRN4JAVA::Convert::to_string(env, name), TRN4JAVA::Convert::to_map(env, arguments));
	}
	catch (std::exception &e)
	{
		ERROR_LOGGER << e.what(); env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
	}
}
void Java_TRN4JAVA_Custom_00024Simulation_00024Encoder_notify(JNIEnv *env, jobject encoder, jlong simulation_id, jlong evaluation_id, jfloatArray estimated_position, jfloatArray perceived_stimulus, jlong rows, jlong cols)
{
	TRN4JAVA::Functor::notify(env, encoder, (unsigned long long)simulation_id, TRN4JAVA::Functor::encoder_global_ref, TRN4JAVA::Functor::encoder_reply, (unsigned long long)simulation_id, (unsigned long long)evaluation_id, TRN4JAVA::Convert::to_float_vector(env, estimated_position),  (std::size_t)rows, (std::size_t)2);
	TRN4JAVA::Functor::notify(env, encoder, (unsigned long long)simulation_id, TRN4JAVA::Functor::loop_global_ref, TRN4JAVA::Functor::loop_reply, (unsigned long long)simulation_id, (unsigned long long)evaluation_id, TRN4JAVA::Convert::to_float_vector(env, perceived_stimulus), (std::size_t)rows, (std::size_t)cols);
}
void Java_TRN4JAVA_Custom_00024Simulation_00024Encoder_install(JNIEnv *env, jclass clazz, jobject encoder) 
{
	TRACE_LOGGER;
	TRN4JAVA::Functor::install(env, encoder, TRN4JAVA::Functor::ENCODER_CALLBACK_SIGNATURE, TRN4JAVA::Functor::encoder_global_ref, TRN4CPP::Simulation::Encoder::Custom::install, TRN4JAVA::Functor::make_function(TRN4JAVA::Functor::measurement_matrix_callback), TRN4JAVA::Functor::encoder_reply);
	TRN4JAVA::Functor::install(env, encoder, TRN4JAVA::Functor::LOOP_CALLBACK_SIGNATURE, TRN4JAVA::Functor::loop_global_ref, TRN4CPP::Simulation::Loop::Custom::install, TRN4JAVA::Functor::make_function(TRN4JAVA::Functor::measurement_matrix_callback), TRN4JAVA::Functor::loop_reply);
}
void Java_TRN4JAVA_Custom_00024Simulation_00024Loop_notify(JNIEnv *env, jobject loop, jlong simulation_id, jlong evaluation_id, jfloatArray elements, jlong rows, jlong cols)
{
	TRN4JAVA::Functor::notify(env, loop, (unsigned long long)simulation_id,TRN4JAVA::Functor::loop_global_ref, TRN4JAVA::Functor::loop_reply, (unsigned long long)simulation_id, (unsigned long long)evaluation_id, TRN4JAVA::Convert::to_float_vector(env, elements), (std::size_t)rows, (std::size_t)cols);
}
void Java_TRN4JAVA_Custom_00024Simulation_00024Loop_install(JNIEnv *env, jclass clazz, jobject loop)
{
	TRN4JAVA::Functor::install(env, loop, TRN4JAVA::Functor::LOOP_CALLBACK_SIGNATURE, TRN4JAVA::Functor::loop_global_ref, TRN4CPP::Simulation::Loop::Custom::install, TRN4JAVA::Functor::make_function(TRN4JAVA::Functor::measurement_matrix_callback), TRN4JAVA::Functor::loop_reply);
}
void Java_TRN4JAVA_Custom_00024Simulation_00024Reservoir_00024Feedforward_install(JNIEnv *env, jclass jclazz, jobject feedforward)
{
	TRACE_LOGGER;
	TRN4JAVA::Functor::install(env, feedforward, TRN4JAVA::Functor::INITIALIZER_CALLBACK_SIGNATURE, TRN4JAVA::Functor::weights_global_ref, TRN4CPP::Simulation::Reservoir::Weights::Feedforward::Custom::install, TRN4JAVA::Functor::make_function(TRN4JAVA::Functor::custom_weights_callback), TRN4JAVA::Functor::weights_reply);
}
void Java_TRN4JAVA_Custom_00024Simulation_00024Reservoir_00024Readout_install(JNIEnv *env, jclass jclazz, jobject readout)
{
	TRACE_LOGGER;
	TRN4JAVA::Functor::install(env, readout, TRN4JAVA::Functor::INITIALIZER_CALLBACK_SIGNATURE, TRN4JAVA::Functor::weights_global_ref, TRN4CPP::Simulation::Reservoir::Weights::Readout::Custom::install, TRN4JAVA::Functor::make_function(TRN4JAVA::Functor::custom_weights_callback), TRN4JAVA::Functor::weights_reply);
}
void Java_TRN4JAVA_Custom_00024Simulation_00024Reservoir_00024Recurrent_install(JNIEnv *env, jclass jclazz, jobject recurrent)
{
	TRACE_LOGGER;
	TRN4JAVA::Functor::install(env, recurrent, TRN4JAVA::Functor::INITIALIZER_CALLBACK_SIGNATURE, TRN4JAVA::Functor::weights_global_ref, TRN4CPP::Simulation::Reservoir::Weights::Recurrent::Custom::install, TRN4JAVA::Functor::make_function(TRN4JAVA::Functor::custom_weights_callback), TRN4JAVA::Functor::weights_reply);
}
void Java_TRN4JAVA_Custom_00024Simulation_00024Reservoir_00024Weights_notify(JNIEnv *env, jobject initializer, jlong simulation_id, jfloatArray weights, jlong batch_size, jlong rows, jlong cols)
{
	TRACE_LOGGER;
	TRN4JAVA::Functor::notify(env, initializer, (unsigned long long)simulation_id, TRN4JAVA::Functor::weights_global_ref, TRN4JAVA::Functor::weights_reply, (unsigned long long)simulation_id, TRN4JAVA::Convert::to_float_vector(env, weights), (std::size_t)batch_size, (std::size_t)rows, (std::size_t)cols);
}
void Java_TRN4JAVA_Custom_00024Simulation_00024Scheduler_notify(JNIEnv *env, jobject scheduler, jlong simulation_id, jlong evaluation_id, jintArray offsets, jintArray durations)
{
	TRACE_LOGGER;
	TRN4JAVA::Functor::notify(env,  scheduler, (unsigned long long)simulation_id, TRN4JAVA::Functor::scheduler_global_ref, TRN4JAVA::Functor::scheduler_reply, (unsigned long long)simulation_id, (unsigned long long)evaluation_id, TRN4JAVA::Convert::to_int_vector(env, offsets), TRN4JAVA::Convert::to_int_vector(env, durations));
}
void Java_TRN4JAVA_Custom_00024Simulation_00024Scheduler_installJava_TRN4JAVA_Simulation_00024Scheduler_install(JNIEnv *env, jclass jclazz, jobject scheduler)
{
	TRACE_LOGGER;
	TRN4JAVA::Functor::install(env, scheduler, TRN4JAVA::Functor::INITIALIZER_CALLBACK_SIGNATURE, TRN4JAVA::Functor::scheduler_global_ref, TRN4CPP::Simulation::Scheduler::Custom::install, TRN4JAVA::Functor::make_function(TRN4JAVA::Functor::recording_scheduler_callback), TRN4JAVA::Functor::scheduler_reply);
}
void Java_TRN4JAVA_Custom_00024Simulation_00024Scheduler_00024Mutator_notify(JNIEnv *env, jobject mutator, jlong simulation_id, jlong evaluation_id, jintArray offsets, jintArray durations)
{
	TRACE_LOGGER;
	TRN4JAVA::Functor::notify(env, mutator, (unsigned long long)simulation_id, TRN4JAVA::Functor::mutator_global_ref, TRN4JAVA::Functor::mutator_reply, (unsigned long long)simulation_id, (unsigned long long)evaluation_id, TRN4JAVA::Convert::to_int_vector(env, offsets), TRN4JAVA::Convert::to_int_vector(env, durations));
}
void Java_TRN4JAVA_Custom_00024Simulation_00024Scheduler_00024Mutator_install(JNIEnv *env, jclass jclazz, jobject mutator)
{
	TRACE_LOGGER;
	TRN4JAVA::Functor::install(env, mutator, TRN4JAVA::Functor::SCHEDULING_CALLBACK_SIGNATURE, TRN4JAVA::Functor::mutator_global_ref, TRN4CPP::Simulation::Scheduler::Mutator::Custom::install, TRN4JAVA::Functor::make_function(TRN4JAVA::Functor::custom_mutator_callback), TRN4JAVA::Functor::mutator_reply);
}

