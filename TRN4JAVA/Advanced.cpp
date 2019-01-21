#include "stdafx.h"

#include "Functor.h"
#include "Convert.h"

#include "TRN4JAVA_Advanced.h"
#include "TRN4JAVA_Advanced_Engine.h"
#include "TRN4JAVA_Advanced_Engine_Events.h"
#include "TRN4JAVA_Advanced_Engine_Events_Ack.h"
#include "TRN4JAVA_Advanced_Engine_Events_Allocated.h"
#include "TRN4JAVA_Advanced_Engine_Events_Completed.h"
#include "TRN4JAVA_Advanced_Engine_Events_Configured.h"
#include "TRN4JAVA_Advanced_Engine_Events_Deallocated.h"
#include "TRN4JAVA_Advanced_Engine_Events_Primed.h"
#include "TRN4JAVA_Advanced_Engine_Events_Processor.h"
#include "TRN4JAVA_Advanced_Engine_Events_Tested.h"
#include "TRN4JAVA_Advanced_Engine_Events_Trained.h"
#include "TRN4JAVA_Advanced_Simulation.h"
#include "TRN4JAVA_Advanced_Simulation_Loop.h"
#include "TRN4JAVA_Advanced_Simulation_Loop_Custom.h"
#include "TRN4JAVA_Advanced_Simulation_Measurement.h"
#include "TRN4JAVA_Advanced_Simulation_Measurement_Position.h"
#include "TRN4JAVA_Advanced_Simulation_Measurement_Position_FrechetDistance.h"
#include "TRN4JAVA_Advanced_Simulation_Measurement_Position_MeanSquareError.h"
#include "TRN4JAVA_Advanced_Simulation_Measurement_Position_Raw.h"
#include "TRN4JAVA_Advanced_Simulation_Measurement_Readout.h"
#include "TRN4JAVA_Advanced_Simulation_Measurement_Readout_FrechetDistance.h"
#include "TRN4JAVA_Advanced_Simulation_Measurement_Readout_MeanSquareError.h"
#include "TRN4JAVA_Advanced_Simulation_Measurement_Readout_Raw.h"
#include "TRN4JAVA_Advanced_Simulation_Recording.h"
#include "TRN4JAVA_Advanced_Simulation_Recording_Performances.h"
#include "TRN4JAVA_Advanced_Simulation_Recording_Scheduling.h"
#include "TRN4JAVA_Advanced_Simulation_Recording_States.h"
#include "TRN4JAVA_Advanced_Simulation_Recording_Weights.h"
#include "TRN4JAVA_Advanced_Simulation_Reservoir.h"
#include "TRN4JAVA_Advanced_Simulation_Reservoir_Weights.h"
#include "TRN4JAVA_Advanced_Simulation_Reservoir_Weights_Feedback.h"
#include "TRN4JAVA_Advanced_Simulation_Reservoir_Weights_Feedback_Custom.h"
#include "TRN4JAVA_Advanced_Simulation_Reservoir_Weights_Feedforward.h"
#include "TRN4JAVA_Advanced_Simulation_Reservoir_Weights_Feedforward_Custom.h"
#include "TRN4JAVA_Advanced_Simulation_Reservoir_Weights_Readout.h"
#include "TRN4JAVA_Advanced_Simulation_Reservoir_Weights_Readout_Custom.h"
#include "TRN4JAVA_Advanced_Simulation_Reservoir_Weights_Recurrent.h"
#include "TRN4JAVA_Advanced_Simulation_Reservoir_Weights_Recurrent_Custom.h"
#include "TRN4JAVA_Advanced_Simulation_Scheduler.h"
#include "TRN4JAVA_Advanced_Simulation_Scheduler_Custom.h"
#include "TRN4JAVA_Advanced_Simulation_Scheduler_Mutator.h"
#include "TRN4JAVA_Advanced_Simulation_Scheduler_Mutator_Custom.h"

#include "TRN4CPP/Advanced.h"

void Java_TRN4JAVA_Advanced_00024Engine_00024Events_00024Ack_install(JNIEnv *env, jclass jclazz, jobject functor)
{
	TRACE_LOGGER;
	TRN4JAVA::Functor::install(env, functor, TRN4JAVA::Functor::EVENT_ACK_CALLBACK_SIGNATURE, TRN4JAVA::Functor::events_global_ref,
		TRN4CPP::Engine::Events::Ack::install,
		TRN4JAVA::Functor::make_function(TRN4JAVA::Functor::event_ack_callback));
}
void Java_TRN4JAVA_Advanced_00024Engine_00024Events_00024Allocated_install(JNIEnv *env, jclass jclazz, jobject functor)
{
	TRACE_LOGGER;
	TRN4JAVA::Functor::install(env, functor, TRN4JAVA::Functor::EVENT_SIMULATION_ALLOCATION_CALLBACK_SIGNATURE, TRN4JAVA::Functor::events_global_ref,
		TRN4CPP::Engine::Events::Allocated::install,
		TRN4JAVA::Functor::make_function(TRN4JAVA::Functor::event_simulation_allocation_callback));
}
void Java_TRN4JAVA_Advanced_00024Engine_00024Events_00024Completed_install(JNIEnv *env, jclass jclazz, jobject functor)
{
	TRACE_LOGGER;
	TRN4JAVA::Functor::install(env, functor, TRN4JAVA::Functor::EVENT_CALLBACK_SIGNATURE, TRN4JAVA::Functor::events_global_ref,
		TRN4CPP::Engine::Events::Completed::install,
		TRN4JAVA::Functor::make_function(TRN4JAVA::Functor::event_callback));
}

void Java_TRN4JAVA_Advanced_00024Engine_00024Events_00024Configured_install(JNIEnv *env, jclass jclazz, jobject functor)
{
	TRACE_LOGGER;
	TRN4JAVA::Functor::install(env, functor, TRN4JAVA::Functor::EVENT_SIMULATION_STATE_CALLBACK_SIGNATURE, TRN4JAVA::Functor::events_global_ref,
		TRN4CPP::Engine::Events::Configured::install,
		TRN4JAVA::Functor::make_function(TRN4JAVA::Functor::event_simulation_state_callback));
}

void Java_TRN4JAVA_Advanced_00024Engine_00024Events_00024Deallocated_install(JNIEnv *env, jclass jclazz, jobject functor)
{
	TRACE_LOGGER;
	TRN4JAVA::Functor::install(env, functor, TRN4JAVA::Functor::EVENT_SIMULATION_ALLOCATION_CALLBACK_SIGNATURE, TRN4JAVA::Functor::events_global_ref,
		TRN4CPP::Engine::Events::Deallocated::install,
		TRN4JAVA::Functor::make_function(TRN4JAVA::Functor::event_simulation_allocation_callback));
}
void Java_TRN4JAVA_Advanced_00024Engine_00024Events_00024Primed_install(JNIEnv *env, jclass jclazz, jobject functor)
{
	TRACE_LOGGER;
	TRN4JAVA::Functor::install(env, functor, TRN4JAVA::Functor::EVENT_SIMULATION_STATE_CALLBACK_SIGNATURE, TRN4JAVA::Functor::events_global_ref,
		TRN4CPP::Engine::Events::Primed::install,
		TRN4JAVA::Functor::make_function(TRN4JAVA::Functor::event_simulation_state_evaluation_callback));
}
void Java_TRN4JAVA_Advanced_00024Engine_00024Events_00024Processor_install(JNIEnv *env, jclass jclazz, jobject functor)
{
	TRACE_LOGGER;
	TRN4JAVA::Functor::install(env, functor, TRN4JAVA::Functor::EVENT_PROCESSOR_CALLBACK_SIGNATURE, TRN4JAVA::Functor::events_global_ref,
		TRN4CPP::Engine::Events::Processor::install,
		TRN4JAVA::Functor::make_function(TRN4JAVA::Functor::event_processor_callback));
}
void Java_TRN4JAVA_Advanced_00024Engine_00024Events_00024Tested_install(JNIEnv *env, jclass jclazz, jobject functor)
{
	TRACE_LOGGER;
	TRN4JAVA::Functor::install(env, functor, TRN4JAVA::Functor::EVENT_SIMULATION_STATE_CALLBACK_SIGNATURE, TRN4JAVA::Functor::events_global_ref,
		TRN4CPP::Engine::Events::Tested::install,
		TRN4JAVA::Functor::make_function(TRN4JAVA::Functor::event_simulation_state_evaluation_callback));
}
void Java_TRN4JAVA_Advanced_00024Engine_00024Events_00024Trained_install(JNIEnv *env, jclass jclazz, jobject functor)
{
	TRACE_LOGGER;
	TRN4JAVA::Functor::install(env, functor, TRN4JAVA::Functor::EVENT_SIMULATION_STATE_CALLBACK_SIGNATURE, TRN4JAVA::Functor::events_global_ref,
		TRN4CPP::Engine::Events::Trained::install,
		TRN4JAVA::Functor::make_function(TRN4JAVA::Functor::event_simulation_state_evaluation_callback));
}

void Java_TRN4JAVA_Advanced_00024Simulation_00024Encoder_00024Custom_configure(JNIEnv *env, jclass clazz, jlong simulation_id, jlong batch_size, jlong stimulus_size,  jobject encoder)
{
	TRACE_LOGGER;
	TRN4JAVA::Functor::install(env, encoder, TRN4JAVA::Functor::ENCODER_CALLBACK_SIGNATURE, TRN4JAVA::Functor::encoder_global_ref,

		std::bind(&TRN4CPP::Simulation::Encoder::Custom::configure, (unsigned long long)simulation_id, (std::size_t)batch_size, (std::size_t)stimulus_size, 
			std::placeholders::_1, std::placeholders::_2, std::placeholders::_3),
		TRN4JAVA::Functor::make_function(TRN4JAVA::Functor::measurement_matrix_callback), TRN4JAVA::Functor::encoder_reply, TRN4JAVA::Functor::loop_reply);
}

void Java_TRN4JAVA_Advanced_00024Simulation_00024Loop_00024Custom_configure(JNIEnv *env, jclass jclazz, jlong simulation_id, jlong batch_size, jlong stimulus_size, jobject stimulus)
{
	TRACE_LOGGER;
	TRN4JAVA::Functor::install(env, stimulus, TRN4JAVA::Functor::LOOP_CALLBACK_SIGNATURE, TRN4JAVA::Functor::loop_global_ref,

		std::bind(&TRN4CPP::Simulation::Loop::Custom::configure, (unsigned long long)simulation_id, (std::size_t)batch_size, (std::size_t)stimulus_size, std::placeholders::_1, std::placeholders::_2),
		TRN4JAVA::Functor::make_function(TRN4JAVA::Functor::measurement_matrix_callback), TRN4JAVA::Functor::loop_reply);
}

void Java_TRN4JAVA_Advanced_00024Simulation_00024Measurement_00024Position_00024FrechetDistance_configure(JNIEnv *env, jclass jclazz, jlong simulation_id, jlong batch_size, jstring norm, jstring aggregator, jobject processed)
{
	TRACE_LOGGER;
	TRN4JAVA::Functor::install(env, processed, TRN4JAVA::Functor::PROCESSED_CALLBACK_SIGNATURE, TRN4JAVA::Functor::processed_global_ref,
		std::bind(&TRN4CPP::Simulation::Measurement::Position::FrechetDistance::configure, (unsigned long long)simulation_id, (std::size_t)batch_size, TRN4JAVA::Convert::to_string(env, norm), TRN4JAVA::Convert::to_string(env, aggregator), std::placeholders::_1),
		TRN4JAVA::Functor::make_function(TRN4JAVA::Functor::measurement_matrix_callback));
}
void Java_TRN4JAVA_Advanced_00024Simulation_00024Measurement_00024Position_00024MeanSquareError_configure(JNIEnv *env, jclass jclazz, jlong simulation_id, jlong batch_size, jobject processed)
{
	TRACE_LOGGER;
	TRN4JAVA::Functor::install(env, processed, TRN4JAVA::Functor::PROCESSED_CALLBACK_SIGNATURE, TRN4JAVA::Functor::processed_global_ref,
		std::bind(&TRN4CPP::Simulation::Measurement::Position::MeanSquareError::configure, (unsigned long long)simulation_id, (std::size_t)batch_size, std::placeholders::_1),
		TRN4JAVA::Functor::make_function(TRN4JAVA::Functor::measurement_matrix_callback));
}
void Java_TRN4JAVA_Advanced_00024Simulation_00024Measurement_00024Position_00024Raw_configure(JNIEnv *env, jclass jclazz, jlong simulation_id, jlong batch_size, jobject raw)
{
	TRACE_LOGGER;
	TRN4JAVA::Functor::install(env, raw, TRN4JAVA::Functor::RAW_CALLBACK_SIGNATURE, TRN4JAVA::Functor::raw_global_ref,
		std::bind(&TRN4CPP::Simulation::Measurement::Position::Custom::configure, (unsigned long long)simulation_id, (std::size_t)batch_size, std::placeholders::_1),
		TRN4JAVA::Functor::make_function(TRN4JAVA::Functor::measurement_raw_callback));
}
void Java_TRN4JAVA_Advanced_00024Simulation_00024Measurement_00024Readout_00024FrechetDistance_configure(JNIEnv *env, jclass jclazz, jlong simulation_id, jlong batch_size, jstring norm, jstring aggregator, jobject processed)
{
	TRACE_LOGGER;
	TRN4JAVA::Functor::install(env, processed, TRN4JAVA::Functor::PROCESSED_CALLBACK_SIGNATURE, TRN4JAVA::Functor::processed_global_ref,
		std::bind(&TRN4CPP::Simulation::Measurement::Readout::FrechetDistance::configure, (unsigned long long)simulation_id, (std::size_t)batch_size, TRN4JAVA::Convert::to_string(env, norm), TRN4JAVA::Convert::to_string(env, aggregator), std::placeholders::_1),
		TRN4JAVA::Functor::make_function(TRN4JAVA::Functor::measurement_matrix_callback));
}
void Java_TRN4JAVA_Advanced_00024Simulation_00024Measurement_00024Readout_00024MeanSquareError_configure(JNIEnv *env, jclass jclazz, jlong simulation_id, jlong batch_size, jobject processed)
{
	TRACE_LOGGER;
	TRN4JAVA::Functor::install(env, processed, TRN4JAVA::Functor::PROCESSED_CALLBACK_SIGNATURE, TRN4JAVA::Functor::processed_global_ref,
		std::bind(&TRN4CPP::Simulation::Measurement::Readout::MeanSquareError::configure, (unsigned long long)simulation_id, (std::size_t)batch_size, std::placeholders::_1),
		TRN4JAVA::Functor::make_function(TRN4JAVA::Functor::measurement_matrix_callback));
}
void Java_TRN4JAVA_Advanced_00024Simulation_00024Measurement_00024Readout_00024Raw_configure(JNIEnv *env, jclass jclazz, jlong simulation_id, jlong batch_size, jobject raw)
{
	TRACE_LOGGER;
	TRN4JAVA::Functor::install(env, raw, TRN4JAVA::Functor::RAW_CALLBACK_SIGNATURE, TRN4JAVA::Functor::raw_global_ref,
		std::bind(&TRN4CPP::Simulation::Measurement::Readout::Custom::configure, (unsigned long long)simulation_id, (std::size_t)batch_size, std::placeholders::_1),
		TRN4JAVA::Functor::make_function(TRN4JAVA::Functor::measurement_raw_callback));
}
void Java_TRN4JAVA_Advanced_00024Simulation_00024Recording_00024Performances_configure(JNIEnv *env, jclass jclazz, jlong simulation_id, jobject performances, jboolean train, jboolean primed, jboolean generate)
{
	TRACE_LOGGER;
	TRN4JAVA::Functor::install(env, performances, TRN4JAVA::Functor::PERFORMANCES_CALLBACK_SIGNATURE, TRN4JAVA::Functor::recording_global_ref,
		std::bind(&TRN4CPP::Simulation::Recording::Performances::configure, (unsigned long long)simulation_id, std::placeholders::_1, (bool)train, (bool)primed, (bool)generate),
		TRN4JAVA::Functor::make_function(TRN4JAVA::Functor::recording_performances_callback));
}
void Java_TRN4JAVA_Simulation_00024Recording_00024Scheduling_configure__JLTRN4JAVA_Simulation_Recording_Scheduling_2(JNIEnv *env, jclass jclazz, jlong simulation_id, jobject scheduling)
{
	TRACE_LOGGER;
	TRN4JAVA::Functor::install(env, scheduling, TRN4JAVA::Functor::SCHEDULING_CALLBACK_SIGNATURE, TRN4JAVA::Functor::recording_global_ref,
		std::bind(&TRN4CPP::Simulation::Recording::Scheduling::configure, (unsigned long long)simulation_id, std::placeholders::_1),
		TRN4JAVA::Functor::make_function(TRN4JAVA::Functor::recording_scheduling_callback));
}
void Java_TRN4JAVA_Advanced_00024Simulation_00024Recording_00024States_configure(JNIEnv *env, jclass jclazz, jlong simulation_id, jobject states, jboolean train, jboolean prime, jboolean generate)
{
	TRACE_LOGGER;
	TRN4JAVA::Functor::install(env, states, TRN4JAVA::Functor::STATES_CALLBACK_SIGNATURE, TRN4JAVA::Functor::recording_global_ref,
		std::bind(&TRN4CPP::Simulation::Recording::States::configure, (unsigned long long)simulation_id, std::placeholders::_1, (bool)train, (bool)prime, (bool)generate),
		TRN4JAVA::Functor::make_function(TRN4JAVA::Functor::recording_states_callback));
}
void Java_TRN4JAVA_Advanced_00024Simulation_00024Recording_00024Weights_configure(JNIEnv *env, jclass jclazz, jlong simulation_id, jobject weights, jboolean initialize, jboolean train)
{
	TRACE_LOGGER;
	TRN4JAVA::Functor::install(env, weights, TRN4JAVA::Functor::WEIGHTS_CALLBACK_SIGNATURE, TRN4JAVA::Functor::recording_global_ref,
		std::bind(&TRN4CPP::Simulation::Recording::Weights::configure, (unsigned long long)simulation_id, std::placeholders::_1, (bool)initialize, (bool)train),
		TRN4JAVA::Functor::make_function(TRN4JAVA::Functor::recording_weights_callback));
}

void Java_TRN4JAVA_Advanced_00024Simulation_00024Reservoir_00024Weights_00024Feedforward_00024Custom_configure(JNIEnv *env, jclass jclazz, jlong simulation_id, jobject initializer)
{
	TRACE_LOGGER;
	TRN4JAVA::Functor::install(env, initializer, TRN4JAVA::Functor::WEIGHTS_CALLBACK_SIGNATURE, TRN4JAVA::Functor::weights_global_ref,
		std::bind(&TRN4CPP::Simulation::Reservoir::Weights::Feedforward::Custom::configure, (unsigned long long)simulation_id, std::placeholders::_1, std::placeholders::_2),
		TRN4JAVA::Functor::make_function(TRN4JAVA::Functor::custom_weights_callback), TRN4JAVA::Functor::weights_reply);
}
void Java_TRN4JAVA_Advanced_00024Simulation_00024Reservoir_00024Weights_00024Readout_00024Custom_configure(JNIEnv *env, jclass jclazz, jlong simulation_id, jobject initializer)
{
	TRACE_LOGGER;
	TRN4JAVA::Functor::install(env, initializer, TRN4JAVA::Functor::WEIGHTS_CALLBACK_SIGNATURE, TRN4JAVA::Functor::weights_global_ref,
		std::bind(&TRN4CPP::Simulation::Reservoir::Weights::Readout::Custom::configure, (unsigned long long)simulation_id, std::placeholders::_1, std::placeholders::_2),
		TRN4JAVA::Functor::make_function(TRN4JAVA::Functor::custom_weights_callback), TRN4JAVA::Functor::weights_reply);
}
void Java_TRN4JAVA_Advanced_00024Simulation_00024Reservoir_00024Weights_00024Recurrent_00024Custom_configure(JNIEnv *env, jclass jclazz, jlong simulation_id, jobject initializer)
{
	TRACE_LOGGER;
	TRN4JAVA::Functor::install(env, initializer, TRN4JAVA::Functor::WEIGHTS_CALLBACK_SIGNATURE, TRN4JAVA::Functor::weights_global_ref,
		std::bind(&TRN4CPP::Simulation::Reservoir::Weights::Recurrent::Custom::configure, (unsigned long long)simulation_id, std::placeholders::_1, std::placeholders::_2),
		TRN4JAVA::Functor::make_function(TRN4JAVA::Functor::custom_weights_callback), TRN4JAVA::Functor::weights_reply);
}
void Java_TRN4JAVA_Advanced_00024Simulation_00024Scheduler_00024Custom_configure(JNIEnv *env, jclass jclazz, jlong simulation_id, jlong seed, jobject scheduler, jstring tag)
{
	TRACE_LOGGER;
	TRN4JAVA::Functor::install(env, scheduler, TRN4JAVA::Functor::SCHEDULER_CALLBACK_SIGNATURE, TRN4JAVA::Functor::scheduler_global_ref,
		std::bind(&TRN4CPP::Simulation::Scheduler::Custom::configure, (unsigned long long)simulation_id, (unsigned long)seed, std::placeholders::_1, std::placeholders::_2, TRN4JAVA::Convert::to_string(env, tag)),
		TRN4JAVA::Functor::make_function(TRN4JAVA::Functor::custom_scheduler_callback), TRN4JAVA::Functor::scheduler_reply);
}
void Java_TRN4JAVA_Advanced_00024Simulation_00024Scheduler_00024Mutator_00024Custom_configure(JNIEnv *env, jclass jclazz, jlong simulation_id, jlong seed, jobject mutator)
{
	TRACE_LOGGER;
	TRN4JAVA::Functor::install(env, mutator, TRN4JAVA::Functor::SCHEDULER_CALLBACK_SIGNATURE, TRN4JAVA::Functor::scheduler_global_ref,
		std::bind(&TRN4CPP::Simulation::Scheduler::Mutator::Custom::configure, (unsigned long long)simulation_id, (unsigned long)seed, std::placeholders::_1, std::placeholders::_2),
		TRN4JAVA::Functor::make_function(TRN4JAVA::Functor::custom_mutator_callback), TRN4JAVA::Functor::mutator_reply);
}
