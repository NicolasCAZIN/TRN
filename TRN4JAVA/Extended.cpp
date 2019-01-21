#include "stdafx.h"

#include "Convert.h"
#include "Functor.h"

#include "TRN4JAVA_Extended.h"
#include "TRN4JAVA_Extended_Engine.h"
#include "TRN4JAVA_Extended_Engine_Execution.h"
#include "TRN4JAVA_Extended_Simulation.h"
#include "TRN4JAVA_Extended_Simulation_Decoder.h"
#include "TRN4JAVA_Extended_Simulation_Decoder_Kernel.h"
#include "TRN4JAVA_Extended_Simulation_Decoder_Kernel_Map.h"
#include "TRN4JAVA_Extended_Simulation_Decoder_Kernel_Model.h"
#include "TRN4JAVA_Extended_Simulation_Decoder_Linear.h"
#include "TRN4JAVA_Extended_Simulation_Encoder.h"
#include "TRN4JAVA_Extended_Simulation_Encoder_Model.h"
#include "TRN4JAVA_Extended_Simulation_Encoder_Custom.h"
#include "TRN4JAVA_Extended_Simulation_Loop.h"
#include "TRN4JAVA_Extended_Simulation_Loop_Copy.h"
#include "TRN4JAVA_Extended_Simulation_Loop_Custom.h"
#include "TRN4JAVA_Extended_Simulation_Loop_SpatialFilter.h"
#include "TRN4JAVA_Extended_Simulation_Measurement.h"
#include "TRN4JAVA_Extended_Simulation_Measurement_Position.h"
#include "TRN4JAVA_Extended_Simulation_Measurement_Position_FrechetDistance.h"
#include "TRN4JAVA_Extended_Simulation_Measurement_Position_MeanSquareError.h"
#include "TRN4JAVA_Extended_Simulation_Measurement_Position_Raw.h"
#include "TRN4JAVA_Extended_Simulation_Measurement_Readout.h"
#include "TRN4JAVA_Extended_Simulation_Measurement_Readout_FrechetDistance.h"
#include "TRN4JAVA_Extended_Simulation_Measurement_Readout_MeanSquareError.h"
#include "TRN4JAVA_Extended_Simulation_Measurement_Readout_Raw.h"
#include "TRN4JAVA_Extended_Simulation_Recording.h"
#include "TRN4JAVA_Extended_Simulation_Recording_Performances.h"
#include "TRN4JAVA_Extended_Simulation_Recording_Scheduling.h"
#include "TRN4JAVA_Extended_Simulation_Recording_States.h"
#include "TRN4JAVA_Extended_Simulation_Recording_Weights.h"
#include "TRN4JAVA_Extended_Simulation_Reservoir.h"
#include "TRN4JAVA_Extended_Simulation_Reservoir_Weights_Feedforward.h"
#include "TRN4JAVA_Extended_Simulation_Reservoir_Weights_Feedforward_Custom.h"
#include "TRN4JAVA_Extended_Simulation_Reservoir_Weights_Feedforward_Gaussian.h"
#include "TRN4JAVA_Extended_Simulation_Reservoir_Weights_Feedforward_Uniform.h"
#include "TRN4JAVA_Extended_Simulation_Reservoir_Weights_Readout.h"
#include "TRN4JAVA_Extended_Simulation_Reservoir_Weights_Readout_Custom.h"
#include "TRN4JAVA_Extended_Simulation_Reservoir_Weights_Readout_Gaussian.h"
#include "TRN4JAVA_Extended_Simulation_Reservoir_Weights_Readout_Uniform.h"
#include "TRN4JAVA_Extended_Simulation_Reservoir_Weights_Recurrent.h"
#include "TRN4JAVA_Extended_Simulation_Reservoir_Weights_Recurrent_Custom.h"
#include "TRN4JAVA_Extended_Simulation_Reservoir_Weights_Recurrent_Gaussian.h"
#include "TRN4JAVA_Extended_Simulation_Reservoir_Weights_Recurrent_Uniform.h"
#include "TRN4JAVA_Extended_Simulation_Reservoir_WidrowHoff.h"
#include "TRN4JAVA_Extended_Simulation_Scheduler.h"
#include "TRN4JAVA_Extended_Simulation_Scheduler_Custom.h"
#include "TRN4JAVA_Extended_Simulation_Scheduler_Mutator.h"
#include "TRN4JAVA_Extended_Simulation_Scheduler_Mutator_Custom.h"
#include "TRN4JAVA_Extended_Simulation_Scheduler_Mutator_Reverse.h"
#include "TRN4JAVA_Extended_Simulation_Scheduler_Mutator_Shuffle.h"
#include "TRN4JAVA_Extended_Simulation_Scheduler_Snippets.h"
#include "TRN4JAVA_Extended_Simulation_Scheduler_Tiled.h"

#include "TRN4CPP/Extended.h"

#include "Helper/Logger.h"

void Java_TRN4JAVA_Extended_00024Engine_00024Execution_run(JNIEnv *env, jclass jclazz)
{
	TRACE_LOGGER;
	try
	{
		TRN4CPP::Engine::Execution::run();
	}
	catch (std::exception &e)
	{
		ERROR_LOGGER << e.what(); env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
	}
}
void Java_TRN4JAVA_Extended_00024Simulation_allocate(JNIEnv *env, jclass jclazz, jlong simulation_id)
{
	TRACE_LOGGER;
	try
	{
		TRN4CPP::Simulation::allocate((unsigned long long)simulation_id);
	}
	catch (std::exception &e)
	{
		ERROR_LOGGER << e.what(); env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
	}
}
void Java_TRN4JAVA_Extended_00024Simulation_deallocate(JNIEnv *env, jclass jclazz, jlong simulation_id)
{
	TRACE_LOGGER; 
	try
	{
		TRN4CPP::Simulation::deallocate((unsigned long long)simulation_id);
		std::unique_lock<std::mutex> guard(TRN4JAVA::Functor::functor_mutex);

		if (TRN4JAVA::Functor::lookup_ref.find((unsigned long long)simulation_id) != TRN4JAVA::Functor::lookup_ref.end())
		{
			DEBUG_LOGGER << "Erasing simulation #" << simulation_id << " entries in lookup_ref";
			TRN4JAVA::Functor::lookup_ref.erase((unsigned long long)simulation_id);
		}
	}
	catch (std::exception &e)
	{
		ERROR_LOGGER << e.what(); env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
	}
}
void Java_TRN4JAVA_Extended_00024Simulation_train(JNIEnv *env, jclass jclazz, jlong simulation_id, jlong evaluation_id, jstring label, jstring incoming, jstring expected)
{
	TRACE_LOGGER;
	try
	{
		TRN4CPP::Simulation::train((unsigned long long)simulation_id, (unsigned long long)evaluation_id, TRN4JAVA::Convert::to_string(env, label), TRN4JAVA::Convert::to_string(env, incoming), TRN4JAVA::Convert::to_string(env, expected));
	}
	catch (std::exception &e)
	{
		ERROR_LOGGER << e.what(); env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
	}
}
void Java_TRN4JAVA_Extended_00024Simulation_test(JNIEnv *env, jclass jclazz, jlong simulation_id, jlong evaluation_id, jstring sequence, jstring incoming, jstring expected, jint preamble, jboolean autonomous, jint supplementary_generations)
{
	TRACE_LOGGER;
	try
	{
		TRN4CPP::Simulation::test((unsigned long long)simulation_id, (unsigned long long)evaluation_id, TRN4JAVA::Convert::to_string(env, sequence), TRN4JAVA::Convert::to_string(env, incoming), TRN4JAVA::Convert::to_string(env, expected), (unsigned int)preamble, (bool)autonomous, (unsigned int)supplementary_generations);
	}
	catch (std::exception &e)
	{
		ERROR_LOGGER << e.what(); env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
	}
}
void Java_TRN4JAVA_Extended_00024Simulation_declare_1sequence(JNIEnv *env, jclass jclazz, jlong simulation_id, jstring label, jstring tag, jfloatArray sequence, jlong observations)
{
	TRACE_LOGGER;
	try
	{
		TRN4CPP::Simulation::declare_sequence((unsigned long long)simulation_id, TRN4JAVA::Convert::to_string(env, label), TRN4JAVA::Convert::to_string(env, tag), TRN4JAVA::Convert::to_float_vector(env, sequence), observations);
	}
	catch (std::exception &e)
	{
		ERROR_LOGGER << e.what(); env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
	}
}
void Java_TRN4JAVA_Extended_00024Simulation_declare_1set(JNIEnv *env, jclass jclazz, jlong simulation_id, jstring label, jstring tag, jobjectArray labels)
{
	TRACE_LOGGER;
	try
	{
		TRN4CPP::Simulation::declare_set((unsigned long long)simulation_id, TRN4JAVA::Convert::to_string(env, label), TRN4JAVA::Convert::to_string(env, tag), TRN4JAVA::Convert::to_string_vector(env, labels));
	}
	catch (std::exception &e)
	{
		ERROR_LOGGER << e.what(); env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
	}
}
void Java_TRN4JAVA_Extended_00024Simulation_configure_1begin(JNIEnv *env, jclass jclazz, jlong simulation_id)
{
	TRACE_LOGGER;
	try
	{
		TRN4CPP::Simulation::configure_begin((unsigned long long)simulation_id);
	}
	catch (std::exception &e)
	{
		ERROR_LOGGER << e.what(); env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
	}
}
void Java_TRN4JAVA_Extended_00024Simulation_configure_1end(JNIEnv *env, jclass jclazz, jlong simulation_id)
{
	TRACE_LOGGER;
	try
	{
		TRN4CPP::Simulation::configure_end((unsigned long long)simulation_id);
	}
	catch (std::exception &e)
	{
		ERROR_LOGGER << e.what(); env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
	}
}

void Java_TRN4JAVA_Extended_00024Simulation_00024Decoder_00024Kernel_00024Map_configure(JNIEnv *env, jclass clazz, jlong simulation_id, jlong batch_size, jlong stimulus_size, jlong rows, jlong cols, jfloat x_min, jfloat x_max, jfloat y_min, jfloat y_max, jfloat sigma, jfloat radius, jfloat angle, jfloat scale, jlong seed, jfloatArray response )
{
	TRACE_LOGGER;
	try
	{
		TRN4CPP::Simulation::Decoder::Kernel::Map::configure((unsigned long long)simulation_id,
			(std::size_t)batch_size, (std::size_t)stimulus_size, 
			(std::size_t)rows, (std::size_t)cols,
			std::make_pair((float)x_min, (float)x_max), std::make_pair((float)y_min, (float)y_max),
			(float)sigma, (float)radius, (float)angle, (float)scale,
			(unsigned long )seed, TRN4JAVA::Convert::to_float_vector(env, response));
	}
	catch (std::exception &e)
	{
		ERROR_LOGGER << e.what(); env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
	}
}
void Java_TRN4JAVA_Extended_00024Simulation_00024Decoder_00024Kernel_00024Model_configure(JNIEnv *env, jclass clazz, jlong simulation_id, jlong batch_size, jlong stimulus_size,  jlong rows, jlong cols, jfloat x_min, jfloat x_max, jfloat y_min, jfloat y_max, jfloat sigma, jfloat radius, jfloat angle, jfloat scale, jlong seed, jfloatArray cx, jfloatArray cy, jfloatArray width)
{
	TRACE_LOGGER;
	try
	{
		TRN4CPP::Simulation::Decoder::Kernel::Model::configure((unsigned long long)simulation_id,
			(std::size_t)batch_size, (std::size_t)stimulus_size,
			(std::size_t)rows, (std::size_t)cols,
			std::make_pair((float)x_min, (float)x_max), std::make_pair((float)y_min, (float)y_max),
			(float)sigma, (float)radius, (float)angle, (float)scale,
			(unsigned long)seed, TRN4JAVA::Convert::to_float_vector(env, cx), TRN4JAVA::Convert::to_float_vector(env, cy), TRN4JAVA::Convert::to_float_vector(env, width));
	}
	catch (std::exception &e)
	{
		ERROR_LOGGER << e.what(); env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
	}
}
void Java_TRN4JAVA_Extended_00024Simulation_00024Decoder_00024Linear_configure(JNIEnv *env, jclass clazz, jlong simulation_id, jlong batch_size, jlong stimulus_size, jfloatArray cx, jfloatArray cy)
{
	TRACE_LOGGER;
	try
	{
		TRN4CPP::Simulation::Decoder::Linear::configure((unsigned long long)simulation_id,
			(std::size_t)batch_size, (std::size_t)stimulus_size,
			TRN4JAVA::Convert::to_float_vector(env, cx), TRN4JAVA::Convert::to_float_vector(env, cy));
	}
	catch (std::exception &e)
	{
		ERROR_LOGGER << e.what(); env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
	}
}
void Java_TRN4JAVA_Extended_00024Simulation_00024Encoder_00024Custom_configure(JNIEnv *env, jclass clazz, jlong simulation_id, jlong batch_size, jlong stimulus_size)
{
	TRACE_LOGGER;
	try
	{
		TRN4CPP::Simulation::Encoder::Custom::configure((unsigned long long)simulation_id,
			(std::size_t)batch_size, (std::size_t)stimulus_size);
	}
	catch (std::exception &e)
	{
		ERROR_LOGGER << e.what(); env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
	}
}
void Java_TRN4JAVA_Extended_00024Simulation_00024Encoder_00024Model_configure(JNIEnv *env, jclass clazz, jlong simulation_id, jlong batch_size, jlong stimulus_size, jfloatArray cx, jfloatArray cy, jfloatArray width)
{
	TRACE_LOGGER;
	try
	{
		TRN4CPP::Simulation::Encoder::Model::configure((unsigned long long)simulation_id,
			(std::size_t)batch_size, (std::size_t)stimulus_size, TRN4JAVA::Convert::to_float_vector(env, cx), TRN4JAVA::Convert::to_float_vector(env, cy), TRN4JAVA::Convert::to_float_vector(env, width));
	}
	catch (std::exception &e)
	{
		ERROR_LOGGER << e.what(); env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
	}
}
void Java_TRN4JAVA_Extended_00024Simulation_00024Loop_00024Copy_configure(JNIEnv *env, jclass jclazz, jlong simulation_id, jlong batch_size, jlong stimulus_size)
{
	TRACE_LOGGER;
	try
	{
		TRN4CPP::Simulation::Loop::Copy::configure((unsigned long long)simulation_id, (std::size_t)batch_size, (std::size_t)stimulus_size);
	}
	catch (std::exception &e)
	{
		ERROR_LOGGER << e.what(); env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
	}
}
void Java_TRN4JAVA_Extended_00024Simulation_00024Loop_00024Custom_configure(JNIEnv *env, jclass jclazz, jlong simulation_id, jlong batch_size, jlong stimulus_size)
{
	TRACE_LOGGER;
	try
	{
		TRN4CPP::Simulation::Loop::Custom::configure((unsigned long long)simulation_id, (std::size_t)batch_size, (std::size_t)stimulus_size);
	}
	catch (std::exception &e)
	{
		ERROR_LOGGER << e.what(); env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
	}
}
void Java_TRN4JAVA_Extended_00024Simulation_00024Loop_00024SpatialFilter_configure(JNIEnv *env, jclass jclazz, jlong simulation_id, jlong batch_size, jlong stimulus_size, jstring tag)
{
	TRACE_LOGGER;
	try
	{
		TRN4CPP::Simulation::Loop::SpatialFilter::configure((unsigned long long)simulation_id, (std::size_t)batch_size, (std::size_t)stimulus_size, TRN4JAVA::Convert::to_string(env, tag));
	}
	catch (std::exception &e)
	{
		ERROR_LOGGER << e.what(); env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
	}
}
void Java_TRN4JAVA_Extended_00024Simulation_00024Measurement_00024Position_00024FrechetDistance_configure(JNIEnv *env, jclass jclazz, jlong simulation_id, jlong batch_size, jstring norm, jstring aggregator)
{
	TRACE_LOGGER;
	try
	{
		TRN4CPP::Simulation::Measurement::Position::FrechetDistance::configure((unsigned long long)simulation_id, (std::size_t)batch_size, TRN4JAVA::Convert::to_string(env, norm), TRN4JAVA::Convert::to_string(env, aggregator));
	}
	catch (std::exception &e)
	{
		ERROR_LOGGER << e.what(); env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
	}
}
void Java_TRN4JAVA_Extended_00024Simulation_00024Measurement_00024Position_00024MeanSquareError_configure(JNIEnv *env, jclass jclazz, jlong simulation_id, jlong batch_size)
{
	TRACE_LOGGER;
	try
	{
		TRN4CPP::Simulation::Measurement::Position::MeanSquareError::configure((unsigned long long)simulation_id, (std::size_t)batch_size);
	}
	catch (std::exception &e)
	{
		ERROR_LOGGER << e.what(); env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
	}
}
void Java_TRN4JAVA_Extended_00024Simulation_00024Measurement_00024Position_00024Raw_configure(JNIEnv *env, jclass jclazz, jlong simulation_id, jlong batch_size)
{
	TRACE_LOGGER;
	try
	{
		TRN4CPP::Simulation::Measurement::Position::Custom::configure((unsigned long long)simulation_id, (std::size_t)batch_size);
	}
	catch (std::exception &e)
	{
		ERROR_LOGGER << e.what(); env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
	}
}
void Java_TRN4JAVA_Extended_00024Simulation_00024Measurement_00024Readout_00024FrechetDistance_configure(JNIEnv *env, jclass jclazz, jlong simulation_id, jlong batch_size, jstring norm, jstring aggregator)
{
	TRACE_LOGGER;
	try
	{
		TRN4CPP::Simulation::Measurement::Readout::FrechetDistance::configure((unsigned long long)simulation_id, (std::size_t)batch_size, TRN4JAVA::Convert::to_string(env, norm), TRN4JAVA::Convert::to_string(env, aggregator));
	}
	catch (std::exception &e)
	{
		ERROR_LOGGER << e.what(); env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
	}
}
void Java_TRN4JAVA_Extended_00024Simulation_00024Measurement_00024Readout_00024MeanSquareError_configure(JNIEnv *env, jclass jclazz, jlong simulation_id, jlong batch_size)
{
	TRACE_LOGGER;
	try
	{
		TRN4CPP::Simulation::Measurement::Readout::MeanSquareError::configure((unsigned long long)simulation_id, (std::size_t)batch_size);
	}
	catch (std::exception &e)
	{
		ERROR_LOGGER << e.what(); env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
	}
}
void Java_TRN4JAVA_Extended_00024Simulation_00024Measurement_00024Readout_00024Raw_configure(JNIEnv *env, jclass jclazz, jlong simulation_id, jlong batch_size)
{
	TRACE_LOGGER;
	try
	{
		TRN4CPP::Simulation::Measurement::Readout::Custom::configure((unsigned long long)simulation_id, (std::size_t)batch_size);
	}
	catch (std::exception &e)
	{
		ERROR_LOGGER << e.what(); env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
	}
}
void Java_TRN4JAVA_Extended_00024Simulation_00024Recording_00024Performances_configure(JNIEnv *env, jclass jclazz, jlong simulation_id, jboolean train, jboolean primed, jboolean generate)
{
	TRACE_LOGGER;
	try
	{
		TRN4CPP::Simulation::Recording::Performances::configure((unsigned long long)simulation_id, (bool)train, (bool)primed, (bool)generate);
	}
	catch (std::exception &e)
	{
		ERROR_LOGGER << e.what(); env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
	}
}
void Java_TRN4JAVA_Extended_00024Simulation_00024Recording_00024Scheduling_configure(JNIEnv *env, jclass jclazz, jlong simulation_id)
{
	TRACE_LOGGER;
	try
	{
		TRN4CPP::Simulation::Recording::Scheduling::configure((unsigned long long)simulation_id);
	}
	catch (std::exception &e)
	{
		ERROR_LOGGER << e.what(); env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
	}
}
void Java_TRN4JAVA_Extended_00024Simulation_00024Recording_00024States_configure(JNIEnv *env, jclass jclazz, jlong simulation_id, jboolean train, jboolean primed, jboolean generate)
{
	TRACE_LOGGER;
	try
	{
		TRN4CPP::Simulation::Recording::States::configure((unsigned long long)simulation_id, (bool)train, (bool)primed, (bool)generate);
	}
	catch (std::exception &e)
	{
		ERROR_LOGGER << e.what(); env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
	}
}
void Java_TRN4JAVA_Extended_00024Simulation_00024Recording_00024Weights_configure(JNIEnv *env, jclass jclazz, jlong simulation_id, jboolean initialize, jboolean train)
{
	TRACE_LOGGER;
	try
	{
		TRN4CPP::Simulation::Recording::Weights::configure((unsigned long long)simulation_id, (bool)initialize, (bool)train);
	}
	catch (std::exception &e)
	{
		ERROR_LOGGER << e.what(); env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
	}
}

void Java_TRN4JAVA_Extended_00024Simulation_00024Reservoir_00024Weights_00024Feedforward_00024Custom_configure(JNIEnv *env, jclass jclazz, jlong simulation_id)
{
	TRACE_LOGGER;
	try
	{
		TRN4CPP::Simulation::Reservoir::Weights::Feedforward::Custom::configure((unsigned long long)simulation_id);
	}
	catch (std::exception &e)
	{
		ERROR_LOGGER << e.what(); env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
	}
}
void Java_TRN4JAVA_Extended_00024Simulation_00024Reservoir_00024Weights_00024Feedforward_00024Gaussian_configure(JNIEnv *env, jclass jclazz, jlong simulation_id, jfloat mu, jfloat sigma, jfloat sparsity)
{
	TRACE_LOGGER;
	try
	{
		TRN4CPP::Simulation::Reservoir::Weights::Feedforward::Gaussian::configure((unsigned long long)simulation_id, (float)mu, (float)sigma, (float)sparsity);
	}
	catch (std::exception &e)
	{
		ERROR_LOGGER << e.what(); env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
	}
}
void Java_TRN4JAVA_Extended_00024Simulation_00024Reservoir_00024Weights_00024Feedforward_00024Uniform_configure(JNIEnv *env, jclass jclazz, jlong simulation_id, jfloat a, jfloat b, jfloat sparsity)
{
	TRACE_LOGGER;
	try
	{
		TRN4CPP::Simulation::Reservoir::Weights::Feedforward::Uniform::configure((unsigned long long)simulation_id, (float)a, (float)b, (float)sparsity);
	}
	catch (std::exception &e)
	{
		ERROR_LOGGER << e.what(); env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
	}
}
void Java_TRN4JAVA_Extended_00024Simulation_00024Reservoir_00024Weights_00024Readout_00024Custom_configure(JNIEnv *env, jclass jclazz, jlong simulation_id)
{
	TRACE_LOGGER;
	try
	{
		TRN4CPP::Simulation::Reservoir::Weights::Readout::Custom::configure((unsigned long long)simulation_id);
	}
	catch (std::exception &e)
	{
		ERROR_LOGGER << e.what(); env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
	}
}
void Java_TRN4JAVA_Extended_00024Simulation_00024Reservoir_00024Weights_00024Readout_00024Gaussian_configure(JNIEnv *env, jclass jclazz, jlong simulation_id, jfloat mu, jfloat sigma, jfloat sparsity)
{
	TRACE_LOGGER;
	try
	{
		TRN4CPP::Simulation::Reservoir::Weights::Readout::Gaussian::configure((unsigned long long)simulation_id, (float)mu, (float)sigma, (float)sparsity);
	}
	catch (std::exception &e)
	{
		ERROR_LOGGER << e.what(); env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
	}
}
void Java_TRN4JAVA_Extended_00024Simulation_00024Reservoir_00024Weights_00024Readout_00024Uniform_configure(JNIEnv *env, jclass jclazz, jlong simulation_id, jfloat a, jfloat b, jfloat sparsity)
{
	TRACE_LOGGER;
	try
	{
		TRN4CPP::Simulation::Reservoir::Weights::Readout::Uniform::configure((unsigned long long)simulation_id, (float)a, (float)b, (float)sparsity);
	}
	catch (std::exception &e)
	{
		ERROR_LOGGER << e.what(); env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
	}
}
void Java_TRN4JAVA_Extended_00024Simulation_00024Reservoir_00024Weights_00024Recurrent_00024Custom_configure(JNIEnv *env, jclass jclazz, jlong simulation_id)
{
	TRACE_LOGGER;
	try
	{
		TRN4CPP::Simulation::Reservoir::Weights::Recurrent::Custom::configure((unsigned long long)simulation_id);
	}
	catch (std::exception &e)
	{
		ERROR_LOGGER << e.what(); env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
	}
}
void Java_TRN4JAVA_Extended_00024Simulation_00024Reservoir_00024Weights_00024Recurrent_00024Gaussian_configure(JNIEnv *env, jclass jclazz, jlong simulation_id, jfloat mu, jfloat sigma, jfloat sparsity)
{
	TRACE_LOGGER;
	try
	{
		TRN4CPP::Simulation::Reservoir::Weights::Recurrent::Gaussian::configure((unsigned long long)simulation_id, (float)mu, (float)sigma, (float)sparsity);
	}
	catch (std::exception &e)
	{
		ERROR_LOGGER << e.what(); env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
	}
}
void Java_TRN4JAVA_Extended_00024Simulation_00024Reservoir_00024Weights_00024Recurrent_00024Uniform_configure(JNIEnv *env, jclass jclazz, jlong simulation_id, jfloat a, jfloat b, jfloat sparsity)
{
	TRACE_LOGGER;
	try
	{
		TRN4CPP::Simulation::Reservoir::Weights::Recurrent::Uniform::configure((unsigned long long)simulation_id, (float)a, (float)b, (float)sparsity);
	}
	catch (std::exception &e)
	{
		ERROR_LOGGER << e.what(); env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
	}
}
void Java_TRN4JAVA_Extended_00024Simulation_00024Reservoir_00024WidrowHoff_configure(JNIEnv *env, jclass jclazz, jlong simulation_id, jlong stimulus_size, jlong prediction_size, jlong reservoir_size, jfloat leak_rate, jfloat initial_state_scale, jfloat learning_rate, jlong seed, jlong batch_size, jlong mini_batch_size)
{
	TRACE_LOGGER;
	try
	{
		TRN4CPP::Simulation::Reservoir::WidrowHoff::configure((unsigned long long)simulation_id, (std::size_t)stimulus_size, (std::size_t)prediction_size, (std::size_t)reservoir_size, (float)leak_rate, (float)initial_state_scale, (float)learning_rate, (unsigned long)seed, (std::size_t)batch_size, (std::size_t)mini_batch_size);
	}
	catch (std::exception &e)
	{
		ERROR_LOGGER << e.what(); env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
	}
}
void Java_TRN4JAVA_Extended_00024Simulation_00024Scheduler_00024Custom_configure(JNIEnv *env, jclass jclazz, jlong simulation_id, jlong seed, jstring tag)
{
	TRACE_LOGGER;
	try
	{
		TRN4CPP::Simulation::Scheduler::Custom::configure((unsigned long long)simulation_id, (unsigned long)seed, TRN4JAVA::Convert::to_string(env, tag));
	}
	catch (std::exception &e)
	{
		ERROR_LOGGER << e.what(); env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
	}
}
void Java_TRN4JAVA_Extended_00024Simulation_00024Scheduler_00024Mutator_00024Custom_configure(JNIEnv *env, jclass jclazz, jlong simulation_id, jlong seed)
{
	TRACE_LOGGER;
	try
	{
		TRN4CPP::Simulation::Scheduler::Mutator::Custom::configure((unsigned long long)simulation_id, (unsigned long)seed);
	}
	catch (std::exception &e)
	{
		ERROR_LOGGER << e.what(); env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
	}
}
void Java_TRN4JAVA_Extended_00024Simulation_00024Scheduler_00024Mutator_00024Reverse_configure(JNIEnv *env, jclass jclazz, jlong simulation_id, jlong seed, jfloat rate, jlong size)
{
	TRACE_LOGGER;
	try
	{
		TRN4CPP::Simulation::Scheduler::Mutator::Reverse::configure((unsigned long long)simulation_id, (unsigned long)seed, (float)rate, (std::size_t)size);
	}
	catch (std::exception &e)
	{
		ERROR_LOGGER << e.what(); env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
	}
}
void Java_TRN4JAVA_Extended_00024Simulation_00024Scheduler_00024Mutator_00024Shuffle_configure(JNIEnv *env, jclass jclazz, jlong simulation_id, jlong seed)
{
	TRACE_LOGGER;
	try
	{
		TRN4CPP::Simulation::Scheduler::Mutator::Shuffle::configure((unsigned long long)simulation_id, (unsigned long)seed);
	}
	catch (std::exception &e)
	{
		ERROR_LOGGER << e.what(); env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
	}
}
void Java_TRN4JAVA_Extended_00024Simulation_00024Scheduler_00024Snippets_configure(JNIEnv *env, jclass jclazz, jlong simulation_id, jlong seed, jint snippets_size, jint time_budget, jfloat  learn_reverse_rate, jfloat generate_reverse_rate, jfloat learning_rate, jfloat discount, jstring tag)
{
	TRACE_LOGGER;
	try
	{
		TRN4CPP::Simulation::Scheduler::Snippets::configure((unsigned long long)simulation_id, (unsigned long)seed, (unsigned int)snippets_size, (unsigned int)time_budget, (float)learn_reverse_rate, (float)generate_reverse_rate, (float)learning_rate, (float)discount, TRN4JAVA::Convert::to_string(env, tag));
	}
	catch (std::exception &e)
	{
		ERROR_LOGGER << e.what(); env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
	}
}
void Java_TRN4JAVA_Extended_00024Simulation_00024Scheduler_00024Tiled_configure(JNIEnv *env, jclass jclazz, jlong simulation_id, jint epochs)
{
	TRACE_LOGGER;
	try
	{
		TRN4CPP::Simulation::Scheduler::Tiled::configure((unsigned long long)simulation_id, (unsigned int)epochs);
	}
	catch (std::exception &e)
	{
		ERROR_LOGGER << e.what(); env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
	}
}

