#include "stdafx.h"
#include "TRN4JAVA_Engine.h"
#include "TRN4JAVA_Engine_Backend.h"
#include "TRN4JAVA_Engine_Backend_Local.h"
#include "TRN4JAVA_Engine_Backend_Remote.h"
#include "TRN4JAVA_Simulation.h"
#include "TRN4JAVA_Simulation_Loop.h"
#include "TRN4JAVA_Simulation_Loop_Position.h"
#include "TRN4JAVA_Simulation_Loop_Stimulus.h"

#include "TLS_JNIEnv.h"
#include "TRN4CPP/Simplified.h"
#include "TRN4CPP/Custom.h"

static const char *LOOP_CALLBACK_SIGNATURE = "(JJJ[FJJ)V";

std::list<jobject>  loop_global_ref;
std::map<jobject, std::function<void(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols)>> loop_reply;

static inline jstring to_jstring(JNIEnv *env, const std::string &string)
{
	return env->NewStringUTF(string.c_str());
}
static inline std::string to_string(JNIEnv *env, jstring string)
{
	const char *cstr = env->GetStringUTFChars(string, NULL);
	std::string str(cstr);
	env->ReleaseStringUTFChars(string, cstr);

	return str;
}
static inline std::vector<std::string> to_string_vector(JNIEnv *env, jobjectArray array)
{
	jsize size = env->GetArrayLength(array);
	std::vector<std::string> vector(size);
	for (jsize k = 0; k < size; k++)
	{
		vector[k] = to_string(env, (jstring)env->GetObjectArrayElement(array, k));
	}
	return vector;
}

static inline jfloatArray to_jfloat_array(JNIEnv *env, const std::vector<float> &vector)
{
	jfloatArray result;
	auto size = vector.size();
	result = env->NewFloatArray(size);
	env->SetFloatArrayRegion(result, 0, size, &vector[0]);

	return result;
}

std::vector<float> to_float_vector(JNIEnv *env, jfloatArray array)
{
	jsize size = env->GetArrayLength(array);
	std::vector<float> vector(size);
	env->GetFloatArrayRegion(array, 0, size, &vector[0]);

	return vector;
}
std::vector<unsigned int> to_unsigned_int_vector(JNIEnv *env, jintArray array)
{
	jsize size = env->GetArrayLength(array);
	std::vector<jint> vector(size);
	env->GetIntArrayRegion(array, 0, size, &vector[0]);

	return std::vector<unsigned int>(vector.begin(), vector.end());
}

static inline jintArray to_jint_array(JNIEnv *env, const std::vector<int> &vector)
{
	jintArray result;
	auto size = vector.size();
	result = env->NewIntArray(size);
	std::vector<long> ivector(size);
	ivector.assign(vector.begin(), vector.end());
	env->SetIntArrayRegion(result, 0, size, &ivector[0]);

	return result;
}


/*
* Class:     TRN4JAVA_Engine
* Method:    initialize
* Signature: ()V
*/
JNIEXPORT void JNICALL Java_TRN4JAVA_Engine_initialize(JNIEnv *env, jclass jclass)
{
	try
	{
		// std::cout << __FUNCTION__ << std::endl;
		TRN4JAVA::init(env);
		TRN4CPP::Engine::initialize();
	}
	catch (std::exception &e)
	{
		env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
	}
}

/*
* Class:     TRN4JAVA_Engine
* Method:    uninitialize
* Signature: ()V
*/
JNIEXPORT void JNICALL Java_TRN4JAVA_Engine_uninitialize(JNIEnv *env, jclass jclass)
{
	try
	{
		// std::cout << __FUNCTION__ << std::endl;
		TRN4CPP::Engine::uninitialize();
	}
	catch (std::exception &e)
	{
		env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
	}
}

JNIEXPORT void JNICALL Java_TRN4JAVA_Engine_00024Backend_00024Local_initialize(JNIEnv *env, jclass jclass, jintArray indices)
{
	try
	{
		// std::cout << __FUNCTION__ << std::endl;
		TRN4JAVA::init(env);
		TRN4CPP::Engine::Backend::Local::initialize(to_unsigned_int_vector(env, indices));
	}
	catch (std::exception &e)
	{
		env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
	}
}
JNIEXPORT void JNICALL Java_TRN4JAVA_Engine_00024Backend_00024Remote_initialize(JNIEnv *env, jclass jclass, jstring host, jint port)
{
	try
	{
		// std::cout << __FUNCTION__ << std::endl;
		TRN4JAVA::init(env);
		TRN4CPP::Engine::Backend::Remote::initialize(to_string(env, host), (unsigned short)port);
	}
	catch (std::exception &e)
	{
		env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
	}
}

JNIEXPORT void JNICALL Java_TRN4JAVA_Simulation_declare(JNIEnv *env, jclass jclass, jstring label, jfloatArray elements, jlong rows, jlong cols, jstring tag)
{
	try
	{
		TRN4CPP::Simulation::declare(to_string(env, label), to_float_vector(env, elements), (std::size_t)rows, (std::size_t)cols, to_string(env, tag));
	}
	catch (std::exception &e)
	{
		env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
	}
}

JNIEXPORT void JNICALL Java_TRN4JAVA_Simulation_compute(JNIEnv *env, jclass jclass, jstring scenario_filename)
{
	try
	{
		TRN4CPP::Simulation::compute(to_string(env, scenario_filename));
	}
	catch (std::exception &e)
	{
		env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
	}
}

JNIEXPORT void JNICALL Java_TRN4JAVA_Simulation_00024Loop_notify(JNIEnv *env, jobject loop, jlong id, jlong trial, jlong evaluation, jfloatArray elements, jlong rows, jlong cols)
{
	TRN4JAVA::getJNIEnv()->MonitorEnter(loop);
	try
	{
	
		// std::cout << __FUNCTION__ << std::endl;
		auto it = std::find_if(std::begin(loop_global_ref), std::end(loop_global_ref), [loop, env](const jobject ref)
		{
			return env->IsSameObject(ref, loop);
		});

		if (it == loop_global_ref.end())
			throw std::runtime_error("Reply loop object not found");
		loop_reply[*it]((std::size_t)id, (std::size_t)trial, (std::size_t)evaluation, to_float_vector(env, elements), (std::size_t)rows, (std::size_t)cols);	
	}
	catch (std::exception &e)
	{
		env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
	}
	TRN4JAVA::getJNIEnv()->MonitorExit(loop);
}

JNIEXPORT void JNICALL Java_TRN4JAVA_Simulation_00024Loop_00024Position_install(JNIEnv *env, jclass jclass, jobject loop)
{
	try
	{
		// std::cout << __FUNCTION__ << std::endl;
		jmethodID callback = env->GetMethodID(env->GetObjectClass(loop), "callback", LOOP_CALLBACK_SIGNATURE);
		if (callback == 0)
		{
			if (env->ExceptionCheck())
			{
				env->ExceptionDescribe();
				env->ExceptionClear();
			}
			throw std::invalid_argument("Can't find JNI method");
		}
		auto loop_ref = env->NewGlobalRef(loop);
		loop_global_ref.push_back(loop_ref);

		TRN4CPP::Simulation::Loop::Position::install
		(
			[loop_ref, callback](const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &prediction, const std::size_t &rows, const std::size_t &cols)
			{
				auto env = TRN4JAVA::getJNIEnv();
				env->CallVoidMethod(loop_ref, callback, (jlong)id, (jlong)(trial), (jlong)evaluation, to_jfloat_array(env, prediction), (jlong)rows, (jlong)cols);
			},
			loop_reply[loop_ref]
		);
	}
	catch (std::exception &e)
	{
		env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
	}
}

JNIEXPORT void JNICALL Java_TRN4JAVA_Simulation_00024Loop_00024Stimulus_install(JNIEnv *env, jclass jclass, jobject loop)
{
	try
	{
		// std::cout << __FUNCTION__ << std::endl;
		jmethodID callback = env->GetMethodID(env->GetObjectClass(loop), "callback", LOOP_CALLBACK_SIGNATURE);
		if (callback == 0)
		{
			if (env->ExceptionCheck())
			{
				env->ExceptionDescribe();
				env->ExceptionClear();
			}
			throw std::invalid_argument("Can't find JNI method");
		}
		auto loop_ref = env->NewGlobalRef(loop);
		loop_global_ref.push_back(loop_ref);

		TRN4CPP::Simulation::Loop::Stimulus::install
		(
			[loop_ref, callback](const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &prediction, const std::size_t &rows, const std::size_t &cols)
			{
				auto env = TRN4JAVA::getJNIEnv();
				env->CallVoidMethod(loop_ref, callback, (jlong)id, (jlong)(trial), (jlong)evaluation, to_jfloat_array(env, prediction), (jlong)rows, (jlong)cols);
			},
			loop_reply[loop_ref]
		);
	}
	catch (std::exception &e)
	{
		env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
	}
}