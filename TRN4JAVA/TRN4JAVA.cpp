#include "stdafx.h"
#include "TRN4JAVA_Engine.h"
#include "TRN4JAVA_Engine_Backend.h"
#include "TRN4JAVA_Engine_Backend_Local.h"
#include "TRN4JAVA_Engine_Backend_Remote.h"
#include "TRN4JAVA_Engine_Events.h"
#include "TRN4JAVA_Engine_Events_Ack.h"
#include "TRN4JAVA_Engine_Events_Allocated.h"
#include "TRN4JAVA_Engine_Events_Configured.h"
#include "TRN4JAVA_Engine_Events_Deallocated.h"
#include "TRN4JAVA_Engine_Events_Primed.h"
#include "TRN4JAVA_Engine_Events_Processor.h"
#include "TRN4JAVA_Engine_Events_Completed.h"
#include "TRN4JAVA_Engine_Events_Tested.h"
#include "TRN4JAVA_Engine_Events_Trained.h"

#include "TRN4JAVA_Engine_Execution.h"

#include "TRN4JAVA_Simulation.h"

#include "TRN4JAVA_Simulation_Loop.h"
#include "TRN4JAVA_Simulation_Loop_Copy.h"
#include "TRN4JAVA_Simulation_Loop_Custom.h"
#include "TRN4JAVA_Simulation_Loop_Position.h"
#include "TRN4JAVA_Simulation_Loop_SpatialFilter.h"
#include "TRN4JAVA_Simulation_Loop_Stimulus.h"

#include "TRN4JAVA_Simulation_Measurement.h"
#include "TRN4JAVA_Simulation_Measurement_Position.h"
#include "TRN4JAVA_Simulation_Measurement_Position_FrechetDistance.h"
#include "TRN4JAVA_Simulation_Measurement_Position_MeanSquareError.h"
#include "TRN4JAVA_Simulation_Measurement_Position_Raw.h"
#include "TRN4JAVA_Simulation_Measurement_Processed.h"
#include "TRN4JAVA_Simulation_Measurement_Raw.h"
#include "TRN4JAVA_Simulation_Measurement_Readout.h"
#include "TRN4JAVA_Simulation_Measurement_Readout_FrechetDistance.h"
#include "TRN4JAVA_Simulation_Measurement_Readout_MeanSquareError.h"
#include "TRN4JAVA_Simulation_Measurement_Readout_Raw.h"

#include "TRN4JAVA_Simulation_Recording.h"
#include "TRN4JAVA_Simulation_Recording_Performances.h"
#include "TRN4JAVA_Simulation_Recording_Scheduling.h"
#include "TRN4JAVA_Simulation_Recording_States.h"
#include "TRN4JAVA_Simulation_Recording_Weights.h"

#include "TRN4JAVA_Simulation_Reservoir.h"
#include "TRN4JAVA_Simulation_Reservoir_Weights.h"
#include "TRN4JAVA_Simulation_Reservoir_Weights_Feedback.h"
#include "TRN4JAVA_Simulation_Reservoir_Weights_Feedback_Custom.h"
#include "TRN4JAVA_Simulation_Reservoir_Weights_Feedback_Gaussian.h"
#include "TRN4JAVA_Simulation_Reservoir_Weights_Feedback_Uniform.h"
#include "TRN4JAVA_Simulation_Reservoir_Weights_Feedforward_Custom.h"
#include "TRN4JAVA_Simulation_Reservoir_Weights_Feedforward_Gaussian.h"
#include "TRN4JAVA_Simulation_Reservoir_Weights_Feedforward_Uniform.h"
#include "TRN4JAVA_Simulation_Reservoir_Weights_Readout.h"
#include "TRN4JAVA_Simulation_Reservoir_Weights_Readout_Custom.h"
#include "TRN4JAVA_Simulation_Reservoir_Weights_Readout_Gaussian.h"
#include "TRN4JAVA_Simulation_Reservoir_Weights_Readout_Uniform.h"
#include "TRN4JAVA_Simulation_Reservoir_Weights_Recurrent.h"
#include "TRN4JAVA_Simulation_Reservoir_Weights_Recurrent_Custom.h"
#include "TRN4JAVA_Simulation_Reservoir_Weights_Recurrent_Gaussian.h"
#include "TRN4JAVA_Simulation_Reservoir_Weights_Recurrent_Uniform.h"
#include "TRN4JAVA_Simulation_Reservoir_WidrowHoff.h"

#include "TRN4JAVA_Simulation_Scheduler.h"
#include "TRN4JAVA_Simulation_Scheduler_Mutator.h"
#include "TRN4JAVA_Simulation_Scheduler_Mutator_Custom.h"
#include "TRN4JAVA_Simulation_Scheduler_Mutator_Reverse.h"
#include "TRN4JAVA_Simulation_Scheduler_Mutator_Shuffle.h"
#include "TRN4JAVA_Simulation_Scheduler_Snippets.h"
#include "TRN4JAVA_Simulation_Scheduler_Tiled.h"

#include "TLS_JNIEnv.h"
#include "TRN4CPP/Simplified.h"
#include "TRN4CPP/Custom.h"
#include "TRN4CPP/Callbacks.h"
#include "TRN4CPP/Extended.h"

static const char *EVENT_CALLBACK_SIGNATURE = "()V";
static const char *EVENT_ACK_CALLBACK_SIGNATURE = "(JJZLjava/lang/String;)V";
static const char *EVENT_PROCESSOR_CALLBACK_SIGNATURE = "(ILjava/lang/String;ILjava/lang/String;)V";
static const char *EVENT_SIMULATION_STATE_CALLBACK_SIGNATURE = "(J)V";
static const char *EVENT_SIMULATION_ALLOCATION_CALLBACK_SIGNATURE = "(JI)V";
static const char *LOOP_CALLBACK_SIGNATURE = "(JJJ[FJJ)V";
static const char *PROCESSED_CALLBACK_SIGNATURE = "(JJJ[FJJ)V";
static const char *RAW_CALLBACK_SIGNATURE = "(JJJ[F[F[FJJJJ)V";
static const char *PERFORMANCES_CALLBACK_SIGNATURE = "(JJJLjava/lang/String;FF)V";
static const char *STATES_CALLBACK_SIGNATURE = "(JLjava/lang/String;Ljava/lang/String;JJJ[FJJ)V";
static const char *WEIGHTS_CALLBACK_SIGNATURE = "(JLjava/lang/String;Ljava/lang/String;JJ[FJJ)V";
static const char *SCHEDULING_CALLBACK_SIGNATURE = "(JJ[I[I)V";
static const char *SCHEDULER_CALLBACK_SIGNATURE = "(JJJ[FJJ[I[I)V";

std::list<jobject>  events_global_ref;
std::list<jobject>  loop_global_ref;
std::list<jobject>  processed_global_ref;
std::list<jobject>  raw_global_ref;
std::list<jobject>  recording_global_ref;
std::list<jobject>  scheduler_global_ref;
std::list<jobject>  mutator_global_ref;
std::list<jobject>  weights_global_ref;

std::map<jobject, std::function<void(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols)>> loop_reply;
std::map<jobject, std::function<void(const unsigned long long &id, const std::size_t &trial, const std::vector<int> &offsets, const std::vector<int> &durations)>> scheduler_reply;
std::map<jobject, std::function<void(const unsigned long long &id, const std::size_t &trial, const std::vector<int> &offsets, const std::vector<int> &durations)>> mutator_reply;
std::map<jobject, std::function<void(const unsigned long long &id, const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)>> weights_reply;

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
static std::vector<float> to_float_vector(JNIEnv *env, jfloatArray array)
{
	jsize size = env->GetArrayLength(array);
	std::vector<float> vector(size);
	env->GetFloatArrayRegion(array, 0, size, &vector[0]);

	return vector;
}
static std::vector<int> to_int_vector(JNIEnv *env, jintArray array)
{
	jsize size = env->GetArrayLength(array);
	std::vector<jint> vector(size);
	env->GetIntArrayRegion(array, 0, size, &vector[0]);

	return std::vector<int>(vector.begin(), vector.end());
}
static std::vector<unsigned int> to_unsigned_int_vector(JNIEnv *env, jintArray array)
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

template<typename Installer, typename ... CallbackArgs>
static void install(JNIEnv *env, const jobject object, const char *signature, std::list<jobject> &global_ref, Installer &installer, const std::function<void(jobject, jmethodID, CallbackArgs ...)> &callback)
{
	try
	{
		jmethodID callback_id = env->GetMethodID(env->GetObjectClass(object), "callback", signature);
		if (callback_id == 0)
		{
			if (env->ExceptionCheck())
			{
				env->ExceptionDescribe();
				env->ExceptionClear();
			}
			throw std::invalid_argument("Can't find JNI method 'callback" + std::string(signature));
		}
		auto ref = env->NewGlobalRef(object);
		global_ref.push_back(ref);


		installer
		(
			[=](CallbackArgs ... args)
		{
			callback(ref, callback_id, args...);
		}
		);
	}
	catch (std::exception &e)
	{
		env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
	}
}
template<typename Notify, typename Installer, typename ... CallbackArgs>
static void install(JNIEnv *env, const jobject object, const char *signature, std::list<jobject> &global_ref, Installer &installer, const std::function<void(jobject, jmethodID, CallbackArgs ...)> &callback, Notify &notify)
{
	try
	{
		jmethodID callback_id = env->GetMethodID(env->GetObjectClass(object), "callback", signature);
		if (callback_id == 0)
		{
			if (env->ExceptionCheck())
			{
				env->ExceptionDescribe();
				env->ExceptionClear();
			}
			throw std::invalid_argument("Can't find JNI method 'callback" + std::string(signature));
		}
		auto ref = env->NewGlobalRef(object);
		global_ref.push_back(ref);


		installer
		(
			[=](CallbackArgs ... args)
		{
			callback(ref, callback_id, args...);
		}
			,
			notify[ref]
			);
	}
	catch (std::exception &e)
	{
		env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
	}
}
template<typename Notify1, typename Notify2, typename Installer, typename ... Callback1Args, typename ... Callback2Args>
static void install(JNIEnv *env, const jobject object1, const jobject object2, const char *signature, std::list<jobject> &global_ref, Installer &installer,
	const std::function<void(jobject, jmethodID, Callback1Args ...)> &callback1, Notify1 &notify1,
	const std::function<void(jobject, jmethodID, Callback2Args ...)> &callback2, Notify2 &notify2)
{
	try
	{
		jmethodID callback_id1 = env->GetMethodID(env->GetObjectClass(object1), "callback", signature);
		jmethodID callback_id2 = env->GetMethodID(env->GetObjectClass(object2), "callback", signature);
		if (callback_id1 == 0)
		{
			if (env->ExceptionCheck())
			{
				env->ExceptionDescribe();
				env->ExceptionClear();
			}
			throw std::invalid_argument("Can't find JNI method 'callback" + std::string(signature));
		}
		auto ref1 = env->NewGlobalRef(object1);
		global_ref.push_back(ref1);
		auto ref2 = env->NewGlobalRef(object2);
		global_ref.push_back(ref2);

		installer
		(
			[=](Callback1Args ... args)
			{
				callback1(ref1, callback_id1, args...);
			}
			,
			notify1[ref1],
			[=](Callback2Args ... args)
			{
				callback2(ref2, callback_id2, args...);
			}
			,
			notify2[ref2]
			);
	}
	catch (std::exception &e)
	{
		env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
	}
}
template<typename Notify, typename ... NotifyArgs>
static void notify(JNIEnv *env, jobject object, const std::list<jobject> &global_ref, Notify &notify, NotifyArgs ... args)
{
	env->MonitorEnter(object);
	try
	{
		auto it = std::find_if(std::begin(global_ref), std::end(global_ref), [=](const jobject ref) 
		{
			return env->IsSameObject(object, ref);
		});

		if (it == global_ref.end())
			throw std::runtime_error("Reply object not found");
		notify[*it](args...);
	}
	catch (std::exception &e)
	{
		env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
	}
	env->MonitorExit(object);
}

// For generic types that are functors, delegate to its 'operator()'
template <typename T>
struct function_traits
	: public function_traits<decltype(&T::operator())>
{
};
// for pointers to member function
template <typename ClassType, typename ReturnType, typename... Args>
struct function_traits<ReturnType(ClassType::*)(Args...) const>
{
	enum { arity = sizeof...(Args) };
	typedef std::function<ReturnType(Args...)> f_type;
};
// for pointers to member function
template <typename ClassType, typename ReturnType, typename... Args>
struct function_traits<ReturnType(ClassType::*)(Args...) >
{
	enum { arity = sizeof...(Args) };
	typedef std::function<ReturnType(Args...)> f_type;
};
// for function pointers
template <typename ReturnType, typename... Args>
struct function_traits<ReturnType(*)(Args...)>
{
	enum { arity = sizeof...(Args) };
	typedef std::function<ReturnType(Args...)> f_type;
};
template <typename L>
static typename function_traits<L>::f_type make_function(L l)
{
	return (typename function_traits<L>::f_type)(l);
}
//handles bind & multiple function call operator()'s
template<typename ReturnType, typename... Args, class T>
auto make_function(T&& t) -> std::function<decltype(ReturnType(t(std::declval<Args>()...)))(Args...)>
{
	return{ std::forward<T>(t) };
}
//handles explicit overloads
template<typename ReturnType, typename... Args>
auto make_function(ReturnType(*p)(Args...)) -> std::function<ReturnType(Args...)>
{
	return{ p };
}
//handles explicit overloads
template<typename ReturnType, typename... Args, typename ClassType>
auto make_function(ReturnType(ClassType::*p)(Args...)) -> std::function<ReturnType(Args...)>
{
	return{ p };
}

static void event_callback(jobject object, jmethodID method)
{
	auto env = TRN4JAVA::getJNIEnv();
	env->CallVoidMethod(object, method);
}
static void event_ack_callback(jobject object, jmethodID method, const unsigned long long &id, const std::size_t &counter, const bool &success, const std::string &cause)
{
	auto env = TRN4JAVA::getJNIEnv();
	env->CallVoidMethod(object, method, (jlong)id, (jlong)counter, (jboolean)success, to_jstring(env, cause));
}
static void event_simulation_state_callback(jobject object, jmethodID method, const unsigned long long &id)
{
	auto env = TRN4JAVA::getJNIEnv();
	env->CallVoidMethod(object, method, (jlong)id);
}
static void event_processor_callback(jobject object, jmethodID method, const int &rank, const std::string &host, const unsigned int &index, const std::string &name)
{
	auto env = TRN4JAVA::getJNIEnv();
	env->CallVoidMethod(object, method, (jint)rank, to_jstring(env, host), (jint)index, to_jstring(env, name));
}
static void event_simulation_allocation_callback(jobject object, jmethodID method, const unsigned long long &id, const int &rank)
{
	auto env = TRN4JAVA::getJNIEnv();
	env->CallVoidMethod(object, method, (jlong)id, (jint)rank);
}
static void measurement_matrix_callback(jobject object, jmethodID method, const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &prediction, const std::size_t &rows, const std::size_t &cols)
{
	auto env = TRN4JAVA::getJNIEnv();
	env->CallVoidMethod(object, method, (jlong)id, (jlong)(trial), (jlong)evaluation, to_jfloat_array(env, prediction), (jlong)rows, (jlong)cols);
}
static void measurement_raw_callback(jobject object, jmethodID method, const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &primed, const std::vector<float> &predicted, const std::vector<float> &expected, const std::size_t &preamble, const std::size_t &pages, const std::size_t &rows, const std::size_t &cols)
{
	auto env = TRN4JAVA::getJNIEnv();
	env->CallVoidMethod(object, method, (jlong)id, (jlong)(trial), (jlong)evaluation, to_jfloat_array(env, primed), to_jfloat_array(env, predicted), to_jfloat_array(env, expected), (jlong)preamble, (jlong)pages, (jlong)rows, (jlong)cols);
}
static void recording_performances_callback(jobject object, jmethodID method, const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::string &phase, const float &cycles_per_second, const float &gflops_per_second)
{
	auto env = TRN4JAVA::getJNIEnv();
	env->CallVoidMethod(object, method, (jlong)id, (jlong)(trial), (jlong)evaluation, to_jstring(env, phase), (jfloat)cycles_per_second, (jfloat)gflops_per_second);
}
static void recording_states_callback(jobject object, jmethodID method, const unsigned long long &id, const std::string &phase, const std::string &label, const std::size_t &batch, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &samples, const std::size_t &rows, const std::size_t &cols)
{
	auto env = TRN4JAVA::getJNIEnv();
	env->CallVoidMethod(object, method, (jlong)id, to_jstring(env, phase), to_jstring(env, label), (jlong)(batch), (jlong)(trial), (jlong)evaluation, to_jfloat_array(env, samples), (jlong)(rows), (jlong)(cols));
}
static void recording_weights_callback(jobject object, jmethodID method, const unsigned long long &id, const std::string &phase, const std::string &label, const std::size_t &batch, const std::size_t &trial, const std::vector<float> &weights, const std::size_t &rows, const std::size_t &cols)
{
	auto env = TRN4JAVA::getJNIEnv();
	env->CallVoidMethod(object, method, (jlong)id, to_jstring(env, phase), to_jstring(env, label), (jlong)(batch), (jlong)(trial), to_jfloat_array(env, weights), (jlong)(rows), (jlong)(cols));
}
static void recording_scheduling_callback(jobject object, jmethodID method, const unsigned long long &id, const std::size_t &trial, const std::vector<int> &offsets, const std::vector<int> &durations)
{
	auto env = TRN4JAVA::getJNIEnv();
	env->CallVoidMethod(object, method, (jlong)id, (jlong)(trial), to_jint_array(env, offsets), to_jint_array(env, durations));
}
static void recording_scheduler_callback(jobject object, jmethodID method, const unsigned long long &id, const unsigned long &seed, const std::size_t &trial, const std::vector<float> &elements, const std::size_t &rows, const std::size_t &cols, const std::vector<int> &offsets, const std::vector<int> &durations)
{
	auto env = TRN4JAVA::getJNIEnv();
	env->CallVoidMethod(object, method, (jlong)id, (jlong)seed, (jlong)trial, to_jfloat_array(env, elements), (jlong)rows, (jlong)cols, to_jint_array(env, offsets), to_jint_array(env, durations));
}
static void custom_mutator_callback(jobject object, jmethodID method, const unsigned long long &id, const unsigned long &seed, const std::size_t &trial, const std::vector<int> &offsets, const std::vector<int> &durations)
{
	auto env = TRN4JAVA::getJNIEnv();
	env->CallVoidMethod(object, method, (jlong)id, (jlong)seed, (jlong)trial, to_jint_array(env, offsets), to_jint_array(env, durations));
}
static void custom_weights_callback(jobject object, jmethodID method, const unsigned long long &id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)
{
	auto env = TRN4JAVA::getJNIEnv();
	env->CallVoidMethod(object, method, (jlong)id, (jlong)seed, (jlong)matrices, (jlong)rows, (jlong)cols);
}
static void custom_scheduler_callback(jobject object, jmethodID method, const unsigned long long &id, const unsigned long &seed, const std::size_t &trial, const std::vector<float> &elements, const std::size_t &rows, const std::size_t &cols, const std::vector<int> &offsets, const std::vector<int> &durations)
{
	auto env = TRN4JAVA::getJNIEnv();
	env->CallVoidMethod(object, method, (jlong)id, (jlong)seed, (jlong)trial, to_jfloat_array(env, elements), (jlong)rows, (jlong)cols, to_jint_array(env, offsets), to_jint_array(env, durations));
}
void Java_TRN4JAVA_Engine_initialize(JNIEnv *env, jclass jclass)
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
// std::cout << __FUNCTION__ << std::endl;
}
void Java_TRN4JAVA_Engine_uninitialize(JNIEnv *env, jclass jclass)
{
	try
	{
		 //std::cout << __FUNCTION__ << std::endl;
		TRN4CPP::Engine::uninitialize();
	}
	catch (std::exception &e)
	{
		env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
	}
// std::cout << __FUNCTION__ << std::endl;
}
void Java_TRN4JAVA_Engine_00024Backend_00024Local_initialize(JNIEnv *env, jclass jclass, jintArray indices)
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
// std::cout << __FUNCTION__ << std::endl;
}
void Java_TRN4JAVA_Engine_00024Backend_00024Remote_initialize(JNIEnv *env, jclass jclass, jstring host, jint port)
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
// std::cout << __FUNCTION__ << std::endl;
}

void Java_TRN4JAVA_Engine_00024Events_00024Ack_install(JNIEnv *env, jclass jclass, jobject ack)
{
	install(env, ack, EVENT_ACK_CALLBACK_SIGNATURE, events_global_ref,
		TRN4CPP::Engine::Events::Ack::install,
		make_function(event_ack_callback));
}
void Java_TRN4JAVA_Engine_00024Events_00024Allocated_install(JNIEnv *env, jclass jclass, jobject allocated)
{
	install(env, allocated, EVENT_SIMULATION_ALLOCATION_CALLBACK_SIGNATURE, events_global_ref,
		TRN4CPP::Engine::Events::Allocated::install,
		make_function(event_simulation_allocation_callback));
}
void Java_TRN4JAVA_Engine_00024Events_00024Configured_install(JNIEnv *env, jclass jclass, jobject configured)
{
	install(env, configured, EVENT_SIMULATION_STATE_CALLBACK_SIGNATURE, events_global_ref,
		TRN4CPP::Engine::Events::Configured::install,
		make_function(event_simulation_state_callback));
}
void Java_TRN4JAVA_Engine_00024Events_00024Deallocated_install(JNIEnv *env, jclass jclass, jobject deallocated)
{
	install(env, deallocated, EVENT_SIMULATION_ALLOCATION_CALLBACK_SIGNATURE, events_global_ref,
		TRN4CPP::Engine::Events::Deallocated::install,
		make_function(event_simulation_allocation_callback));
}
void Java_TRN4JAVA_Engine_00024Events_00024Primed_install(JNIEnv *env, jclass jclass, jobject primed)
{
	install(env, primed, EVENT_SIMULATION_STATE_CALLBACK_SIGNATURE, events_global_ref,
		TRN4CPP::Engine::Events::Primed::install,
		make_function(event_simulation_state_callback));
}
void Java_TRN4JAVA_Engine_00024Events_00024Processor_install(JNIEnv *env, jclass jclass, jobject processor)
{
	install(env, processor, EVENT_PROCESSOR_CALLBACK_SIGNATURE, events_global_ref,
		TRN4CPP::Engine::Events::Processor::install,
		make_function(event_processor_callback));
}
void Java_TRN4JAVA_Engine_00024Events_00024Completed_install(JNIEnv *env, jclass jclass, jobject completed)
{
	install(env, completed, EVENT_CALLBACK_SIGNATURE, events_global_ref,
		TRN4CPP::Engine::Events::Completed::install,
		make_function(event_callback));
}
void Java_TRN4JAVA_Engine_00024Events_00024Tested_install(JNIEnv *env, jclass jclass, jobject tested)
{
	install(env, tested, EVENT_SIMULATION_STATE_CALLBACK_SIGNATURE, events_global_ref,
		TRN4CPP::Engine::Events::Tested::install,
		make_function(event_simulation_state_callback));
}
void Java_TRN4JAVA_Engine_00024Events_00024Trained_install(JNIEnv *env, jclass jclass, jobject trained)
{
	install(env, trained, EVENT_SIMULATION_STATE_CALLBACK_SIGNATURE, events_global_ref,
		TRN4CPP::Engine::Events::Trained::install,
		make_function(event_simulation_state_callback));
}
void Java_TRN4JAVA_Engine_00024Execution_run(JNIEnv *env, jclass jclass)
{
	try
	{
		TRN4CPP::Engine::Execution::run();
	}
	catch (std::exception &e)
	{
		env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
	}
// std::cout << __FUNCTION__ << std::endl;
}
void Java_TRN4JAVA_Simulation_declare(JNIEnv *env, jclass jclass, jstring label, jfloatArray elements, jlong rows, jlong cols, jstring tag)
{
	try
	{
		TRN4CPP::Simulation::declare(to_string(env, label), to_float_vector(env, elements), (std::size_t)rows, (std::size_t)cols, to_string(env, tag));
	}
	catch (std::exception &e)
	{
		env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
	}
// std::cout << __FUNCTION__ << std::endl;
}
void Java_TRN4JAVA_Simulation_compute(JNIEnv *env, jclass jclass, jstring scenario_filename)
{
	try
	{
		TRN4CPP::Simulation::compute(to_string(env, scenario_filename));
	}
	catch (std::exception &e)
	{
		env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
	}
// std::cout << __FUNCTION__ << std::endl;
}
void Java_TRN4JAVA_Simulation_allocate(JNIEnv *env, jclass jclass, jlong id)
{
	try
	{
		TRN4CPP::Simulation::allocate((unsigned long long)id);
	}
	catch (std::exception &e)
	{
		env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
	}
// std::cout << __FUNCTION__ << std::endl;
 }
void Java_TRN4JAVA_Simulation_deallocate(JNIEnv *env, jclass jclass, jlong id)
 {
	 try
	 {
		 TRN4CPP::Simulation::deallocate((unsigned long long)id);
	 }
	 catch (std::exception &e)
	 {
		 env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
	 }
	// std::cout << __FUNCTION__ << std::endl;
 }
void Java_TRN4JAVA_Simulation_train(JNIEnv *env, jclass jclass, jlong id, jstring label, jstring incoming, jstring expected)
{
	try
	{
		TRN4CPP::Simulation::train((unsigned long long)id, to_string(env, label), to_string(env, incoming), to_string(env, expected));
	}
	catch (std::exception &e)
	{
		env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
	}
// std::cout << __FUNCTION__ << std::endl;
}
void Java_TRN4JAVA_Simulation_test(JNIEnv *env, jclass jclass, jlong id, jstring sequence, jstring incoming, jstring expected, jint preamble, jboolean autonomous, jint supplementary_generations) 
{
	try
	{
		TRN4CPP::Simulation::test((unsigned long long)id, to_string(env, sequence), to_string(env, incoming), to_string(env, expected), (unsigned int)preamble, (bool)autonomous, (unsigned int)supplementary_generations);
	}
	catch (std::exception &e)
	{
		env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
	}
// std::cout << __FUNCTION__ << std::endl;
}
void Java_TRN4JAVA_Simulation_declare_1sequence(JNIEnv *env, jclass jclass, jlong id, jstring label, jstring tag, jfloatArray sequence, jlong observations)
{
	try
	{
		TRN4CPP::Simulation::declare_sequence((unsigned long long)id, to_string(env, label), to_string(env, tag), to_float_vector(env, sequence), observations);
	}
	catch (std::exception &e)
	{
		env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
	}
// std::cout << __FUNCTION__ << std::endl;
}
void Java_TRN4JAVA_Simulation_declare_1set(JNIEnv *env, jclass jclass, jlong id, jstring label, jstring tag, jobjectArray labels)
{
	try
	{
		TRN4CPP::Simulation::declare_set((unsigned long long)id, to_string(env, label), to_string(env, tag), to_string_vector(env, labels));
	}
	catch (std::exception &e)
	{
		env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
	}
// std::cout << __FUNCTION__ << std::endl;
}
void Java_TRN4JAVA_Simulation_configure_1begin(JNIEnv *env, jclass jclass, jlong id) 
{
	try
	{
		TRN4CPP::Simulation::configure_begin((unsigned long long)id);
	}
	catch (std::exception &e)
	{
		env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
	}
// std::cout << __FUNCTION__ << std::endl;
}
void Java_TRN4JAVA_Simulation_configure_1end(JNIEnv *env, jclass jclass, jlong id) 
{
	try
	{
		TRN4CPP::Simulation::configure_end((unsigned long long)id);
	}
	catch (std::exception &e)
	{
		env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
	}
// std::cout << __FUNCTION__ << std::endl;
}
void Java_TRN4JAVA_Simulation_00024Loop_00024Copy_configure(JNIEnv *env, jclass jclass, jlong id, jlong batch_size, jlong stimulus_size) 
{
	try
	{
		TRN4CPP::Simulation::Loop::Copy::configure((unsigned long long)id, (std::size_t)batch_size, (std::size_t)stimulus_size);
	}
	catch (std::exception &e)
	{
		env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
	}
// std::cout << __FUNCTION__ << std::endl;
}
void Java_TRN4JAVA_Simulation_00024Loop_00024Custom_configure(JNIEnv *env, jclass jclass, jlong id, jlong batch_size, jlong stimulus_size, jobject stimulus)
{
	install(env, stimulus, LOOP_CALLBACK_SIGNATURE, loop_global_ref, 
		std::bind(&TRN4CPP::Simulation::Loop::Custom::configure, (unsigned long long)id, (std::size_t)batch_size, (std::size_t)stimulus_size, std::placeholders::_1, std::placeholders::_2), 
		make_function(measurement_matrix_callback), loop_reply);
// std::cout << __FUNCTION__ << std::endl;
}
void Java_TRN4JAVA_Simulation_00024Loop_00024SpatialFilter_configure(JNIEnv *env, jclass jclass, jlong id, jlong batch_size, jlong stimulus_size, jlong seed, jobject position, jobject stimulus, jlong rows, jlong cols, jfloat x_min, jfloat x_max, jfloat y_min, jfloat y_max, jfloatArray response, jfloat sigma, jfloat radius, jfloat scale, jstring tag)
{
	install(env, position, stimulus, LOOP_CALLBACK_SIGNATURE, loop_global_ref,
		std::bind(&TRN4CPP::Simulation::Loop::SpatialFilter::configure, (unsigned long long)id, (std::size_t)batch_size, (std::size_t)stimulus_size, (unsigned long)seed, 
			std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4,
		(std::size_t)rows, (std::size_t)cols, std::make_pair((float)x_min, (float)x_max), std::make_pair((float)y_min, (float)y_max), to_float_vector(env, response), (float)sigma, (float)radius, (float)scale, to_string(env, tag)),
		make_function(measurement_matrix_callback), loop_reply,
		make_function(measurement_matrix_callback), loop_reply
		);
// std::cout << __FUNCTION__ << std::endl;
}
void Java_TRN4JAVA_Simulation_00024Measurement_00024Position_00024FrechetDistance_configure(JNIEnv *env, jclass jclass, jlong id, jlong batch_size, jobject processed)
{
	install(env, processed, PROCESSED_CALLBACK_SIGNATURE, processed_global_ref, 
		std::bind(&TRN4CPP::Simulation::Measurement::Position::FrechetDistance::configure, (unsigned long long)id, (std::size_t)batch_size, std::placeholders::_1),
		make_function(measurement_matrix_callback));
// std::cout << __FUNCTION__ << std::endl;
}
void Java_TRN4JAVA_Simulation_00024Measurement_00024Position_00024MeanSquareError_configure(JNIEnv *env, jclass jclass, jlong id, jlong batch_size, jobject processed)
{
	install(env, processed, PROCESSED_CALLBACK_SIGNATURE, processed_global_ref,
		std::bind(&TRN4CPP::Simulation::Measurement::Position::MeanSquareError::configure, (unsigned long long)id, (std::size_t)batch_size, std::placeholders::_1),
		make_function(measurement_matrix_callback));
// std::cout << __FUNCTION__ << std::endl;
}
void Java_TRN4JAVA_Simulation_00024Measurement_00024Position_00024Raw_configure(JNIEnv *env, jclass jclass, jlong id, jlong batch_size, jobject raw)
{
	install(env, raw, RAW_CALLBACK_SIGNATURE, raw_global_ref,
		std::bind(&TRN4CPP::Simulation::Measurement::Position::Custom::configure, (unsigned long long)id, (std::size_t)batch_size, std::placeholders::_1),
		make_function(measurement_raw_callback));
// std::cout << __FUNCTION__ << std::endl;
}
void Java_TRN4JAVA_Simulation_00024Measurement_00024Readout_00024FrechetDistance_configure(JNIEnv *env, jclass jclass, jlong id, jlong batch_size, jobject processed)
{
	install(env, processed, PROCESSED_CALLBACK_SIGNATURE, processed_global_ref,
		std::bind(&TRN4CPP::Simulation::Measurement::Readout::FrechetDistance::configure, (unsigned long long)id, (std::size_t)batch_size, std::placeholders::_1),
		make_function(measurement_matrix_callback));
// std::cout << __FUNCTION__ << std::endl;
}
void Java_TRN4JAVA_Simulation_00024Measurement_00024Readout_00024MeanSquareError_configure(JNIEnv *env, jclass jclass, jlong id, jlong batch_size, jobject processed)
{
	install(env, processed, PROCESSED_CALLBACK_SIGNATURE, processed_global_ref,
		std::bind(&TRN4CPP::Simulation::Measurement::Readout::MeanSquareError::configure, (unsigned long long)id, (std::size_t)batch_size, std::placeholders::_1),
		make_function(measurement_matrix_callback));
// std::cout << __FUNCTION__ << std::endl;
}
void Java_TRN4JAVA_Simulation_00024Measurement_00024Readout_00024Raw_configure(JNIEnv *env, jclass jclass, jlong id, jlong batch_size, jobject raw)
{
	install(env, raw, RAW_CALLBACK_SIGNATURE, raw_global_ref,
		std::bind(&TRN4CPP::Simulation::Measurement::Readout::Custom::configure, (unsigned long long)id, (std::size_t)batch_size, std::placeholders::_1),
		make_function(measurement_raw_callback));
// std::cout << __FUNCTION__ << std::endl;
}
void Java_TRN4JAVA_Simulation_00024Recording_00024Performances_configure(JNIEnv *env, jclass jclass, jlong id, jobject performances, jboolean train, jboolean primed, jboolean generate)
{
	install(env, performances, PERFORMANCES_CALLBACK_SIGNATURE, recording_global_ref,
		std::bind(&TRN4CPP::Simulation::Recording::Performances::configure, (unsigned long long)id, std::placeholders::_1, (bool)train, (bool)primed, (bool)generate),
		make_function(recording_performances_callback));
// std::cout << __FUNCTION__ << std::endl;
}
void Java_TRN4JAVA_Simulation_00024Recording_00024Scheduling_configure(JNIEnv *env, jclass jclass, jlong id, jobject scheduling)
{
	install(env, scheduling, SCHEDULING_CALLBACK_SIGNATURE, recording_global_ref,
		std::bind(&TRN4CPP::Simulation::Recording::Scheduling::configure, (unsigned long long)id, std::placeholders::_1),
		make_function(recording_scheduling_callback));
// std::cout << __FUNCTION__ << std::endl;
}
void Java_TRN4JAVA_Simulation_00024Recording_00024States_configure(JNIEnv *env, jclass jclass, jlong id, jobject states, jboolean train, jboolean prime, jboolean generate)
{
	install(env, states, STATES_CALLBACK_SIGNATURE, recording_global_ref,
		std::bind(&TRN4CPP::Simulation::Recording::States::configure, (unsigned long long)id, std::placeholders::_1, (bool)train, (bool)prime, (bool)generate),
		make_function(recording_states_callback));
// std::cout << __FUNCTION__ << std::endl;
}
void Java_TRN4JAVA_Simulation_00024Recording_00024Weights_configure(JNIEnv *env, jclass jclass, jlong id, jobject weights, jboolean initialize, jboolean train)
{
	install(env, weights, WEIGHTS_CALLBACK_SIGNATURE, recording_global_ref,
		std::bind(&TRN4CPP::Simulation::Recording::Weights::configure, (unsigned long long)id, std::placeholders::_1, (bool)initialize, (bool)train),
		make_function(recording_weights_callback));
// std::cout << __FUNCTION__ << std::endl;
}
void Java_TRN4JAVA_Simulation_00024Reservoir_00024Weights_00024Feedback_00024Custom_configure(JNIEnv *env, jclass jclass, jlong id, jobject initializer) 
{
	install(env, initializer, WEIGHTS_CALLBACK_SIGNATURE, weights_global_ref,
		std::bind(&TRN4CPP::Simulation::Reservoir::Weights::Feedback::Custom::configure, (unsigned long long)id, std::placeholders::_1, std::placeholders::_2),
		make_function(custom_weights_callback), weights_reply);
}
void Java_TRN4JAVA_Simulation_00024Reservoir_00024Weights_00024Feedback_00024Gaussian_configure(JNIEnv *env, jclass jclass, jlong id, jfloat mu, jfloat sigma) 
{
	try
	{
		TRN4CPP::Simulation::Reservoir::Weights::Feedback::Gaussian::configure((unsigned long long)id, (float)mu, (float)sigma);
	}
	catch (std::exception &e)
	{
		env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
	}
// std::cout << __FUNCTION__ << std::endl;
}
void Java_TRN4JAVA_Simulation_00024Reservoir_00024Weights_00024Feedback_00024Uniform_configure(JNIEnv *env, jclass jclass, jlong id, jfloat a, jfloat b, jfloat sparsity) 
{
	try
	{
		TRN4CPP::Simulation::Reservoir::Weights::Feedback::Uniform::configure((unsigned long long)id, (float)a, (float)b, (float)sparsity);
	}
	catch (std::exception &e)
	{
		env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
	}
// std::cout << __FUNCTION__ << std::endl;
}
void Java_TRN4JAVA_Simulation_00024Reservoir_00024Weights_00024Feedforward_00024Custom_configure(JNIEnv *env, jclass jclass, jlong id, jobject initializer)
{
	install(env, initializer, WEIGHTS_CALLBACK_SIGNATURE, weights_global_ref,
		std::bind(&TRN4CPP::Simulation::Reservoir::Weights::Feedforward::Custom::configure, (unsigned long long)id, std::placeholders::_1, std::placeholders::_2),
		make_function(custom_weights_callback), weights_reply);
// std::cout << __FUNCTION__ << std::endl;
}
void Java_TRN4JAVA_Simulation_00024Reservoir_00024Weights_00024Feedforward_00024Gaussian_configure(JNIEnv *env, jclass jclass, jlong id, jfloat mu, jfloat sigma)
{
	try
	{
		TRN4CPP::Simulation::Reservoir::Weights::Feedforward::Gaussian::configure((unsigned long long)id, (float)mu, (float)sigma);
	}
	catch (std::exception &e)
	{
		env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
	}
// std::cout << __FUNCTION__ << std::endl;
}
void Java_TRN4JAVA_Simulation_00024Reservoir_00024Weights_00024Feedforward_00024Uniform_configure(JNIEnv *env, jclass jclass, jlong id, jfloat a, jfloat b, jfloat sparsity)
{
	try
	{
		TRN4CPP::Simulation::Reservoir::Weights::Feedforward::Uniform::configure((unsigned long long)id, (float)a, (float)b, (float)sparsity);
	}
	catch (std::exception &e)
	{
		env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
	}
// std::cout << __FUNCTION__ << std::endl;
}
void Java_TRN4JAVA_Simulation_00024Reservoir_00024Weights_00024Readout_00024Custom_configure(JNIEnv *env, jclass jclass, jlong id, jobject initializer)
{
	install(env, initializer, WEIGHTS_CALLBACK_SIGNATURE, weights_global_ref,
		std::bind(&TRN4CPP::Simulation::Reservoir::Weights::Readout::Custom::configure, (unsigned long long)id, std::placeholders::_1, std::placeholders::_2),
		make_function(custom_weights_callback), weights_reply);
// std::cout << __FUNCTION__ << std::endl;
}
void Java_TRN4JAVA_Simulation_00024Reservoir_00024Weights_00024Readout_00024Gaussian_configure(JNIEnv *env, jclass jclass, jlong id, jfloat mu, jfloat sigma) 
{
	try
	{
		TRN4CPP::Simulation::Reservoir::Weights::Readout::Gaussian::configure((unsigned long long)id, (float)mu, (float)sigma);
	}
	catch (std::exception &e)
	{
		env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
	}
// std::cout << __FUNCTION__ << std::endl;
}
void Java_TRN4JAVA_Simulation_00024Reservoir_00024Weights_00024Readout_00024Uniform_configure(JNIEnv *env, jclass jclass, jlong id, jfloat a, jfloat b, jfloat sparsity) 
{
	try
	{
		TRN4CPP::Simulation::Reservoir::Weights::Readout::Uniform::configure((unsigned long long)id, (float)a, (float)b, (float)sparsity);
	}
	catch (std::exception &e)
	{
		env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
	}
// std::cout << __FUNCTION__ << std::endl;
}
void Java_TRN4JAVA_Simulation_00024Reservoir_00024Weights_00024Recurrent_00024Custom_configure(JNIEnv *env, jclass jclass, jlong id, jobject initializer)
{
	install(env, initializer, WEIGHTS_CALLBACK_SIGNATURE, weights_global_ref,
		std::bind(&TRN4CPP::Simulation::Reservoir::Weights::Recurrent::Custom::configure, (unsigned long long)id, std::placeholders::_1, std::placeholders::_2),
		make_function(custom_weights_callback), weights_reply);
// std::cout << __FUNCTION__ << std::endl;
}
void Java_TRN4JAVA_Simulation_00024Reservoir_00024Weights_00024Recurrent_00024Gaussian_configure(JNIEnv *env, jclass jclass, jlong id, jfloat mu, jfloat sigma) 
{
	try
	{
		TRN4CPP::Simulation::Reservoir::Weights::Recurrent::Gaussian::configure((unsigned long long)id, (float)mu, (float)sigma);
	}
	catch (std::exception &e)
	{
		env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
	}
// std::cout << __FUNCTION__ << std::endl;
}
void Java_TRN4JAVA_Simulation_00024Reservoir_00024Weights_00024Recurrent_00024Uniform_configure(JNIEnv *env, jclass jclass, jlong id, jfloat a, jfloat b, jfloat sparsity)
{
	try
	{
		TRN4CPP::Simulation::Reservoir::Weights::Recurrent::Uniform::configure((unsigned long long)id, (float)a, (float)b, (float)sparsity);
	}
	catch (std::exception &e)
	{
		env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
	}
// std::cout << __FUNCTION__ << std::endl;
}
void Java_TRN4JAVA_Simulation_00024Reservoir_00024WidrowHoff_configure(JNIEnv *env, jclass jclass, jlong id, jlong stimulus_size, jlong prediction_size, jlong reservoir_size, jfloat leak_rate, jfloat initial_state_scale, jfloat learning_rate, jlong seed, jlong batch_size)
{
	try
	{
		TRN4CPP::Simulation::Reservoir::WidrowHoff::configure((unsigned long long)id, (std::size_t)stimulus_size, (std::size_t)prediction_size, (std::size_t)reservoir_size, (float)leak_rate, (float)initial_state_scale, (float)learning_rate, (unsigned long)seed, (std::size_t)batch_size);
	}
	catch (std::exception &e)
	{
		env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
	}
// std::cout << __FUNCTION__ << std::endl;
}
void Java_TRN4JAVA_Simulation_00024Scheduler_00024Custom_configure(JNIEnv *env, jclass jclass, jlong id, jlong seed, jobject scheduler, jstring tag) 
{
	install(env, scheduler, SCHEDULER_CALLBACK_SIGNATURE, scheduler_global_ref,
		std::bind(&TRN4CPP::Simulation::Scheduler::Custom::configure, (unsigned long long)id, (unsigned long) seed,std::placeholders::_1, std::placeholders::_2, to_string(env, tag)),
		make_function(custom_scheduler_callback), scheduler_reply);
// std::cout << __FUNCTION__ << std::endl;
}
void Java_TRN4JAVA_Simulation_00024Scheduler_00024Mutator_00024Custom_configure(JNIEnv *env, jclass jclass, jlong id, jlong seed, jobject mutator)
{
	install(env, mutator, SCHEDULER_CALLBACK_SIGNATURE, scheduler_global_ref,
		std::bind(&TRN4CPP::Simulation::Scheduler::Mutator::Custom::configure, (unsigned long long)id, (unsigned long)seed, std::placeholders::_1, std::placeholders::_2),
		make_function(custom_mutator_callback), mutator_reply);
// std::cout << __FUNCTION__ << std::endl;
}
void Java_TRN4JAVA_Simulation_00024Scheduler_00024Mutator_00024Reverse_configure(JNIEnv *env, jclass jclass, jlong id, jlong seed, jfloat rate, jlong size) 
{
	try
	{
		TRN4CPP::Simulation::Scheduler::Mutator::Reverse::configure((unsigned long long)id, (unsigned long)seed, (float)rate, (std::size_t)size);
	}
	catch (std::exception &e)
	{
		env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
	}
// std::cout << __FUNCTION__ << std::endl;
}
void Java_TRN4JAVA_Simulation_00024Scheduler_00024Mutator_00024Shuffle_configure(JNIEnv *env, jclass jclass, jlong id, jlong seed)
{
	try
	{
		TRN4CPP::Simulation::Scheduler::Mutator::Shuffle::configure((unsigned long long)id, (unsigned long)seed);
	}
	catch (std::exception &e)
	{
		env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
	}
// std::cout << __FUNCTION__ << std::endl;
}
void Java_TRN4JAVA_Simulation_00024Scheduler_00024Snippets_configure(JNIEnv *env, jclass jclass, jlong id, jlong seed, jint snippets_size, jint time_budget, jstring tag)
{
	try
	{
		TRN4CPP::Simulation::Scheduler::Snippets::configure((unsigned long long)id, (unsigned long)seed, (unsigned int)snippets_size, (unsigned int)time_budget, to_string(env, tag));
	}
	catch (std::exception &e)
	{
		env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
	}
// std::cout << __FUNCTION__ << std::endl;
}
void Java_TRN4JAVA_Simulation_00024Scheduler_00024Tiled_configure(JNIEnv *env, jclass jclass, jlong id, jint epochs)
{
	try
	{
		TRN4CPP::Simulation::Scheduler::Tiled::configure((unsigned long long)id, (unsigned int)epochs);
	}
	catch (std::exception &e)
	{
		env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
	}
// std::cout << __FUNCTION__ << std::endl;
}
void Java_TRN4JAVA_Simulation_00024Loop_notify(JNIEnv *env, jobject loop, jlong id, jlong trial, jlong evaluation, jfloatArray elements, jlong rows, jlong cols)
{
	notify(env, loop, loop_global_ref, loop_reply, (std::size_t)id, (std::size_t)trial, (std::size_t)evaluation, to_float_vector(env, elements), (std::size_t)rows, (std::size_t)cols);
// std::cout << __FUNCTION__ << std::endl;
}
void Java_TRN4JAVA_Simulation_00024Loop_00024Position_install(JNIEnv *env, jclass jclass, jobject loop)
{
	install(env, loop, LOOP_CALLBACK_SIGNATURE, loop_global_ref, TRN4CPP::Simulation::Loop::Position::install, make_function(measurement_matrix_callback), loop_reply);
// std::cout << __FUNCTION__ << std::endl;
}
void Java_TRN4JAVA_Simulation_00024Loop_00024Stimulus_install(JNIEnv *env, jclass jclass, jobject loop)
{
	install(env, loop, LOOP_CALLBACK_SIGNATURE, loop_global_ref, TRN4CPP::Simulation::Loop::Stimulus::install, make_function(measurement_matrix_callback), loop_reply);
// std::cout << __FUNCTION__ << std::endl;
}
void Java_TRN4JAVA_Simulation_00024Measurement_00024Position_00024FrechetDistance_install(JNIEnv *env, jclass jclass, jobject frechet_distance)
{
	install(env, frechet_distance, PROCESSED_CALLBACK_SIGNATURE, processed_global_ref, TRN4CPP::Simulation::Measurement::Position::FrechetDistance::install, make_function(measurement_matrix_callback));
// std::cout << __FUNCTION__ << std::endl;
}
void Java_TRN4JAVA_Simulation_00024Measurement_00024Position_00024MeanSquareError_install(JNIEnv *env, jclass jclass, jobject mean_square_error)
{
	install(env, mean_square_error, PROCESSED_CALLBACK_SIGNATURE, processed_global_ref, TRN4CPP::Simulation::Measurement::Position::MeanSquareError::install, make_function(measurement_matrix_callback));
// std::cout << __FUNCTION__ << std::endl;
}
void Java_TRN4JAVA_Simulation_00024Measurement_00024Position_00024Raw_install(JNIEnv *env, jclass jclass, jobject raw) 
{
	install(env, raw, RAW_CALLBACK_SIGNATURE, raw_global_ref, TRN4CPP::Simulation::Measurement::Position::Raw::install, make_function(measurement_raw_callback));
// std::cout << __FUNCTION__ << std::endl;
}
void Java_TRN4JAVA_Simulation_00024Measurement_00024Readout_00024FrechetDistance_install(JNIEnv *env, jclass jclass, jobject frechet_distance)
{
	install(env, frechet_distance, PROCESSED_CALLBACK_SIGNATURE, processed_global_ref, TRN4CPP::Simulation::Measurement::Readout::FrechetDistance::install, make_function(measurement_matrix_callback));
// std::cout << __FUNCTION__ << std::endl;
}
void Java_TRN4JAVA_Simulation_00024Measurement_00024Readout_00024MeanSquareError_install(JNIEnv *env, jclass jclass, jobject mean_square_error)
{
	install(env, mean_square_error, PROCESSED_CALLBACK_SIGNATURE, processed_global_ref, TRN4CPP::Simulation::Measurement::Readout::MeanSquareError::install, make_function(measurement_matrix_callback));
// std::cout << __FUNCTION__ << std::endl;
}
void Java_TRN4JAVA_Simulation_00024Measurement_00024Readout_00024Raw_install(JNIEnv *env, jclass jclass, jobject raw)
{
	install(env, raw, RAW_CALLBACK_SIGNATURE, raw_global_ref, TRN4CPP::Simulation::Measurement::Readout::Raw::install, make_function(measurement_raw_callback));
// std::cout << __FUNCTION__ << std::endl;
}
void Java_TRN4JAVA_Simulation_00024Recording_00024Performances_install(JNIEnv *env, jclass jclass, jobject performances) 
{
	install(env, performances, PERFORMANCES_CALLBACK_SIGNATURE, recording_global_ref, TRN4CPP::Simulation::Recording::Performances::install, make_function(recording_performances_callback));
// std::cout << __FUNCTION__ << std::endl;
}
void Java_TRN4JAVA_Simulation_00024Recording_00024States_install(JNIEnv *env, jclass jclass, jobject states)
{
	install(env, states, STATES_CALLBACK_SIGNATURE, recording_global_ref, TRN4CPP::Simulation::Recording::States::install, make_function(recording_states_callback));
// std::cout << __FUNCTION__ << std::endl;
}
void Java_TRN4JAVA_Simulation_00024Recording_00024Weights_install(JNIEnv *env, jclass jclass, jobject weights) 
{
	install(env, weights, WEIGHTS_CALLBACK_SIGNATURE, recording_global_ref, TRN4CPP::Simulation::Recording::Weights::install, make_function(recording_weights_callback));
// std::cout << __FUNCTION__ << std::endl;
}
void Java_TRN4JAVA_Simulation_00024Recording_00024Scheduling_install(JNIEnv *env, jclass jclass, jobject scheduling) 
{
	install(env, scheduling, SCHEDULING_CALLBACK_SIGNATURE, recording_global_ref, TRN4CPP::Simulation::Recording::Scheduling::install, make_function(recording_scheduling_callback));
// std::cout << __FUNCTION__ << std::endl;
}
void Java_TRN4JAVA_Simulation_00024Scheduler_install(JNIEnv *env, jclass jclass, jobject scheduler)
{
	install(env, scheduler, SCHEDULER_CALLBACK_SIGNATURE, scheduler_global_ref, TRN4CPP::Simulation::Scheduler::Custom::install, make_function(recording_scheduler_callback), scheduler_reply);
// std::cout << __FUNCTION__ << std::endl;
}
void Java_TRN4JAVA_Simulation_00024Scheduler_notify(JNIEnv *env, jobject scheduler, jlong id, jlong trial, jintArray offsets, jintArray durations)
{
	notify(env, scheduler, scheduler_global_ref, scheduler_reply, (std::size_t)id, (std::size_t)trial, to_int_vector(env, offsets), to_int_vector(env, durations));
// std::cout << __FUNCTION__ << std::endl;
}
void Java_TRN4JAVA_Simulation_00024Scheduler_00024Mutator_install(JNIEnv *env, jclass jclass, jobject mutator)
{
	install(env, mutator, SCHEDULING_CALLBACK_SIGNATURE, mutator_global_ref, TRN4CPP::Simulation::Scheduler::Mutator::Custom::install, make_function(custom_mutator_callback), mutator_reply);
// std::cout << __FUNCTION__ << std::endl;
}
void Java_TRN4JAVA_Simulation_00024Reservoir_00024Weights_notify(JNIEnv *env, jobject initializer, jlong id, jfloatArray weights, jlong batch_size, jlong rows, jlong cols)
{
	notify(env, initializer, weights_global_ref, weights_reply, (std::size_t)id, to_float_vector(env, weights), (std::size_t)batch_size, (std::size_t)rows, (std::size_t)cols);
// std::cout << __FUNCTION__ << std::endl;
}
void Java_TRN4JAVA_Simulation_00024Scheduler_00024Mutator_notify(JNIEnv *env, jobject mutator, jlong id, jlong trial, jintArray offsets, jintArray durations)
{
	notify(env, mutator, mutator_global_ref, mutator_reply, (std::size_t)id, (std::size_t)trial, to_int_vector(env, offsets), to_int_vector(env, durations));
// std::cout << __FUNCTION__ << std::endl;
}
void Java_TRN4JAVA_Simulation_00024Reservoir_00024Weights_00024Feedback_install(JNIEnv *env, jclass jclass, jobject feedback) 
{
	install(env, feedback, WEIGHTS_CALLBACK_SIGNATURE, weights_global_ref, TRN4CPP::Simulation::Reservoir::Weights::Feedback::Custom::install, make_function(custom_weights_callback), weights_reply);
// std::cout << __FUNCTION__ << std::endl;
}
void Java_TRN4JAVA_Simulation_00024Reservoir_00024Weights_00024Feedforward_install(JNIEnv *env, jclass jclass, jobject feedforward) 
{
	install(env, feedforward, WEIGHTS_CALLBACK_SIGNATURE, weights_global_ref, TRN4CPP::Simulation::Reservoir::Weights::Feedforward::Custom::install, make_function(custom_weights_callback), weights_reply);
// std::cout << __FUNCTION__ << std::endl;
}
void Java_TRN4JAVA_Simulation_00024Reservoir_00024Weights_00024Recurrent_install(JNIEnv *env, jclass jclass, jobject recurrent)
{
	install(env, recurrent, WEIGHTS_CALLBACK_SIGNATURE, weights_global_ref, TRN4CPP::Simulation::Reservoir::Weights::Recurrent::Custom::install, make_function(custom_weights_callback), weights_reply);
// std::cout << __FUNCTION__ << std::endl;
}
void Java_TRN4JAVA_Simulation_00024Reservoir_00024Weights_00024Readout_install(JNIEnv *env, jclass jclass, jobject readout)
{
	install(env, readout, WEIGHTS_CALLBACK_SIGNATURE, weights_global_ref, TRN4CPP::Simulation::Reservoir::Weights::Readout::Custom::install, make_function(custom_weights_callback), weights_reply);
// std::cout << __FUNCTION__ << std::endl;
}

