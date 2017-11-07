#include "stdafx.h"
#include "TRN4JAVA_Engine.h"
#include "TRN4JAVA_Engine_Backend.h"
#include "TRN4JAVA_Engine_Backend_Local.h"
#include "TRN4JAVA_Engine_Backend_Remote.h"

#include "TRN4JAVA_Simulation.h"
#include "TRN4JAVA_Simulation_Scheduler.h"
#include "TRN4JAVA_Simulation_Scheduler_Mutator.h"
#include "TRN4JAVA_Simulation_Loop.h"
#include "TRN4JAVA_Simulation_Loop_Position.h"
#include "TRN4JAVA_Simulation_Loop_Stimulus.h"

#include "TRN4JAVA_Simulation_Measurement.h"
#include "TRN4JAVA_Simulation_Measurement_Position.h"
#include "TRN4JAVA_Simulation_Measurement_Position_FrechetDistance.h"
#include "TRN4JAVA_Simulation_Measurement_Position_MeanSquareError.h"
#include "TRN4JAVA_Simulation_Measurement_Position_Raw.h"
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
#include "TRN4JAVA_Simulation_Reservoir_Weights_Feedforward.h"
#include "TRN4JAVA_Simulation_Reservoir_Weights_Recurrent.h"
#include "TRN4JAVA_Simulation_Reservoir_Weights_Feedback.h"

#include "TLS_JNIEnv.h"
#include "TRN4CPP/Simplified.h"
#include "TRN4CPP/Custom.h"
#include "TRN4CPP/Callbacks.h"

static const char *LOOP_CALLBACK_SIGNATURE = "(JJJ[FJJ)V";
static const char *PROCESSED_CALLBACK_SIGNATURE = "(JJJ[FJJ)V";
static const char *RAW_CALLBACK_SIGNATURE = "(JJJ[F[F[FJJJJ)V";
static const char *PERFORMANCES_CALLBACK_SIGNATURE = "(JJJLjava/lang/String;FF)V";
static const char *STATES_CALLBACK_SIGNATURE = "(JLjava/lang/String;Ljava/lang/String;JJJ[FJJ)V";
static const char *WEIGHTS_CALLBACK_SIGNATURE = "(JLjava/lang/String;Ljava/lang/String;JJ[FJJ)V";
static const char *SCHEDULING_CALLBACK_SIGNATURE = "(JJ[I[I)V";
static const char *SCHEDULER_CALLBACK_SIGNATURE = "(JJJ[FJJ[I[I)V";


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
}

template<typename Installer, typename ... CallbackArgs>
void install(JNIEnv *env, const jobject object, const char *signature, std::list<jobject> &global_ref, Installer &installer, const std::function<void(jobject, jmethodID, CallbackArgs ...)> &callback)
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
void install(JNIEnv *env, const jobject object, const char *signature, std::list<jobject> &global_ref, Installer &installer, const std::function<void(jobject, jmethodID, CallbackArgs ...)> &callback,Notify &notify) 
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
template<typename Notify, typename ... NotifyArgs>
void notify(JNIEnv *env, jobject object, const std::list<jobject> &global_ref, Notify &notify, NotifyArgs ... args)
{
	env->MonitorEnter(object);
	try
	{
		auto it = std::find_if(std::begin(global_ref), std::end(global_ref), [object, env](const jobject ref)
		{
			return env->IsSameObject(ref, object);
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
static void recordings_performances_callback(jobject object, jmethodID method, const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::string &phase, const float &cycles_per_second, const float &gflops_per_second)
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
	env->CallVoidMethod(object, method, (jlong)id, (jlong)seed, (jlong)trial, to_jfloat_array(env, elements), to_jint_array(env, durations), (jlong)rows, (jlong)cols, to_jint_array(env, offsets), to_jint_array(env, durations));
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
 void Java_TRN4JAVA_Simulation_00024Loop_notify(JNIEnv *env, jobject loop, jlong id, jlong trial, jlong evaluation, jfloatArray elements, jlong rows, jlong cols)
{
	notify(env, loop, loop_global_ref, loop_reply, (std::size_t)id, (std::size_t)trial, (std::size_t)evaluation, to_float_vector(env, elements), (std::size_t)rows, (std::size_t)cols);
}
 void Java_TRN4JAVA_Simulation_00024Loop_00024Position_install(JNIEnv *env, jclass jclass, jobject loop)
{
	install(env, loop, LOOP_CALLBACK_SIGNATURE, loop_global_ref, TRN4CPP::Simulation::Loop::Position::install, make_function(measurement_matrix_callback), loop_reply);
}
 void Java_TRN4JAVA_Simulation_00024Loop_00024Stimulus_install(JNIEnv *env, jclass jclass, jobject loop)
{
	install(env, loop, LOOP_CALLBACK_SIGNATURE, loop_global_ref, TRN4CPP::Simulation::Loop::Stimulus::install, make_function(measurement_matrix_callback), loop_reply);
}
 void Java_TRN4JAVA_Simulation_00024Measurement_00024Position_00024FrechetDistance_install(JNIEnv *env, jclass jclass, jobject frechet_distance)
{
	install(env, frechet_distance, PROCESSED_CALLBACK_SIGNATURE, processed_global_ref, TRN4CPP::Simulation::Measurement::Position::FrechetDistance::install, make_function(measurement_matrix_callback));
}
 void Java_TRN4JAVA_Simulation_00024Measurement_00024Position_00024MeanSquareError_install(JNIEnv *env, jclass jclass, jobject mean_square_error)
{
	install(env, mean_square_error, PROCESSED_CALLBACK_SIGNATURE, processed_global_ref, TRN4CPP::Simulation::Measurement::Position::MeanSquareError::install, make_function(measurement_matrix_callback));
}
 void Java_TRN4JAVA_Simulation_00024Measurement_00024Position_00024Raw_install(JNIEnv *env, jclass jclass, jobject raw) 
{
	install(env, raw, RAW_CALLBACK_SIGNATURE, raw_global_ref, TRN4CPP::Simulation::Measurement::Position::Raw::install, make_function(measurement_raw_callback));
}
 void Java_TRN4JAVA_Simulation_00024Measurement_00024Readout_00024FrechetDistance_install(JNIEnv *env, jclass jclass, jobject frechet_distance)
{
	install(env, frechet_distance, PROCESSED_CALLBACK_SIGNATURE, processed_global_ref, TRN4CPP::Simulation::Measurement::Readout::FrechetDistance::install, make_function(measurement_matrix_callback));
}
 void Java_TRN4JAVA_Simulation_00024Measurement_00024Readout_00024MeanSquareError_install(JNIEnv *env, jclass jclass, jobject mean_square_error)
{
	install(env, mean_square_error, PROCESSED_CALLBACK_SIGNATURE, processed_global_ref, TRN4CPP::Simulation::Measurement::Readout::MeanSquareError::install, make_function(measurement_matrix_callback));
}
 void Java_TRN4JAVA_Simulation_00024Measurement_00024Readout_00024Raw_install(JNIEnv *env, jclass jclass, jobject raw)
{
	install(env, raw, RAW_CALLBACK_SIGNATURE, raw_global_ref, TRN4CPP::Simulation::Measurement::Readout::Raw::install, make_function(measurement_raw_callback));
}
 void Java_TRN4JAVA_Simulation_00024Recording_00024Performances_install(JNIEnv *env, jclass jclass, jobject performances) 
{
	install(env, performances, PERFORMANCES_CALLBACK_SIGNATURE, recording_global_ref, TRN4CPP::Simulation::Recording::Performances::install, make_function(recordings_performances_callback));
}
 void Java_TRN4JAVA_Simulation_00024Recording_00024States_install(JNIEnv *env, jclass jclass, jobject states)
{
	install(env, states, STATES_CALLBACK_SIGNATURE, recording_global_ref, TRN4CPP::Simulation::Recording::States::install, make_function(recording_states_callback));
}
 void Java_TRN4JAVA_Simulation_00024Recording_00024Weights_install(JNIEnv *env, jclass jclass, jobject weights) 
{
	install(env, weights, WEIGHTS_CALLBACK_SIGNATURE, recording_global_ref, TRN4CPP::Simulation::Recording::Weights::install, make_function(recording_weights_callback));
}
 void Java_TRN4JAVA_Simulation_00024Recording_00024Scheduling_install(JNIEnv *env, jclass jclass, jobject scheduling) 
{
	install(env, scheduling, SCHEDULING_CALLBACK_SIGNATURE, recording_global_ref, TRN4CPP::Simulation::Recording::Scheduling::install, make_function(recording_scheduling_callback));
}
 void Java_TRN4JAVA_Simulation_00024Scheduler_install(JNIEnv *env, jclass jclass, jobject scheduler)
{
	install(env, scheduler, SCHEDULER_CALLBACK_SIGNATURE, scheduler_global_ref, TRN4CPP::Simulation::Scheduler::Custom::install, make_function(recording_scheduler_callback), scheduler_reply);
}
 void Java_TRN4JAVA_Simulation_00024Scheduler_notify(JNIEnv *env, jobject scheduler, jlong id, jlong trial, jintArray offsets, jintArray durations)
{
	notify(env, scheduler, scheduler_global_ref, scheduler_reply, (std::size_t)id, (std::size_t)trial, to_int_vector(env, offsets), to_int_vector(env, durations));
}
 void Java_TRN4JAVA_Simulation_00024Scheduler_00024Mutator_install(JNIEnv *env, jclass jclass, jobject mutator)
{
	install(env, mutator, SCHEDULING_CALLBACK_SIGNATURE, mutator_global_ref, TRN4CPP::Simulation::Scheduler::Mutator::Custom::install, make_function(custom_mutator_callback), mutator_reply);
}
 void Java_TRN4JAVA_Simulation_00024Reservoir_00024Weights_notify(JNIEnv *env, jobject initializer, jlong id, jfloatArray weights, jlong batch_size, jlong rows, jlong cols)
{
	notify(env, initializer, weights_global_ref, weights_reply, (std::size_t)id, to_float_vector(env, weights), (std::size_t)batch_size, (std::size_t)rows, (std::size_t)cols);
}
 void Java_TRN4JAVA_Simulation_00024Scheduler_00024Mutator_notify(JNIEnv *env, jobject mutator, jlong id, jlong trial, jintArray offsets, jintArray durations)
{
	notify(env, mutator, mutator_global_ref, mutator_reply, (std::size_t)id, (std::size_t)trial, to_int_vector(env, offsets), to_int_vector(env, durations));
}
 void Java_TRN4JAVA_Simulation_00024Reservoir_00024Weights_00024Feedback_install(JNIEnv *env, jclass jclass, jobject feedback) 
{
	install(env, feedback, WEIGHTS_CALLBACK_SIGNATURE, weights_global_ref, TRN4CPP::Simulation::Reservoir::Weights::Feedback::Custom::install, make_function(custom_weights_callback), weights_reply);
}
 void Java_TRN4JAVA_Simulation_00024Reservoir_00024Weights_00024Feedforward_install(JNIEnv *env, jclass jclass, jobject feedforward) 
{
	install(env, feedforward, WEIGHTS_CALLBACK_SIGNATURE, weights_global_ref, TRN4CPP::Simulation::Reservoir::Weights::Feedforward::Custom::install, make_function(custom_weights_callback), weights_reply);
}
 void Java_TRN4JAVA_Simulation_00024Reservoir_00024Weights_00024Recurrent_install(JNIEnv *env, jclass jclass, jobject recurrent)
{
	install(env, recurrent, WEIGHTS_CALLBACK_SIGNATURE, weights_global_ref, TRN4CPP::Simulation::Reservoir::Weights::Recurrent::Custom::install, make_function(custom_weights_callback), weights_reply);
}
 void Java_TRN4JAVA_Simulation_00024Reservoir_00024Weights_00024Readout_install(JNIEnv *env, jclass jclass, jobject readout)
{
	install(env, readout, WEIGHTS_CALLBACK_SIGNATURE, weights_global_ref, TRN4CPP::Simulation::Reservoir::Weights::Readout::Custom::install, make_function(custom_weights_callback), weights_reply);
}

