#include "stdafx.h"
#include "Functor.h"
#include "Convert.h"
#include "JNIEnv.h"
#include "Helper/Logger.h"



const char *TRN4JAVA::Functor::PROCESSED_CALLBACK_SIGNATURE = "(JJJ[FJJ)V";
const char *TRN4JAVA::Functor::RAW_CALLBACK_SIGNATURE = "(JJ[F[F[FJJJJ)V";
const char *TRN4JAVA::Functor::PERFORMANCES_CALLBACK_SIGNATURE = "(JJJLjava/lang/String;FF)V";
const char *TRN4JAVA::Functor::STATES_CALLBACK_SIGNATURE = "(JJLjava/lang/String;Ljava/lang/String;J[FJJ)V";
const char *TRN4JAVA::Functor::WEIGHTS_CALLBACK_SIGNATURE = "(JJLjava/lang/String;Ljava/lang/String;J[FJJ)V";
const char *TRN4JAVA::Functor::SCHEDULER_CALLBACK_SIGNATURE = "(JJ[FJJ[I[I)V";
const char *TRN4JAVA::Functor::LOOP_CALLBACK_SIGNATURE = "(JJ[FJJ)V";
const char *TRN4JAVA::Functor::ENCODER_CALLBACK_SIGNATURE = TRN4JAVA::Functor::LOOP_CALLBACK_SIGNATURE;
const char *TRN4JAVA::Functor::INITIALIZER_CALLBACK_SIGNATURE = "(J[FJJJ)V";
const char *TRN4JAVA::Functor::SCHEDULING_CALLBACK_SIGNATURE = "(JJ[I[I)V";
const char *TRN4JAVA::Functor::EVENT_CALLBACK_SIGNATURE = "()V";
const char *TRN4JAVA::Functor::EVENT_ACK_CALLBACK_SIGNATURE = "(JJZLjava/lang/String;)V";
const char *TRN4JAVA::Functor::EVENT_PROCESSOR_CALLBACK_SIGNATURE = "(ILjava/lang/String;ILjava/lang/String;)V";
const char *TRN4JAVA::Functor::EVENT_SIMULATION_STATE_CALLBACK_SIGNATURE = "(JJ)V";
const char *TRN4JAVA::Functor::EVENT_SIMULATION_ALLOCATION_CALLBACK_SIGNATURE = "(JI)V";

std::mutex TRN4JAVA::Functor::functor_mutex;

std::vector<jobject>  TRN4JAVA::Functor::processed_global_ref;
std::vector<jobject>  TRN4JAVA::Functor::raw_global_ref;
std::vector<jobject>  TRN4JAVA::Functor::recording_global_ref;
std::vector<jobject>  TRN4JAVA::Functor::loop_global_ref;
std::vector<jobject>  TRN4JAVA::Functor::encoder_global_ref;
std::vector<jobject>  TRN4JAVA::Functor::scheduler_global_ref;
std::vector<jobject>  TRN4JAVA::Functor::mutator_global_ref;
std::vector<jobject>  TRN4JAVA::Functor::weights_global_ref;
std::vector<jobject>  TRN4JAVA::Functor::events_global_ref;

std::map<unsigned long long, std::vector<jobject>> TRN4JAVA::Functor::lookup_ref;

std::map<jobject, std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &stimulus, const std::size_t &rows, const std::size_t &cols)>> TRN4JAVA::Functor::loop_reply;
std::map<jobject, std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols)>> TRN4JAVA::Functor::encoder_reply;
std::map<jobject, std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<int> &offsets, const std::vector<int> &durations)>> TRN4JAVA::Functor::scheduler_reply;
std::map<jobject, std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<int> &offsets, const std::vector<int> &durations)>> TRN4JAVA::Functor::mutator_reply;
std::map<jobject, std::function<void(const unsigned long long &simulation_id, const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)>> TRN4JAVA::Functor::weights_reply;

void TRN4JAVA::Functor::recording_scheduler_callback(jobject object, jmethodID method, const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const unsigned long &seed, const std::vector<float> &elements, const std::size_t &rows, const std::size_t &cols, const std::vector<int> &offsets, const std::vector<int> &durations)
{
	TRACE_LOGGER;
	auto env = TRN4JAVA::JNIEnv::get();
	env->CallVoidMethod(object, method, (jlong)simulation_id, (jlong)evaluation_id, (jlong)seed, TRN4JAVA::Convert::to_jfloat_array(env, elements), (jlong)rows, (jlong)cols, TRN4JAVA::Convert::to_jint_array(env, offsets), TRN4JAVA::Convert::to_jint_array(env, durations));
}
void TRN4JAVA::Functor::custom_mutator_callback(jobject object, jmethodID method, const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const unsigned long &seed, const std::vector<int> &offsets, const std::vector<int> &durations)
{
	TRACE_LOGGER;
	auto env = TRN4JAVA::JNIEnv::get();
	env->CallVoidMethod(object, method, (jlong)simulation_id, (jlong)evaluation_id, (jlong)seed, TRN4JAVA::Convert::to_jint_array(env, offsets), TRN4JAVA::Convert::to_jint_array(env, durations));
}
void TRN4JAVA::Functor::custom_weights_callback(jobject object, jmethodID method, const unsigned long long &simulation_id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)
{
	TRACE_LOGGER;
	auto env = TRN4JAVA::JNIEnv::get();
	env->CallVoidMethod(object, method, (jlong)simulation_id, (jlong)seed, (jlong)matrices, (jlong)rows, (jlong)cols);
}
void TRN4JAVA::Functor::custom_scheduler_callback(jobject object, jmethodID method, const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const unsigned long &seed, const std::vector<float> &elements, const std::size_t &rows, const std::size_t &cols, const std::vector<int> &offsets, const std::vector<int> &durations)
{
	TRACE_LOGGER;
	auto env = TRN4JAVA::JNIEnv::get();
	env->CallVoidMethod(object, method, (jlong)simulation_id, (jlong)evaluation_id, (jlong)seed, TRN4JAVA::Convert::to_jfloat_array(env, elements), (jlong)rows, (jlong)cols, TRN4JAVA::Convert::to_jint_array(env, offsets), TRN4JAVA::Convert::to_jint_array(env, durations));
}
void TRN4JAVA::Functor::measurement_matrix_callback(jobject object, jmethodID method, const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &prediction, const std::size_t &rows, const std::size_t &cols)
{
	TRACE_LOGGER;
	auto env = TRN4JAVA::JNIEnv::get();
	env->CallVoidMethod(object, method, (jlong)simulation_id, (jlong)evaluation_id, TRN4JAVA::Convert::to_jfloat_array(env, prediction), (jlong)rows, (jlong)cols);
}
void TRN4JAVA::Functor::measurement_raw_callback(jobject object, jmethodID method, const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &primed, const std::vector<float> &predicted, const std::vector<float> &expected, const std::size_t &preamble, const std::size_t &pages, const std::size_t &rows, const std::size_t &cols)
{
	TRACE_LOGGER;
	auto env = TRN4JAVA::JNIEnv::get();
	env->CallVoidMethod(object, method, (jlong)simulation_id, (jlong)evaluation_id, TRN4JAVA::Convert::to_jfloat_array(env, primed), TRN4JAVA::Convert::to_jfloat_array(env, predicted), TRN4JAVA::Convert::to_jfloat_array(env, expected), (jlong)preamble, (jlong)pages, (jlong)rows, (jlong)cols);
}
void TRN4JAVA::Functor::recording_performances_callback(jobject object, jmethodID method, const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::string &phase, const float &cycles_per_second, const float &gflops_per_second)
{
	TRACE_LOGGER;
	auto env = TRN4JAVA::JNIEnv::get();
	env->CallVoidMethod(object, method, (jlong)simulation_id, (jlong)evaluation_id, TRN4JAVA::Convert::to_jstring(env, phase), (jfloat)cycles_per_second, (jfloat)gflops_per_second);
}
void TRN4JAVA::Functor::recording_states_callback(jobject object, jmethodID method, const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::string &phase, const std::string &label, const std::size_t &batch, const std::vector<float> &samples, const std::size_t &rows, const std::size_t &cols)
{
	TRACE_LOGGER;
	auto env = TRN4JAVA::JNIEnv::get();
	env->CallVoidMethod(object, method, (jlong)simulation_id, (jlong)evaluation_id, TRN4JAVA::Convert::to_jstring(env, phase), TRN4JAVA::Convert::to_jstring(env, label), (jlong)(batch), TRN4JAVA::Convert::to_jfloat_array(env, samples), (jlong)(rows), (jlong)(cols));
}
void TRN4JAVA::Functor::recording_weights_callback(jobject object, jmethodID method, const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::string &phase, const std::string &label, const std::size_t &batch, const std::vector<float> &weights, const std::size_t &rows, const std::size_t &cols)
{
	TRACE_LOGGER;
	auto env = TRN4JAVA::JNIEnv::get();
	env->CallVoidMethod(object, method, (jlong)simulation_id, TRN4JAVA::Convert::to_jstring(env, phase), TRN4JAVA::Convert::to_jstring(env, label), (jlong)(batch),  TRN4JAVA::Convert::to_jfloat_array(env, weights), (jlong)(rows), (jlong)(cols));
}
void TRN4JAVA::Functor::recording_scheduling_callback(jobject object, jmethodID method, const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<int> &offsets, const std::vector<int> &durations)
{
	TRACE_LOGGER;
	auto env = TRN4JAVA::JNIEnv::get();
	env->CallVoidMethod(object, method, (jlong)simulation_id, (jlong)evaluation_id, TRN4JAVA::Convert::to_jint_array(env, offsets), TRN4JAVA::Convert::to_jint_array(env, durations));
}
void TRN4JAVA::Functor::event_ack_callback(jobject object, jmethodID method, const unsigned long long &simulation_id, const std::size_t &counter, const bool &success, const std::string &cause)
{
	TRACE_LOGGER;
	auto env = TRN4JAVA::JNIEnv::get();
	env->CallVoidMethod(object, method, (jlong)simulation_id, (jlong)counter, (jboolean)success, TRN4JAVA::Convert::to_jstring(env, cause));
}
void TRN4JAVA::Functor::event_simulation_allocation_callback(jobject object, jmethodID method, const unsigned long long &simulation_id, const int &rank)
{
	TRACE_LOGGER;
	auto env = TRN4JAVA::JNIEnv::get();
	env->CallVoidMethod(object, method, (jlong)simulation_id, (jint)rank);
}
void TRN4JAVA::Functor::event_simulation_state_callback(jobject object, jmethodID method, const unsigned long long &simulation_id)
{
	TRACE_LOGGER;
	auto env = TRN4JAVA::JNIEnv::get();
	env->CallVoidMethod(object, method, (jlong)simulation_id);
}
void TRN4JAVA::Functor::event_simulation_state_evaluation_callback(jobject object, jmethodID method, const unsigned long long &simulation_id, const unsigned long long &evaluation_id)
{
	TRACE_LOGGER;
	auto env = TRN4JAVA::JNIEnv::get();
	env->CallVoidMethod(object, method, (jlong)simulation_id, (jlong)evaluation_id);
}
void TRN4JAVA::Functor::event_processor_callback(jobject object, jmethodID method, const int &rank, const std::string &host, const unsigned int &index, const std::string &name)
{
	TRACE_LOGGER;
	auto env = TRN4JAVA::JNIEnv::get();
	env->CallVoidMethod(object, method, (jint)rank, TRN4JAVA::Convert::to_jstring(env, host), (jint)index, TRN4JAVA::Convert::to_jstring(env, name));
}
void TRN4JAVA::Functor::event_callback(jobject object, jmethodID method)
{
	TRACE_LOGGER;
	auto env = TRN4JAVA::JNIEnv::get();
	env->CallVoidMethod(object, method);
}