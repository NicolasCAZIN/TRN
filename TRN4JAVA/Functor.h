#pragma once

#include "Helper/Logger.h"

namespace TRN4JAVA
{
	namespace Functor
	{
		extern const char *PROCESSED_CALLBACK_SIGNATURE;
		extern const char *RAW_CALLBACK_SIGNATURE;
		extern const char *PERFORMANCES_CALLBACK_SIGNATURE;
		extern const char *STATES_CALLBACK_SIGNATURE;
		extern const char *WEIGHTS_CALLBACK_SIGNATURE;
		extern const char *SCHEDULER_CALLBACK_SIGNATURE;
		extern const char *LOOP_CALLBACK_SIGNATURE;
		extern const char *ENCODER_CALLBACK_SIGNATURE;
		extern const char *INITIALIZER_CALLBACK_SIGNATURE;
		extern const char *SCHEDULING_CALLBACK_SIGNATURE;
		extern const char *EVENT_CALLBACK_SIGNATURE;
		extern const char *EVENT_ACK_CALLBACK_SIGNATURE;
		extern const char *EVENT_PROCESSOR_CALLBACK_SIGNATURE;
		extern const char *EVENT_SIMULATION_STATE_CALLBACK_SIGNATURE;
		extern const char *EVENT_SIMULATION_ALLOCATION_CALLBACK_SIGNATURE;

		extern std::vector<jobject>  events_global_ref;
		extern std::vector<jobject>  processed_global_ref;
		extern std::vector<jobject>  raw_global_ref;
		extern std::vector<jobject>  recording_global_ref;
		extern std::vector<jobject>  loop_global_ref;
		extern std::vector<jobject>  encoder_global_ref;
		extern std::vector<jobject>  scheduler_global_ref;
		extern std::vector<jobject>  mutator_global_ref;
		extern std::vector<jobject>  weights_global_ref;

		extern std::map<unsigned long long, std::vector<jobject>> lookup_ref;

		extern std::map<jobject, std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &stimulus, const std::size_t &rows, const std::size_t &cols)>> loop_reply;
		extern std::map<jobject, std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols)>> encoder_reply;
		extern std::map<jobject, std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<int> &offsets, const std::vector<int> &durations)>> scheduler_reply;
		extern std::map<jobject, std::function<void(const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<int> &offsets, const std::vector<int> &durations)>> mutator_reply;
		extern std::map<jobject, std::function<void(const unsigned long long &simulation_id, const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)>> weights_reply;

		extern std::mutex functor_mutex;
		template<typename Installer, typename ... CallbackArgs>
		static void install(JNIEnv *env, const jobject object, const char *signature, std::vector<jobject> &global_ref,  Installer &installer, const std::function<void(jobject, jmethodID, CallbackArgs ...)> &callback)
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
				{
					//std::unique_lock<std::mutex> guard(functor_mutex);
					global_ref.push_back(ref);
				}
				
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
				ERROR_LOGGER << e.what(); env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
			}
		}
		template<typename Notify, typename Installer, typename ... CallbackArgs>
		static void install(JNIEnv *env, const jobject object, const char *signature, std::vector<jobject> &global_ref,  Installer &installer, const std::function<void(jobject, jmethodID, CallbackArgs ...)> &callback, Notify &notify)
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
				{
					//std::unique_lock<std::mutex> guard(functor_mutex);
					global_ref.push_back(ref);
				}
				
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
				ERROR_LOGGER << e.what(); env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
			}
		}
		template<typename Notify1, typename Notify2, typename Installer, typename ... CallbackArgs>
		static void install(JNIEnv *env, const jobject object,const char *signature, std::vector<jobject> &global_ref, Installer &installer,
			const std::function<void(jobject, jmethodID, CallbackArgs ...)> &callback, Notify1 &notify1, Notify2 &notify2)
		{
			TRACE_LOGGER;
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
			
				{
					//std::unique_lock<std::mutex> guard(functor_mutex);

					global_ref.push_back(ref);
					
				}
				
				installer
				(
					[=](CallbackArgs ... args)
				{
					callback(ref, callback_id, args...);
				}
					,
					notify1[ref],
					notify2[ref]
					);
			}
			catch (std::exception &e)
			{
				ERROR_LOGGER << e.what(); env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
			}
		}
		template<typename Notify, typename ... NotifyArgs>
		static void notify(JNIEnv *env, jobject object, const unsigned long long &simulation_id, const std::vector<jobject> &global_ref, Notify &notify, NotifyArgs ... args)
		{
			//env->MonitorEnter(object);
			TRACE_LOGGER;
			try
			{
				jobject target = NULL;
				{
				
					/*std::unique_lock<std::mutex> guard(functor_mutex);
					auto lookup_it = std::find_if(std::begin(lookup_ref[id]), std::end(lookup_ref[id]), [=](const jobject ref)
					{
						return env->IsSameObject(object, ref);
					});
					if (lookup_it == lookup_ref[id].end())
					{*/
						auto global_it = std::find_if(std::begin(global_ref), std::end(global_ref), [=](const jobject ref)
						{
							return env->IsSameObject(object, ref);
						});
						if (global_it == global_ref.end())
							throw std::runtime_error("Reply object not found");
						target = *global_it;
			
					/*	lookup_ref[id].push_back(target);
					}
					else
					{
						TRACE_LOGGER << "Found in lookup (" << lookup_ref[id].size() << ")";
						target = *lookup_it;
					}*/
				}

				notify[target](args...);
			}
			catch (std::exception &e)
			{
				ERROR_LOGGER << e.what(); env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
			}
			//env->MonitorExit(object);
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

		void recording_scheduler_callback(jobject object, jmethodID method, const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const unsigned long &seed, const std::vector<float> &elements, const std::size_t &rows, const std::size_t &cols, const std::vector<int> &offsets, const std::vector<int> &durations);
		void custom_mutator_callback(jobject object, jmethodID method, const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const unsigned long &seed, const std::vector<int> &offsets, const std::vector<int> &durations);
		void custom_weights_callback(jobject object, jmethodID method, const unsigned long long &simulation_id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols);
		void custom_scheduler_callback(jobject object, jmethodID method, const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const unsigned long &seed,const std::vector<float> &elements, const std::size_t &rows, const std::size_t &cols, const std::vector<int> &offsets, const std::vector<int> &durations);
		void measurement_matrix_callback(jobject object, jmethodID method, const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &prediction, const std::size_t &rows, const std::size_t &cols);
		void measurement_raw_callback(jobject object, jmethodID method, const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<float> &primed, const std::vector<float> &predicted, const std::vector<float> &expected, const std::size_t &preamble, const std::size_t &pages, const std::size_t &rows, const std::size_t &cols);
		void recording_performances_callback(jobject object, jmethodID method, const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::string &phase, const float &cycles_per_second, const float &gflops_per_second);
		void recording_states_callback(jobject object, jmethodID method, const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::string &phase, const std::string &label, const std::size_t &batch, const std::vector<float> &samples, const std::size_t &rows, const std::size_t &cols);
		void recording_weights_callback(jobject object, jmethodID method, const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::string &phase, const std::string &label, const std::size_t &batch, const std::vector<float> &weights, const std::size_t &rows, const std::size_t &cols);
		void recording_scheduling_callback(jobject object, jmethodID method, const unsigned long long &simulation_id, const unsigned long long &evaluation_id, const std::vector<int> &offsets, const std::vector<int> &durations);
		void event_ack_callback(jobject object, jmethodID method, const unsigned long long &simulation_id, const std::size_t &counter, const bool &success, const std::string &cause);
		void event_simulation_allocation_callback(jobject object, jmethodID method, const unsigned long long &simulation_id, const int &rank);
		void event_simulation_state_callback(jobject object, jmethodID method, const unsigned long long &simulation_id);
		void event_simulation_state_evaluation_callback(jobject object, jmethodID method, const unsigned long long &simulation_id, const unsigned long long &evaluation_id);
		void event_processor_callback(jobject object, jmethodID method, const int &rank, const std::string &host, const unsigned int &index, const std::string &name);
		void event_callback(jobject object, jmethodID method);
	}
};
