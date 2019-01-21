#include "stdafx.h"

#include "TRN4CPP/TRN4CPP.h"
#include "TRN4JAVA_Api.h"
#include "TLS_JNIEnv.h"

std::map<jmethodID, jobject> global_ref;
std::map<jobject, std::function<void(const std::vector<float> &stimulus)>> loop_notify;
std::map<jobject, std::function<void(const std::vector<unsigned int> &offsets, const std::vector<unsigned int> &durations)>> scheduler_notify;
std::map<jobject, std::function<void(const std::vector<float> &weights, const size_t &rows, const size_t &cols)>> initializer_notify;

static inline jfloatArray to_jfloat_array(JNIEnv *env, const std::vector<float> &vector)
{
	jfloatArray result;
	auto size = vector.size();
	result = env->NewFloatArray(size);
	env->SetFloatArrayRegion(result, 0, size, &vector[0]);

	return result;
}
static inline jstring to_jstring(JNIEnv *env, const std::string &string)
{
	return env->NewStringUTF(QString::fromStdString(string).toUtf8().constData());
}


static inline std::string to_string(JNIEnv *env, jstring string)
{
	const char *cstr = env->GetStringUTFChars(string, NULL);
	auto str = QString::fromUtf8(cstr, env->GetStringLength(string)).toStdString();
	env->ReleaseStringUTFChars(string, cstr);

	return str;
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
	std::vector<long> vector(size);
	env->GetIntArrayRegion(array, 0, size, &vector[0]);
	
	return std::vector<unsigned int>(vector.begin(), vector.end());
}


void JNICALL Java_TRN4JAVA_Api_initialize_1local(JNIEnv *env, jclass clazz, jint index, jint seed)
{
	try
	{
		TRN4JAVA::init(env);
		std::cout << "TRN4JAVA : call to " << __FUNCTION__ << std::endl;
		TRN4CPP::initialize_local((unsigned int)index, (unsigned long)seed);
		std::cout << "TRN4JAVA : sucessful call to " << __FUNCTION__ << std::endl;
	}
	catch (std::exception &e)
	{
		env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
	}
}


/*
* Class:     TRN4JAVA
* Method:    allocate
* Signature: (I)V
*/
void JNICALL Java_TRN4JAVA_Api_allocate(JNIEnv *env, jclass clazz, jint id)
{
	try
	{
		std::cout << "TRN4JAVA : call to " << __FUNCTION__ << std::endl;
		TRN4CPP::allocate((unsigned int)id);
		std::cout << "TRN4JAVA : sucessful call to " << __FUNCTION__ << std::endl;
	}
	catch (std::exception &e)
	{
		env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
	}
}

/*
* Class:     TRN4JAVA
* Method:    deallocate
* Signature: (I)V
*/
void JNICALL Java_TRN4JAVA_Api_deallocate(JNIEnv *env, jclass clazz, jint id)
{
	try
	{
		std::cout << "TRN4JAVA : call to " << __FUNCTION__ << std::endl;
		TRN4CPP::deallocate((unsigned int)id);
		std::cout << "TRN4JAVA : sucessful call to " << __FUNCTION__ << std::endl;
	}
	catch (std::exception &e)
	{
		env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
	}
}

/*
* Class:     TRN4JAVA
* Method:    train
* Signature: (ILjava/lang/String;)V
*/
void JNICALL Java_TRN4JAVA_Api_train(JNIEnv *env, jclass clazz, jint id, jstring sequence)
{
	try
	{
		std::cout << "TRN4JAVA : call to " << __FUNCTION__ << std::endl;
		TRN4CPP::train((unsigned int)id, to_string(env, sequence));
		
		std::cout << "TRN4JAVA : sucessful call to " << __FUNCTION__ << std::endl;
	}
	catch (std::exception &e)
	{
		env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
	}
}

/*
* Class:     TRN4JAVA
* Method:    test
* Signature: (ILjava/lang/String;I)V
*/
void JNICALL Java_TRN4JAVA_Api_test(JNIEnv *env, jclass clazz, jint id, jstring sequence, jint preamble)
{
	try
	{
		std::cout << "TRN4JAVA : call to " << __FUNCTION__ << std::endl;
		TRN4CPP::test((unsigned int)id, to_string(env, sequence), (size_t)preamble);

		std::cout << "TRN4JAVA : sucessful call to " << __FUNCTION__ << std::endl;
	}
	catch (std::exception &e)
	{
		env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
	}
}

/*
* Class:     TRN4JAVA
* Method:    declare_sequence
* Signature: (ILjava/lang/String;[F[F[FI)V
*/
void JNICALL Java_TRN4JAVA_Api_declare_1sequence(JNIEnv *env, jclass clazz, jint id, jstring label, jfloatArray incoming, jfloatArray expected, jfloatArray reward, jint observations)
{
	try
	{
		std::cout << "TRN4JAVA : call to " << __FUNCTION__ << std::endl;
		TRN4CPP::declare_sequence((unsigned int)id, to_string(env, label), to_float_vector(env, incoming), to_float_vector(env, expected), to_float_vector(env, reward), (size_t)observations);

		std::cout << "TRN4JAVA : sucessful call to " << __FUNCTION__ << std::endl;
	}
	catch (std::exception &e)
	{
		env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
	}
}

/*
* Class:     TRN4JAVA
* Method:    setup_states
* Signature: (ILTRN4JAVA/Matrix;)V
*/
void JNICALL Java_TRN4JAVA_Api_setup_1states(JNIEnv *env, jclass clazz, jint id, jobject states)
{
	try
	{
		std::cout << "TRN4JAVA : call to " << __FUNCTION__ << std::endl;
		jmethodID callback = env->GetMethodID(env->GetObjectClass(states), "callback", "(Ljava/lang/String;[FII)V");
		if (callback == 0) 
		{
			if (env->ExceptionCheck())
			{
				env->ExceptionDescribe();
				env->ExceptionClear();
			}
			throw std::invalid_argument("Can't find JNI method");
		} 
		global_ref[callback] = env->NewGlobalRef(states);
		TRN4CPP::setup_states(id, [callback](const std::string &label, const std::vector<float> &data, const size_t &rows, const size_t &cols)
		{
			auto env = TRN4JAVA::getJNIEnv();
			env->CallVoidMethod(global_ref[callback], callback, to_jstring(env, label), to_jfloat_array(env, data), (jint)rows, (jint)cols);
		});

		std::cout << "TRN4JAVA : sucessful call to " << __FUNCTION__ << std::endl;
	}
	catch (std::exception &e)
	{
		env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
	}
}

/*
* Class:     TRN4JAVA
* Method:    setup_weights
* Signature: (ILTRN4JAVA/Matrix;)V
*/
void JNICALL Java_TRN4JAVA_Api_setup_1weights(JNIEnv *env, jclass clazz, jint id, jobject weights)
{
	try
	{
		std::cout << "TRN4JAVA : call to " << __FUNCTION__ << std::endl;
		jmethodID callback = env->GetMethodID(env->GetObjectClass(weights), "callback", "(Ljava/lang/String;[FII)V");
		if (callback == 0)
		{
			if (env->ExceptionCheck())
			{
				env->ExceptionDescribe();
				env->ExceptionClear();
			}

			throw std::invalid_argument("Can't find JNI method");
		}
		global_ref[callback] = env->NewGlobalRef(weights);
		TRN4CPP::setup_weights(id, [callback](const std::string &label, const std::vector<float> &data, const size_t &rows, const size_t &cols)
		{
			auto env = TRN4JAVA::getJNIEnv();
			env->CallVoidMethod(global_ref[callback], callback, to_jstring(env, label), to_jfloat_array(env, data), (jint)rows, (jint)cols);
		});

		std::cout << "TRN4JAVA : sucessful call to " << __FUNCTION__ << std::endl;
	}
	catch (std::exception &e)
	{
		env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
	}
}

/*
* Class:     TRN4JAVA
* Method:    setup_performances
* Signature: (ILTRN4JAVA/Performances;)V
*/
void JNICALL Java_TRN4JAVA_Api_setup_1performances(JNIEnv *env, jclass clazz, jint id, jobject performances)
{
	try
	{
		std::cout << "TRN4JAVA : call to " << __FUNCTION__ << std::endl;
		jmethodID callback = env->GetMethodID(env->GetObjectClass(performances), "callback", "(Ljava/lang/String;F)V");
		if (callback == 0)
		{
			if (env->ExceptionCheck())
			{
				env->ExceptionDescribe();
				env->ExceptionClear();
			}
			throw std::invalid_argument("Can't find JNI method");
		}
		global_ref[callback] = env->NewGlobalRef(performances);
		TRN4CPP::setup_performances(id, [callback](const std::string &phase, const float &cycles_per_second)
		{
			auto env = TRN4JAVA::getJNIEnv();
			env->CallVoidMethod(global_ref[callback], callback, to_jstring(env, phase), (jfloat)cycles_per_second);
		});

		std::cout << "TRN4JAVA : sucessful call to " << __FUNCTION__ << std::endl;
	}
	catch (std::exception &e)
	{
		env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
	}
}


/*
* Class:     TRN4JAVA
* Method:    configure_begin
* Signature: (I)V
*/
void JNICALL Java_TRN4JAVA_Api_configure_1begin(JNIEnv *env, jclass clazz, jint id)
{
	try
	{
		std::cout << "TRN4JAVA : call to " << __FUNCTION__ << std::endl;
		TRN4CPP::configure_begin((unsigned int)id);
		std::cout << "TRN4JAVA : sucessful call to " << __FUNCTION__ << std::endl;
	}
	catch (std::exception &e)
	{
		env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
	}
}

/*
* Class:     TRN4JAVA
* Method:    configure_end
* Signature: (I)V
*/
void JNICALL Java_TRN4JAVA_Api_configure_1end(JNIEnv *env, jclass clazz, jint id)
{
	try
	{
		std::cout << "TRN4JAVA : call to " << __FUNCTION__ << std::endl;
		TRN4CPP::configure_end((unsigned int)id);
		std::cout << "TRN4JAVA : sucessful call to " << __FUNCTION__ << std::endl;
	}
	catch (std::exception &e)
	{
		env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
	}
}

/*
* Class:     TRN4JAVA
* Method:    configure_reservoir_widrow_hoff
* Signature: (IIIIFFF)V
*/
void JNICALL Java_TRN4JAVA_Api_configure_1reservoir_1widrow_1hoff(JNIEnv *env, jclass clazz, jint id, jint stimulus_size, jint prediction_size, jint reservoir_size, jfloat leak_rate, jfloat initial_state_scale, jfloat learning_rate)
{
	try
	{
		std::cout << "TRN4JAVA : call to " << __FUNCTION__ << std::endl;
		TRN4CPP::configure_reservoir_widrow_hoff((unsigned int)id, (size_t)stimulus_size, (size_t)prediction_size, (size_t)reservoir_size, (float)leak_rate, (float)initial_state_scale, (float)learning_rate);
		std::cout << "TRN4JAVA : sucessful call to " << __FUNCTION__ << std::endl;
	}
	catch (std::exception &e)
	{
		env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
	}
}

/*
* Class:     TRN4JAVA
* Method:    configure_loop_copy
* Signature: (II)V
*/
void JNICALL Java_TRN4JAVA_Api_configure_1loop_1copy(JNIEnv *env, jclass clazz, jint id, jint stimulus_size)
{
	try
	{
		std::cout << "TRN4JAVA : call to " << __FUNCTION__ << std::endl;
		TRN4CPP::configure_loop_copy((unsigned int)id, (size_t)stimulus_size);
		std::cout << "TRN4JAVA : sucessful call to " << __FUNCTION__ << std::endl;
	}
	catch (std::exception &e)
	{
		env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
	}
}

/*
* Class:     TRN4JAVA
* Method:    configure_loop_spatial_filter
* Signature: (IIIIF)V
*/
void JNICALL Java_TRN4JAVA_Api_configure_1loop_1spatial_1filter(JNIEnv *env, jclass clazz, jint id, jint stimulus_size, jint rows, jint cols, jfloat sigma)
{
	try
	{
		std::cout << "TRN4JAVA : call to " << __FUNCTION__ << std::endl;
		TRN4CPP::configure_loop_spatial_filter((unsigned int)id, (size_t)stimulus_size, (size_t)rows, (size_t)cols, (float)sigma);
		std::cout << "TRN4JAVA : sucessful call to " << __FUNCTION__ << std::endl;
	}
	catch (std::exception &e)
	{
		env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
	}
}

/*
* Class:     TRN4JAVA
* Method:    configure_loop_custom
* Signature: (IILTRN4JAVA/Loop;)V
*/


void JNICALL Java_TRN4JAVA_Api_configure_1loop_1custom(JNIEnv *env, jclass clazz, jint id, jint stimulus_size, jobject loop)
{
	try
	{
		std::cout << "TRN4JAVA : call to " << __FUNCTION__ << std::endl;
		jmethodID callback = env->GetMethodID(env->GetObjectClass(loop), "callback", "([F)V");
		if (callback == 0)
		{
			if (env->ExceptionCheck())
			{
				env->ExceptionDescribe();
				env->ExceptionClear();
			}
			throw std::invalid_argument("Can't find JNI method");
		}
		global_ref[callback] = env->NewGlobalRef(loop);
		std::cout << (TRN4JAVA::init(env)) << std::endl;
	//	jfloatArray action = to_jfloat_array(env, prediction);
		/*env->CallVoidMethod(loop, callback, action);
		{
			if (env->ExceptionCheck())
			{
				env->ExceptionDescribe();
				env->ExceptionClear();
			}
		}*/
		std::cout << "LOOP in loop_custom" << callback << std::endl;

		TRN4CPP::configure_loop_custom((unsigned int)id, (size_t)stimulus_size,
			[callback, stimulus_size](const std::vector<float> &prediction)
			{
				std::cout << "getJNIenv" << std::endl;
				auto env = TRN4JAVA::getJNIEnv();
				std::cout << "action" << std::endl;
				jfloatArray action = to_jfloat_array(env, prediction);
				std::cout << "invoking Java callback" << std::endl;
				env->CallVoidMethod(global_ref[callback], callback, action);
			},
			loop_notify[global_ref[callback]]);
		

		std::cout << "TRN4JAVA : sucessful call to " << __FUNCTION__ << std::endl;
	}
	catch (std::exception &e)
	{
		env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
	}
}

/*
* Class:     TRN4JAVA
* Method:    configure_scheduler_tiled
* Signature: (II)V
*/
void JNICALL Java_TRN4JAVA_Api_configure_1scheduler_1tiled(JNIEnv *env, jclass clazz, jint id, jint epochs)
{
	try
	{
		std::cout << "TRN4JAVA : call to " << __FUNCTION__ << std::endl;
		TRN4CPP::configure_scheduler_tiled((unsigned int)id, (unsigned int)epochs);
		std::cout << "TRN4JAVA : sucessful call to " << __FUNCTION__ << std::endl;
	}
	catch (std::exception &e)
	{
		env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
	}
}

/*
* Class:     TRN4JAVA
* Method:    configure_scheduler_snippets
* Signature: (IIIZ)V
*/
void JNICALL Java_TRN4JAVA_Api_configure_1scheduler_1snippets(JNIEnv *env, jclass clazz, jint id, jint snippets_size, jint time_budget, jboolean reward_driven)
{
	try
	{
		std::cout << "TRN4JAVA : call to " << __FUNCTION__ << std::endl;
		TRN4CPP::configure_scheduler_snippets((unsigned int)id, (size_t)snippets_size, (size_t)(time_budget), (bool)reward_driven);
		std::cout << "TRN4JAVA : sucessful call to " << __FUNCTION__ << std::endl;
	}
	catch (std::exception &e)
	{
		env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
	}
}

/*
* Class:     TRN4JAVA
* Method:    configure_scheduler_custom
* Signature: (IILTRN4JAVA/Scheduler;)V
*/
void JNICALL Java_TRN4JAVA_Api_configure_1scheduler_1custom(JNIEnv *env, jclass clazz, jint id, jobject scheduler)
{
	try
	{
		std::cout << "TRN4JAVA : call to " << __FUNCTION__ << std::endl;
		jmethodID callback = env->GetMethodID(env->GetObjectClass(scheduler), "callback", "([F[F[FI)V");
		if (callback == 0)
		{
			if (env->ExceptionCheck())
			{
				env->ExceptionDescribe();
				env->ExceptionClear();
			}
			throw std::invalid_argument("Can't find JNI method");
		}
		global_ref[callback] = env->NewGlobalRef(scheduler);
		TRN4CPP::configure_scheduler_custom((unsigned int)id,
			[scheduler, callback](const std::vector<float> &incoming, const std::vector<float> &expected, const std::vector<float> &reward, const size_t &observations)
			{
				auto env = TRN4JAVA::getJNIEnv();
				env->CallVoidMethod(global_ref[callback], callback, to_jfloat_array(env, incoming), to_jfloat_array(env, expected), to_jfloat_array(env, reward), (jint)observations);
			},
			scheduler_notify[global_ref[callback]]
		);

		std::cout << "TRN4JAVA : sucessful call to " << __FUNCTION__ << std::endl;
	}
	catch (std::exception &e)
	{
		env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
	}
}

/*
* Class:     TRN4JAVA
* Method:    configure_readout_uniform
* Signature: (IFFF)V
*/
void JNICALL Java_TRN4JAVA_Api_configure_1readout_1uniform(JNIEnv *env, jclass clazz, jint id, jfloat a, jfloat b, jfloat sparsity)
{
	try
	{
		std::cout << "TRN4JAVA : call to " << __FUNCTION__ << std::endl;
		TRN4CPP::configure_readout_uniform((unsigned int)id, (float)a, (float)b, (float)sparsity);
		std::cout << "TRN4JAVA : sucessful call to " << __FUNCTION__ << std::endl;
	}
	catch (std::exception &e)
	{
		env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
	}
}

/*
* Class:     TRN4JAVA
* Method:    configure_readout_gaussian
* Signature: (IFF)V
*/
void JNICALL Java_TRN4JAVA_Api_configure_1readout_1gaussian(JNIEnv *env, jclass clazz, jint id, jfloat mu, jfloat sigma)
{
	try
	{
		std::cout << "TRN4JAVA : call to " << __FUNCTION__ << std::endl;
		TRN4CPP::configure_readout_gaussian((unsigned int)id, (float)mu, (float)sigma);
		std::cout << "TRN4JAVA : sucessful call to " << __FUNCTION__ << std::endl;
	}
	catch (std::exception &e)
	{
		env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
	}
}

/*
* Class:     TRN4JAVA
* Method:    configure_readout_custom
* Signature: (ILTRN4JAVA/Initializer;)V
*/
void JNICALL Java_TRN4JAVA_Api_configure_1readout_1custom(JNIEnv *env, jclass clazz, jint id, jobject initializer)
{
	try
	{
		std::cout << "TRN4JAVA : call to " << __FUNCTION__ << std::endl;
		jmethodID callback = env->GetMethodID(env->GetObjectClass(initializer), "callback", "(II)V");
		if (callback == 0)
		{
			if (env->ExceptionCheck())
			{
				env->ExceptionDescribe();
				env->ExceptionClear();
			}
			throw std::invalid_argument("Can't find JNI method");
		}
		global_ref[callback] = env->NewGlobalRef(initializer);
		TRN4CPP::configure_readout_custom((unsigned int)id, [initializer, callback](const size_t &rows, const size_t &cols)
		{
			auto env = TRN4JAVA::getJNIEnv();
			env->CallVoidMethod(global_ref[callback], callback, (jint)rows, (jint)cols);
		},
		initializer_notify[global_ref[callback]]
		);

		std::cout << "TRN4JAVA : sucessful call to " << __FUNCTION__ << std::endl;
	}
	catch (std::exception &e)
	{
		env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
	}
}

/*
* Class:     TRN4JAVA
* Method:    configure_feedback_uniform
* Signature: (IFFF)V
*/
void JNICALL Java_TRN4JAVA_Api_configure_1feedback_1uniform(JNIEnv *env, jclass clazz, jint id, jfloat a, jfloat b, jfloat sparsity)
{
	try
	{
		std::cout << "TRN4JAVA : call to " << __FUNCTION__ << std::endl;
		TRN4CPP::configure_feedback_uniform((unsigned int)id, (float)a, (float)b, (float)sparsity);
		std::cout << "TRN4JAVA : sucessful call to " << __FUNCTION__ << std::endl;
	}
	catch (std::exception &e)
	{
		env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
	}
}

/*
* Class:     TRN4JAVA
* Method:    configure_feedback_gaussian
* Signature: (IFF)V
*/
void JNICALL Java_TRN4JAVA_Api_configure_1feedback_1gaussian(JNIEnv *env, jclass clazz, jint id, jfloat mu, jfloat sigma)
{
	try
	{
		std::cout << "TRN4JAVA : call to " << __FUNCTION__ << std::endl;
		TRN4CPP::configure_feedback_gaussian((unsigned int)id, (float)mu, (float)sigma);
		std::cout << "TRN4JAVA : sucessful call to " << __FUNCTION__ << std::endl;
	}
	catch (std::exception &e)
	{
		env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
	}
}

/*
* Class:     TRN4JAVA
* Method:    configure_feedback_custom
* Signature: (ILTRN4JAVA/Initializer;)V
*/
void JNICALL Java_TRN4JAVA_Api_configure_1feedback_1custom(JNIEnv *env, jclass clazz, jint id, jobject initializer)
{
	try
	{
		std::cout << "TRN4JAVA : call to " << __FUNCTION__ << std::endl;
		jmethodID callback = env->GetMethodID(env->GetObjectClass(initializer), "callback", "(II)V");
		if (callback == 0)
		{
			if (env->ExceptionCheck())
			{
				env->ExceptionDescribe();
				env->ExceptionClear();
			}
			throw std::invalid_argument("Can't find JNI method");
		}
		global_ref[callback] = env->NewGlobalRef(initializer);
		TRN4CPP::configure_feedback_custom((unsigned int)id, [initializer, callback](const size_t &rows, const size_t &cols)
		{
			auto env = TRN4JAVA::getJNIEnv();
			env->CallVoidMethod(global_ref[callback], callback, (jint)rows, (jint)cols);
		},
			initializer_notify[global_ref[callback]]
			);

		std::cout << "TRN4JAVA : sucessful call to " << __FUNCTION__ << std::endl;
	}
	catch (std::exception &e)
	{
		env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
	}
}

/*
* Class:     TRN4JAVA
* Method:    configure_recurrent_uniform
* Signature: (IFFF)V
*/
void JNICALL Java_TRN4JAVA_Api_configure_1recurrent_1uniform(JNIEnv *env, jclass clazz, jint id, jfloat a, jfloat b, jfloat sparsity)
{
	try
	{
		std::cout << "TRN4JAVA : call to " << __FUNCTION__ << std::endl;
		TRN4CPP::configure_recurrent_uniform((unsigned int)id, (float)a, (float)b, (float)sparsity);
		std::cout << "TRN4JAVA : sucessful call to " << __FUNCTION__ << std::endl;
	}
	catch (std::exception &e)
	{
		env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
	}
}

/*
* Class:     TRN4JAVA
* Method:    configure_recurrent_gaussian
* Signature: (IFF)V
*/
void JNICALL Java_TRN4JAVA_Api_configure_1recurrent_1gaussian(JNIEnv *env, jclass clazz, jint id, jfloat mu, jfloat sigma)
{
	try
	{
		std::cout << "TRN4JAVA : call to " << __FUNCTION__ << std::endl;
		TRN4CPP::configure_recurrent_gaussian((unsigned int)id, (float)mu, (float)sigma);
		std::cout << "TRN4JAVA : sucessful call to " << __FUNCTION__ << std::endl;
	}
	catch (std::exception &e)
	{
		env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
	}
}

/*
* Class:     TRN4JAVA
* Method:    configure_recurrent_custom
* Signature: (ILTRN4JAVA/Initializer;)V
*/
void JNICALL Java_TRN4JAVA_Api_configure_1recurrent_1custom(JNIEnv *env, jclass clazz, jint id, jobject initializer)
{
	try
	{
		std::cout << "TRN4JAVA : call to " << __FUNCTION__ << std::endl;
		jmethodID callback = env->GetMethodID(env->GetObjectClass(initializer), "callback", "(II)V");
		if (callback == 0)
		{
			if (env->ExceptionCheck())
			{
				env->ExceptionDescribe();
				env->ExceptionClear();
			}
			throw std::invalid_argument("Can't find JNI method");
		}
		global_ref[callback] = env->NewGlobalRef(initializer);
		TRN4CPP::configure_recurrent_custom((unsigned int)id, [initializer, callback](const size_t &rows, const size_t &cols)
			{
				auto env = TRN4JAVA::getJNIEnv();
				env->CallVoidMethod(global_ref[callback], callback, (jint)rows, (jint)cols);
			},
			initializer_notify[global_ref[callback]]
			);

		std::cout << "TRN4JAVA : sucessful call to " << __FUNCTION__ << std::endl;
	}
	catch (std::exception &e)
	{
		env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
	}
}

/*
* Class:     TRN4JAVA
* Method:    configure_feedforward_uniform
* Signature: (IFFF)V
*/
void JNICALL Java_TRN4JAVA_Api_configure_1feedforward_1uniform(JNIEnv *env, jclass clazz, jint id, jfloat a, jfloat b, jfloat sparsity)
{
	try
	{
		std::cout << "TRN4JAVA : call to " << __FUNCTION__ << std::endl;
		TRN4CPP::configure_feedforward_uniform((unsigned int)id, (float)a, (float)b, (float)sparsity);
		std::cout << "TRN4JAVA : sucessful call to " << __FUNCTION__ << std::endl;
	}
	catch (std::exception &e)
	{
		env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
	}
}

/*
* Class:     TRN4JAVA
* Method:    configure_feedforward_gaussian
* Signature: (IFF)V
*/
void JNICALL Java_TRN4JAVA_Api_configure_1feedforward_1gaussian(JNIEnv *env, jclass clazz, jint id, jfloat mu, jfloat sigma)
{
	try
	{
		std::cout << "TRN4JAVA : call to " << __FUNCTION__ << std::endl;
		TRN4CPP::configure_feedforward_gaussian((unsigned int)id, (float)mu, (float)sigma);
		std::cout << "TRN4JAVA : sucessful call to " << __FUNCTION__ << std::endl;
	}
	catch (std::exception &e)
	{
		env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
	}
}


/*
* Class:     TRN4JAVA
* Method:    configure_feedforward_custom
* Signature: (ILTRN4JAVA/Initializer;)V
*/
void JNICALL Java_TRN4JAVA_Api_configure_1feedforward_1custom(JNIEnv *env, jclass clazz, jint id, jobject initializer)
{

	try
	{
		std::cout << "TRN4JAVA : call to " << __FUNCTION__ << std::endl;
		jmethodID callback = env->GetMethodID(env->GetObjectClass(initializer), "callback", "(II)V");
		if (callback == 0)
		{
			if (env->ExceptionCheck())
			{
				env->ExceptionDescribe();
				env->ExceptionClear();
			}
			throw std::invalid_argument("Can't find JNI method");
		}
		global_ref[callback] = env->NewGlobalRef(initializer);
		TRN4CPP::configure_feedforward_custom((unsigned int)id, [initializer, callback](const size_t &rows, const size_t &cols)
		{
			auto env = TRN4JAVA::getJNIEnv();
			env->CallVoidMethod(global_ref[callback], callback, (jint)rows, (jint)cols);
		},
			initializer_notify[global_ref[callback]]
			);

		std::cout << "TRN4JAVA : sucessful call to " << __FUNCTION__ << std::endl;
	}
	catch (std::exception &e)
	{
		env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
	}
}