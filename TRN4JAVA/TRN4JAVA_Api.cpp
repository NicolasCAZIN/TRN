#include "stdafx.h"

#include "TRN4CPP/TRN4CPP.h"
#include "TRN4JAVA_Api.h"
#include "TLS_JNIEnv.h"

std::list<jobject> states_global_ref;
std::list<jobject>  weights_global_ref;
std::list<jobject>  performances_global_ref;
std::list<jobject>  scheduler_global_ref;
std::list<jobject>  loop_global_ref;
std::list<jobject>  initializer_global_ref;
std::map<jobject, std::function<void(const std::vector<float> &stimulus)>> loop_notify;
std::map<jobject, std::function<void(const std::vector<int> &offsets, const std::vector<int> &durations)>> scheduler_notify;
std::map<jobject, std::function<void(const std::vector<float> &weights, const std::size_t &rows, const std::size_t &cols)>> initializer_notify;

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
std::vector<int> to_unsigned_int_vector(JNIEnv *env, jintArray array)
{
	jsize size = env->GetArrayLength(array);
	std::vector<long> vector(size);
	env->GetIntArrayRegion(array, 0, size, &vector[0]);
	
	return std::vector<int>(vector.begin(), vector.end());
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

void JNICALL Java_TRN4JAVA_Api_initialize_1local(JNIEnv *env, jclass clazz, jint index, jint seed)
{
	try
	{
		TRN4JAVA::init(env);
		std::cout << "TRN4JAVA : call to " << __FUNCTION__ << std::endl;
		TRN4CPP::initialize_local((int)index, (unsigned long)seed);
		std::cout << "TRN4JAVA : sucessful call to " << __FUNCTION__ << std::endl;
	}
	catch (std::exception &e)
	{
		env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
	}
}
void JNICALL Java_TRN4JAVA_Api_initialize_1remote(JNIEnv *env, jclass clazz, jstring host, jint port)
{
	try
	{
		TRN4JAVA::init(env);
		std::cout << "TRN4JAVA : call to " << __FUNCTION__ << std::endl;

		TRN4CPP::initialize_remote(to_string(env, host), (unsigned short)port);
		std::cout << "TRN4JAVA : sucessful call to " << __FUNCTION__ << std::endl;
	}
	catch (std::exception &e)
	{
		env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
	}
}
void JNICALL Java_TRN4JAVA_Api_initialize_1distributed(JNIEnv *env, jclass clazz, jobjectArray args)
{
	try
	{
		TRN4JAVA::init(env);
		std::cout << "TRN4JAVA : call to " << __FUNCTION__ << std::endl;
		auto args_v = to_string_vector(env, args);
		char **argv = new char*[args_v.size() + 2];
		argv[0] = "TRN4JAVA";
		auto argc = args_v.size() + 1;
		for (std::size_t k = 0; k < args_v.size(); k++)
			argv[k + 1] = (char *)args_v[k].c_str();
		argv[argc] = NULL;
	
		TRN4CPP::initialize_distributed(argc, argv);
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
		TRN4CPP::allocate((int)id);
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
		TRN4CPP::deallocate((int)id);
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
void JNICALL Java_TRN4JAVA_Api_train(JNIEnv *env, jclass clazz, jint id, jstring label, jstring incoming, jstring expected)
{
	try
	{
		std::cout << "TRN4JAVA : call to " << __FUNCTION__ << std::endl;
		TRN4CPP::train((int)id, to_string(env, label), to_string(env, incoming), to_string(env, expected));
		
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
void JNICALL Java_TRN4JAVA_Api_test(JNIEnv *env, jclass clazz, jint id, jstring label, jstring incoming, jstring expected, jint preamble)
{
	try
	{
		std::cout << "TRN4JAVA : call to " << __FUNCTION__ << std::endl;
		TRN4CPP::test((int)id, to_string(env, label), to_string(env, incoming), to_string(env, expected), (std::size_t)preamble);

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
void JNICALL Java_TRN4JAVA_Api_declare_1sequence(JNIEnv *env, jclass clazz, jint id, jstring label, jstring tag, jfloatArray elements, jint observations)
{
	try
	{
		std::cout << "TRN4JAVA : call to " << __FUNCTION__ << std::endl;
		TRN4CPP::declare_sequence((int)id, to_string(env, label), to_string(env, tag), to_float_vector(env, elements), (std::size_t)observations);

		std::cout << "TRN4JAVA : sucessful call to " << __FUNCTION__ << std::endl;
	}
	catch (std::exception &e)
	{
		env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
	}
}
/*
* Class:     TRN4JAVA_Api
* Method:    declare_batch
* Signature: (ILjava/lang/String;Ljava/lang/String;[Ljava/lang/String;)V
*/
JNIEXPORT void JNICALL Java_TRN4JAVA_Api_declare_1batch(JNIEnv *env, jclass clazz, jint id, jstring label, jstring tag, jobjectArray labels)
{
	try
	{
		std::cout << "TRN4JAVA : call to " << __FUNCTION__ << std::endl;
		TRN4CPP::declare_batch((int)id, to_string(env, label), to_string(env, tag), to_string_vector(env, labels));

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
		auto ref = env->NewGlobalRef(states);
		states_global_ref.push_back(ref);
		TRN4CPP::setup_states(id, [ref, callback](const std::string &label, const std::vector<float> &data, const std::size_t &rows, const std::size_t &cols)
		{
			auto env = TRN4JAVA::getJNIEnv();
			env->CallVoidMethod(ref, callback, to_jstring(env, label), to_jfloat_array(env, data), (jint)rows, (jint)cols);
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
		auto ref = env->NewGlobalRef(weights);
		weights_global_ref.push_back(ref);
		TRN4CPP::setup_weights(id, [ref, callback](const std::string &label, const std::vector<float> &data, const std::size_t &rows, const std::size_t &cols)
		{
			auto env = TRN4JAVA::getJNIEnv();
			env->CallVoidMethod(ref, callback, to_jstring(env, label), to_jfloat_array(env, data), (jint)rows, (jint)cols);
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
		auto ref = env->NewGlobalRef(performances);
		performances_global_ref.push_back(ref);
		TRN4CPP::setup_performances(id, [ref, callback](const std::string &phase, const float &cycles_per_second)
		{
			auto env = TRN4JAVA::getJNIEnv();
			env->CallVoidMethod(ref, callback, to_jstring(env, phase), (jfloat)cycles_per_second);
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
		TRN4CPP::configure_begin((int)id);
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
		TRN4CPP::configure_end((int)id);
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
		TRN4CPP::configure_reservoir_widrow_hoff((int)id, (std::size_t)stimulus_size, (std::size_t)prediction_size, (std::size_t)reservoir_size, (float)leak_rate, (float)initial_state_scale, (float)learning_rate);
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
		TRN4CPP::configure_loop_copy((int)id, (std::size_t)stimulus_size);
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
void JNICALL Java_TRN4JAVA_Api_configure_1loop_1spatial_1filter(JNIEnv *env, jclass clazz, jint id, jint stimulus_size, jobject position, jobject stimulus, jint rows, jint cols, jfloat x_min, jfloat x_max, jfloat y_min, jfloat y_max, jfloatArray response, jfloat sigma, jfloat radius, jstring tag)
{
	try
	{
		std::cout << "TRN4JAVA : call to " << __FUNCTION__ << std::endl;

		jmethodID position_callback = env->GetMethodID(env->GetObjectClass(position), "callback", "([F)V");
		if (position_callback == 0)
		{
			if (env->ExceptionCheck())
			{
				env->ExceptionDescribe();
				env->ExceptionClear();
			}
			throw std::invalid_argument("Can't find JNI method");
		}
		auto position_ref = env->NewGlobalRef(position);
		loop_global_ref.push_back(position_ref);
		jmethodID stimulus_callback = env->GetMethodID(env->GetObjectClass(stimulus), "callback", "([F)V");
		if (stimulus_callback == 0)
		{
			if (env->ExceptionCheck())
			{
				env->ExceptionDescribe();
				env->ExceptionClear();
			}
			throw std::invalid_argument("Can't find JNI method");
		}
		auto stimulus_ref =env->NewGlobalRef(stimulus);
		loop_global_ref.push_back(stimulus_ref);
		TRN4CPP::configure_loop_spatial_filter((int)id, (std::size_t)stimulus_size, 
			[position_ref, position_callback, stimulus_size](const std::vector<float> &prediction)
		{
			auto env = TRN4JAVA::getJNIEnv();
			jfloatArray action = to_jfloat_array(env, prediction);
			env->CallVoidMethod(position_ref, position_callback, action);
		},
			loop_notify[position_ref],
			[stimulus_ref, stimulus_callback, stimulus_size](const std::vector<float> &prediction)
		{
			auto env = TRN4JAVA::getJNIEnv();
			jfloatArray action = to_jfloat_array(env, prediction);
			env->CallVoidMethod(stimulus_ref, stimulus_callback, action);
		},
			loop_notify[stimulus_ref],
			
			(std::size_t)rows, (std::size_t)cols, 
			std::make_pair((float)x_min, (float)x_max), std::make_pair((float)y_min, (float)y_max),
			to_float_vector(TRN4JAVA::getJNIEnv(), response),
			(float)sigma, (float)radius, to_string(env, tag));
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
		auto ref = env->NewGlobalRef(loop);
		loop_global_ref.push_back(ref);
	
	//	jfloatArray action = to_jfloat_array(env, prediction);
		/*env->CallVoidMethod(loop, callback, action);
		{
			if (env->ExceptionCheck())
			{
				env->ExceptionDescribe();
				env->ExceptionClear();
			}
		}*/
		TRN4CPP::configure_loop_custom((int)id, (std::size_t)stimulus_size,
			[ref, callback, stimulus_size](const std::vector<float> &prediction)
			{
				auto env = TRN4JAVA::getJNIEnv();
				jfloatArray action = to_jfloat_array(env, prediction);
				env->CallVoidMethod(ref, callback, action);
			},
			loop_notify[ref]);
		

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
		TRN4CPP::configure_scheduler_tiled((int)id, (int)epochs);
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
void JNICALL Java_TRN4JAVA_Api_configure_1scheduler_1snippets(JNIEnv *env, jclass clazz, jint id, jint snippets_size, jint time_budget, jstring tag)
{
	try
	{
		std::cout << "TRN4JAVA : call to " << __FUNCTION__ << std::endl;
		TRN4CPP::configure_scheduler_snippets((int)id, (std::size_t)snippets_size, (std::size_t)(time_budget), to_string(env,tag));
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
void JNICALL Java_TRN4JAVA_Api_configure_1scheduler_1custom(JNIEnv *env, jclass clazz, jint id, jobject scheduler, jstring tag)
{
	try
	{
		std::cout << "TRN4JAVA : call to " << __FUNCTION__ << std::endl;
		jmethodID callback = env->GetMethodID(env->GetObjectClass(scheduler), "callback", "([FII[I[I)V");
		if (callback == 0)
		{
			if (env->ExceptionCheck())
			{
				env->ExceptionDescribe();
				env->ExceptionClear();
			}
			throw std::invalid_argument("Can't find JNI method");
		}
		auto ref = env->NewGlobalRef(scheduler);
		scheduler_global_ref.push_back(ref);
		TRN4CPP::configure_scheduler_custom((int)id,
			[ref, callback](const std::vector<float> &elements, const std::size_t &rows, const std::size_t &cols, const std::vector<int> &offsets, const std::vector<int> &durations)
			{
				auto env = TRN4JAVA::getJNIEnv();
				env->CallVoidMethod(ref, callback, to_jfloat_array(env, elements), (jint)rows, (jint)cols, to_jint_array(env, offsets), to_jint_array(env, durations) );
			},
			scheduler_notify[ref],
				to_string(TRN4JAVA::getJNIEnv(), tag)
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
		TRN4CPP::configure_readout_uniform((int)id, (float)a, (float)b, (float)sparsity);
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
		TRN4CPP::configure_readout_gaussian((int)id, (float)mu, (float)sigma);
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
		auto ref = env->NewGlobalRef(initializer);
		initializer_global_ref.push_back(ref);
		TRN4CPP::configure_readout_custom((int)id, [initializer, ref, callback](const std::size_t &rows, const std::size_t &cols)
		{
			auto env = TRN4JAVA::getJNIEnv();
			env->CallVoidMethod(ref, callback, (jint)rows, (jint)cols);
		},
		initializer_notify[ref]
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
		TRN4CPP::configure_feedback_uniform((int)id, (float)a, (float)b, (float)sparsity);
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
		TRN4CPP::configure_feedback_gaussian((int)id, (float)mu, (float)sigma);
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
		auto ref = env->NewGlobalRef(initializer);
		initializer_global_ref.push_back(ref);
		TRN4CPP::configure_feedback_custom((int)id, [initializer, ref, callback](const std::size_t &rows, const std::size_t &cols)
		{
			auto env = TRN4JAVA::getJNIEnv();
			env->CallVoidMethod(ref, callback, (jint)rows, (jint)cols);
		},
			initializer_notify[ref]
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
		TRN4CPP::configure_recurrent_uniform((int)id, (float)a, (float)b, (float)sparsity);
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
		TRN4CPP::configure_recurrent_gaussian((int)id, (float)mu, (float)sigma);
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
		auto ref = env->NewGlobalRef(initializer);
		initializer_global_ref.push_back(ref);
		TRN4CPP::configure_recurrent_custom((int)id, [initializer, ref, callback](const std::size_t &rows, const std::size_t &cols)
			{
				auto env = TRN4JAVA::getJNIEnv();
				env->CallVoidMethod(ref, callback, (jint)rows, (jint)cols);
			},
			initializer_notify[ref]
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
		TRN4CPP::configure_feedforward_uniform((int)id, (float)a, (float)b, (float)sparsity);
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
		TRN4CPP::configure_feedforward_gaussian((int)id, (float)mu, (float)sigma);
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
		auto ref = env->NewGlobalRef(initializer);
		initializer_global_ref.push_back(ref);
		TRN4CPP::configure_feedforward_custom((int)id, [initializer, ref, callback](const std::size_t &rows, const std::size_t &cols)
		{
			auto env = TRN4JAVA::getJNIEnv();
			env->CallVoidMethod(ref, callback, (jint)rows, (jint)cols);
		},
			initializer_notify[ref]
			);

		std::cout << "TRN4JAVA : sucessful call to " << __FUNCTION__ << std::endl;
	}
	catch (std::exception &e)
	{
		env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
	}
}