#include "stdafx.h"

#include "TRN4CPP/TRN4CPP.h"
#include "TRN4JAVA_Api_Initializer.h"
#include "TLS_JNIEnv.h"
extern std::list<jobject> initializer_global_ref;
extern std::map<jobject, std::function<void(const std::vector<float> &weights, const std::size_t &rows, const std::size_t &cols)>> initializer_notify;
extern std::vector<float> to_float_vector(JNIEnv *env, jfloatArray array);




JNIEXPORT void JNICALL Java_TRN4JAVA_Api_00024Initializer_notify(JNIEnv *env, jobject initializer, jfloatArray weights, jint rows, jint cols)
{
	try
	{
		//TRN4JAVA::getJNIEnv()->MonitorEnter(initializer);
		std::cout << "TRN4JAVA : call to " << __FUNCTION__ << std::endl;
		for (auto ref : initializer_global_ref)
		{
			if (env->IsSameObject(ref, initializer))
			{
				initializer_notify[ref](to_float_vector(env, weights), (std::size_t)rows, (std::size_t)cols);
				std::cout << "TRN4JAVA : initializer notify functor called" << std::endl;
				break;
			}
		}
		//TRN4JAVA::getJNIEnv()->MonitorExit(initializer);
		
		

		std::cout << "TRN4JAVA : sucessful call to " << __FUNCTION__ << std::endl;
	}
	catch (std::exception &e)
	{
		env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
	}
}