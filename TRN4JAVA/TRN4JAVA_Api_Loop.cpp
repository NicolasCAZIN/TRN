#include "stdafx.h"

#include "TRN4CPP/TRN4CPP.h"
#include "TRN4JAVA_Api_Loop.h"
#include "TLS_JNIEnv.h"
extern std::list<jobject>  loop_global_ref;


extern std::map<jobject, std::function<void(const std::vector<float> &stimulus)>> loop_notify;
extern std::vector<float> to_float_vector(JNIEnv *env, jfloatArray array);

JNIEXPORT void JNICALL Java_TRN4JAVA_Api_00024Loop_notify(JNIEnv *env, jobject loop, jfloatArray stimulus)
{
	try
	{
		//TRN4JAVA::getJNIEnv()->MonitorEnter(loop);
		std::cout << "TRN4JAVA : call to " << __FUNCTION__ << std::endl;
		for (auto ref : loop_global_ref)
		{
			if (env->IsSameObject(ref, loop))
			{
				loop_notify[ref](to_float_vector(env, stimulus));
				std::cout << "TRN4JAVA : loop notify functor called" << std::endl;
				break;
			}
		}
		//TRN4JAVA::getJNIEnv()->MonitorExit(loop);
		std::cout << "TRN4JAVA : sucessful call to " << __FUNCTION__ << std::endl;
	}
	catch (std::exception &e)
	{
		env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
	}
}