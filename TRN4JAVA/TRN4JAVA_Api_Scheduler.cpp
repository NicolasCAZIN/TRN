
#include "stdafx.h"

#include "TRN4CPP/TRN4CPP.h"
#include "TRN4JAVA_Api_Scheduler.h"
#include "TLS_JNIEnv.h"

extern std::list<jobject>  scheduler_global_ref;
extern std::map<jobject, std::function<void(const std::vector<int> &offsets, const std::vector<int> &durations)>> scheduler_notify;
extern std::vector<int> to_unsigned_int_vector(JNIEnv *env, jintArray array);

JNIEXPORT void JNICALL Java_TRN4JAVA_Api_00024Scheduler_notify(JNIEnv *env, jobject scheduler, jintArray offsets, jintArray durations)
{
	try
	{
		//TRN4JAVA::getJNIEnv()->MonitorEnter(scheduler);
		std::cout << "TRN4JAVA : call to " << __FUNCTION__ << std::endl;
		for (auto ref : scheduler_global_ref)
		{
			if (env->IsSameObject(ref, scheduler))
			{
				scheduler_notify[ref](to_unsigned_int_vector(env, offsets), to_unsigned_int_vector(env, durations));
				std::cout << "TRN4JAVA : scheduler notify functor called" << std::endl;
				break;
			}
		}

		std::cout << "TRN4JAVA : sucessful call to " << __FUNCTION__ << std::endl;
		//TRN4JAVA::getJNIEnv()->MonitorExit(scheduler);
	}
	catch (std::exception &e)
	{
		env->ThrowNew(env->FindClass("java/lang/RuntimeException"), e.what());
	}
}

