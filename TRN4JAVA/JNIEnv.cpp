#include "stdafx.h"
#include "JNIEnv.h"

static JavaVM *vm = NULL;

struct TLS_JNIEnv
{
	bool _detach;
	JNIEnv *env;

	TLS_JNIEnv()
	{
		assert(vm != NULL);
		//	INFORMATION_LOGGER <<   "Attaching " << boost::this_thread::getid() ;
		vm->AttachCurrentThread((void **)&env, NULL);
		assert(env != NULL);
		_detach = true;
	}

	TLS_JNIEnv(JNIEnv *e)
	{
		env = e;
		_detach = false;
	}

	~TLS_JNIEnv()
	{
		if (_detach)
		{
			assert(vm != NULL);
			//	INFORMATION_LOGGER <<   "Detaching " << boost::this_thread::getid() ;
			vm->DetachCurrentThread();
		}
	}
};

static boost::thread_specific_ptr<TLS_JNIEnv> envs;

::JNIEnv * TRN4JAVA::JNIEnv::get()
{
	TLS_JNIEnv *tenv = envs.get();
	if (tenv == NULL) {
		tenv = new TLS_JNIEnv();
		envs.reset(tenv);
	}
	return tenv->env;
}

void TRN4JAVA::JNIEnv::set(::JNIEnv *env)
{
	if (env->GetJavaVM(&vm) < 0)
		throw std::runtime_error("Unable to obtain JVM instance");
	envs.reset(new TLS_JNIEnv(env));
}