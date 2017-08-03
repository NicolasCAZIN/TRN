#include "stdafx.h"
#include "TLS_JNIEnv.h"

static JavaVM *vm = NULL;

struct TLS_JNIEnv
{
	bool _detach;
	JNIEnv *env;

	TLS_JNIEnv()
	{
		assert(vm != NULL);
		std::cout << "Attaching " << boost::this_thread::get_id() << std::endl;
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
			std::cout << "Detaching " << boost::this_thread::get_id() << std::endl;
			vm->DetachCurrentThread();
		}
	}
};

static boost::thread_specific_ptr<TLS_JNIEnv> envs;


JNIEnv *TRN4JAVA::getJNIEnv() 
{
	TLS_JNIEnv *tenv = envs.get();
	if (tenv == NULL) {
		tenv = new TLS_JNIEnv();
		envs.reset(tenv);
	}
	return tenv->env;
}

bool TRN4JAVA::init(JNIEnv *env)
{
	if (env->GetJavaVM(&vm) < 0)
		return false;
	envs.reset(new TLS_JNIEnv(env));
	return true;
}
