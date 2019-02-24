#include "stdafx.h"
#include "JNIEnv.h"
#include "Helper/Logger.h"

static JavaVM *vm = nullptr;
static std::mutex mutex;

class TRN4JAVA::JNIEnv::Proxy::Handle
{
public:
	::JNIEnv *env;
	std::unique_lock<std::mutex> lock;

	Handle() : lock(mutex)
	{
		vm->AttachCurrentThread((void **)&env, NULL);
	}

	~Handle()
	{
		vm->DetachCurrentThread();
	}
};



TRN4JAVA::JNIEnv::Proxy::Proxy() : 
	handle(std::make_unique<Handle>())
{
}

TRN4JAVA::JNIEnv::Proxy::~Proxy()
{
	handle.reset();
}

TRN4JAVA::JNIEnv::Proxy::operator ::JNIEnv *()
{
	return handle->env;
}

::JNIEnv * TRN4JAVA::JNIEnv::Proxy:: operator->()
{
	return handle->env;
}


void TRN4JAVA::JNIEnv::declare(::JNIEnv *env)
{
	std::unique_lock<std::mutex> lock(mutex);
	if (vm == nullptr)
	{
		if (env->GetJavaVM(&vm) < 0)
			throw std::runtime_error("Unable to obtain JVM instance");
		INFORMATION_LOGGER << "JVM instance obtained";
	}
	/*else
	{
		WARNING_LOGGER << "JVM instance already obtained";
	}*/
}