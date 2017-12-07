#pragma once

#include <jni.h>

namespace TRN4JAVA
{
	namespace JNIEnv
	{
		::JNIEnv *get();
		void  set(::JNIEnv *env);
	};
};