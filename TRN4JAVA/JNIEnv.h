#pragma once

#include <jni.h>

namespace TRN4JAVA
{
	namespace JNIEnv
	{
		class Proxy
		{
		private :
			class Handle;
			std::unique_ptr<Handle> handle;

		public :
			Proxy();
			~Proxy();

		public:
			 operator ::JNIEnv * ();
			 ::JNIEnv * operator->();
		};

		void  declare(::JNIEnv *env);
	}


};