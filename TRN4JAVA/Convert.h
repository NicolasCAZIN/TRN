#pragma once

#include "trn4java_global.h"

namespace TRN4JAVA
{
	namespace Convert
	{
		jstring TRN4JAVA_EXPORT to_jstring(::JNIEnv *env, const std::string &string);
		std::string TRN4JAVA_EXPORT to_string(::JNIEnv *env, jstring string);
		std::map<std::string, std::string> TRN4JAVA_EXPORT to_map(::JNIEnv *env, jobject object);
		std::vector<std::string> TRN4JAVA_EXPORT to_string_vector(::JNIEnv *env, jobjectArray array);
		jfloatArray TRN4JAVA_EXPORT to_jfloat_array(::JNIEnv *env, const std::vector<float> &vector);
		std::vector<float> TRN4JAVA_EXPORT to_float_vector(::JNIEnv *env, jfloatArray array);
		std::vector<int> TRN4JAVA_EXPORT to_int_vector(::JNIEnv *env, jintArray array);
		std::vector<unsigned int> TRN4JAVA_EXPORT to_unsigned_int_vector(::JNIEnv *env, jintArray array);
		jintArray TRN4JAVA_EXPORT to_jint_array(::JNIEnv *env, const std::vector<int> &vector);
	};
};
