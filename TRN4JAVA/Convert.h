#pragma once

namespace TRN4JAVA
{
	namespace Convert
	{
		jstring to_jstring(::JNIEnv *env, const std::string &string);
		std::string to_string(::JNIEnv *env, jstring string);
		std::map<std::string, std::string> to_map(::JNIEnv *env, jobject object);
		std::vector<std::string> to_string_vector(::JNIEnv *env, jobjectArray array);
		jfloatArray to_jfloat_array(::JNIEnv *env, const std::vector<float> &vector);
		std::vector<float> to_float_vector(::JNIEnv *env, jfloatArray array);
		std::vector<int> to_int_vector(::JNIEnv *env, jintArray array);
		std::vector<unsigned int> to_unsigned_int_vector(::JNIEnv *env, jintArray array);
		jintArray to_jint_array(::JNIEnv *env, const std::vector<int> &vector);
	};
};
