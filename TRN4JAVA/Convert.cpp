#include "stdafx.h"
#include "Convert.h"

jstring TRN4JAVA::Convert::to_jstring(::JNIEnv *env, const std::string &string)
{
	return env->NewStringUTF(string.c_str());
}
std::string TRN4JAVA::Convert::to_string(::JNIEnv *env, jstring string)
{
	const char *cstr = env->GetStringUTFChars(string, NULL);
	std::string str(cstr);
	env->ReleaseStringUTFChars(string, cstr);

	return str;
}
std::map<std::string, std::string> TRN4JAVA::Convert::to_map(::JNIEnv *env, jobject object)
{
	std::map<std::string, std::string> map;

	if (!env->IsInstanceOf(object, env->FindClass("java/util/Map")))
		throw std::invalid_argument("Object is not a Map");
	jmethodID keySet = env->GetMethodID(
		env->GetObjectClass(object),
		"keySet",
		"()Ljava/util/Set;"
	);
	if (keySet == 0)
		throw std::invalid_argument("object does not implement a keySet() method");
	jmethodID get = env->GetMethodID(
		env->GetObjectClass(object),
		"get",
		"(Ljava/lang/Object;)Ljava/lang/Object;"
	);
	if (keySet == 0)
		throw std::invalid_argument("object does not implement a get() method");
	jobject set = env->CallObjectMethod(object, keySet);
	if (set == NULL)
		throw std::invalid_argument("Can't get map keys");
	jmethodID toArray = env->GetMethodID(
		env->GetObjectClass(set),
		"toArray",
		"()[Ljava/lang/Object;"
	);
	if (toArray == 0)
		throw std::invalid_argument("object does not implement a toArray() method");
	auto keys = (jobjectArray)env->CallObjectMethod(set, toArray);
	if (keys == NULL)
		throw std::invalid_argument("Can't keys array");
	for (std::size_t k = 0; k < env->GetArrayLength(keys); k++)
	{
		auto key = (jstring)env->GetObjectArrayElement(keys, k);
		if (key == NULL)
			throw std::invalid_argument("key is null");
		auto value = (jstring)env->CallObjectMethod(object, get, key);
		if (value == NULL)
			throw std::invalid_argument("value is null");
		map[boost::to_upper_copy(to_string(env, key))] = to_string(env, value);
	}

	return map;
}
std::vector<std::string> TRN4JAVA::Convert::to_string_vector(::JNIEnv *env, jobjectArray array)
{
	jsize size = env->GetArrayLength(array);
	std::vector<std::string> vector(size);
	for (jsize k = 0; k < size; k++)
	{
		vector[k] = to_string(env, (jstring)env->GetObjectArrayElement(array, k));
	}
	return vector;
}
jfloatArray TRN4JAVA::Convert::to_jfloat_array(::JNIEnv *env, const std::vector<float> &vector)
{
	jfloatArray result;
	auto size = vector.size();
	result = env->NewFloatArray(size);
	env->SetFloatArrayRegion(result, 0, size, &vector[0]);

	return result;
}
std::vector<float> TRN4JAVA::Convert::to_float_vector(::JNIEnv *env, jfloatArray array)
{
	jsize size = env->GetArrayLength(array);
	std::vector<float> vector(size);
	env->GetFloatArrayRegion(array, 0, size, &vector[0]);

	return vector;
}
std::vector<int> TRN4JAVA::Convert::to_int_vector(::JNIEnv *env, jintArray array)
{
	jsize size = env->GetArrayLength(array);
	std::vector<jint> vector(size);
	env->GetIntArrayRegion(array, 0, size, &vector[0]);

	return std::vector<int>(vector.begin(), vector.end());
}
std::vector<unsigned int> TRN4JAVA::Convert::to_unsigned_int_vector(::JNIEnv *env, jintArray array)
{
	jsize size = env->GetArrayLength(array);
	std::vector<jint> vector(size);
	env->GetIntArrayRegion(array, 0, size, &vector[0]);

	return std::vector<unsigned int>(vector.begin(), vector.end());
}
jintArray TRN4JAVA::Convert::to_jint_array(::JNIEnv *env, const std::vector<int> &vector)
{
	jintArray result;
	auto size = vector.size();
	result = env->NewIntArray(size);
	std::vector<long> ivector(size);
	ivector.assign(vector.begin(), vector.end());
	env->SetIntArrayRegion(result, 0, size, &ivector[0]);

	return result;
}