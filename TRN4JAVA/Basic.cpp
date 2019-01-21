#include "stdafx.h"

#include "JNIEnv.h"
#include "Convert.h"
#include "TRN4JAVA_Basic.h"
#include "TRN4JAVA_Basic_Engine.h"
#include "TRN4JAVA_Basic_Engine_Backend.h"
#include "TRN4JAVA_Basic_Engine_Backend_Distributed.h"
#include "TRN4JAVA_Basic_Engine_Backend_Local.h"
#include "TRN4JAVA_Basic_Engine_Backend_Remote.h"

#include "TRN4JAVA_Basic_Logging.h"
#include "TRN4JAVA_Basic_Logging_Severity.h"
#include "TRN4JAVA_Basic_Logging_Severity_Trace.h"
#include "TRN4JAVA_Basic_Logging_Severity_Debug.h"
#include "TRN4JAVA_Basic_Logging_Severity_Information.h"
#include "TRN4JAVA_Basic_Logging_Severity_Warning.h"
#include "TRN4JAVA_Basic_Logging_Severity_Error.h"

#include "TRN4JAVA_Basic_Simulation.h"
#include "TRN4JAVA_Basic_Simulation_Evaluation.h"
#include "TRN4JAVA_Basic_Simulation_Evaluation_Identifier.h"
#include "TRN4JAVA_Basic_Simulation_Identifier.h"

#include "TRN4CPP/Basic.h"
#include "Helper/Logger.h"

void Java_TRN4JAVA_Basic_00024Engine_initialize(JNIEnv *env, jclass jclazz)
{
	TRACE_LOGGER;
	try
	{
		TRN4JAVA::JNIEnv::set(env);
		TRN4CPP::Engine::initialize();
	}
	catch (std::exception &e)
	{
		ERROR_LOGGER << e.what(); env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
	}
}
void Java_TRN4JAVA_Basic_00024Engine_uninitialize(JNIEnv *env, jclass jclazz)
{
	TRACE_LOGGER;
	try
	{
		TRN4CPP::Engine::uninitialize();
	}
	catch (std::exception &e)
	{
		ERROR_LOGGER << e.what(); env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
	}
}
void Java_TRN4JAVA_Basic_00024Engine_00024Backend_00024Local_initialize(JNIEnv *env, jclass jclazz, jintArray indices)
{
	TRACE_LOGGER;
	try
	{
		TRN4JAVA::JNIEnv::set(env);
		TRN4CPP::Engine::Backend::Local::initialize(TRN4JAVA::Convert::to_unsigned_int_vector(env, indices));
	}
	catch (std::exception &e)
	{
		ERROR_LOGGER << e.what(); env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
	}
}
void Java_TRN4JAVA_Basic_00024Engine_00024Backend_00024Remote_initialize(JNIEnv *env, jclass jclazz, jstring host, jint port)
{
	TRACE_LOGGER;
	try
	{
		TRN4JAVA::JNIEnv::set(env);
		TRN4CPP::Engine::Backend::Remote::initialize(TRN4JAVA::Convert::to_string(env, host), (unsigned short)port);
	}
	catch (std::exception &e)
	{
		ERROR_LOGGER << e.what(); env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
	}
}

void Java_TRN4JAVA_Basic_00024Engine_00024Backend_00024Distributed_initialize(JNIEnv *env, jclass jclazz, jobjectArray args)
{
	TRACE_LOGGER;
	try
	{
		TRN4JAVA::JNIEnv::set(env);
		auto array = TRN4JAVA::Convert::to_string_vector(env, args);
		int argc = array.size();
		char **argv = new char *[argc];

		for (std::size_t k = 0; k < array.size(); k++)
		{
			argv[k] = const_cast<char *>(array[k].c_str());
		}
		argv[argc - 1] = NULL;
		TRN4CPP::Engine::Backend::Distributed::initialize(argc, argv);
		delete argv;
	}
	catch (std::exception &e)
	{
		ERROR_LOGGER << e.what(); env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
	}
}
void Java_TRN4JAVA_Basic_00024Logging_00024Severity_00024Trace_setup(JNIEnv *env, jclass jclazz)
{
	TRACE_LOGGER;
	try
	{
		TRN4CPP::Logging::Severity::Trace::setup();
	}
	catch (std::exception &e)
	{
		ERROR_LOGGER << e.what(); env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
	}
}
void Java_TRN4JAVA_Basic_00024Logging_00024Severity_00024Debug_setup(JNIEnv *env, jclass jclazz)
{
	TRACE_LOGGER;
	try
	{
		TRN4CPP::Logging::Severity::Debug::setup();
	}
	catch (std::exception &e)
	{
		ERROR_LOGGER << e.what(); env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
	}
}

void Java_TRN4JAVA_Basic_00024Logging_00024Severity_00024Information_setup(JNIEnv *env, jclass jclazz)
{
	TRACE_LOGGER;
	try
	{
		TRN4CPP::Logging::Severity::Information::setup();
	}
	catch (std::exception &e)
	{
		ERROR_LOGGER << e.what(); env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
	}
}
void Java_TRN4JAVA_Basic_00024Logging_00024Severity_00024Warning_setup(JNIEnv *env, jclass jclazz)
{
	TRACE_LOGGER;
	try
	{
		TRN4CPP::Logging::Severity::Warning::setup();
	}
	catch (std::exception &e)
	{
		ERROR_LOGGER << e.what(); env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
	}
}
void Java_TRN4JAVA_Basic_00024Logging_00024Severity_00024Error_setup(JNIEnv *env, jclass jclazz)
{
	TRACE_LOGGER;
	try
	{
		TRN4CPP::Logging::Severity::Error::setup();
	}
	catch (std::exception &e)
	{
		ERROR_LOGGER << e.what(); env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
	}
}

jlong Java_TRN4JAVA_Basic_00024Simulation_encode(JNIEnv *env, jclass jclazz, jobject identifier)
{
	TRACE_LOGGER;

	unsigned long long simulation_id;

	try
	{
		auto condition_number_field = env->GetFieldID(env->GetObjectClass(identifier), "condition_number", "S");
		if (condition_number_field == 0)
			throw std::invalid_argument("Can't find field condition_number");
		auto frontend_number_field = env->GetFieldID(env->GetObjectClass(identifier), "frontend_number", "S");
		if (frontend_number_field == 0)
			throw std::invalid_argument("Can't find field frontend_number");
		auto bundle_size_field = env->GetFieldID(env->GetObjectClass(identifier), "batch_number", "I");
		if (bundle_size_field == 0)
			throw std::invalid_argument("Can't find field batch_number");

		unsigned short condition_number = (unsigned short)env->GetShortField(identifier, condition_number_field);
		unsigned short frontend_number = (unsigned short)env->GetShortField(identifier, frontend_number_field);
		unsigned int batch_number = (unsigned int)env->GetIntField(identifier, bundle_size_field);

		TRN4CPP::Simulation::encode(frontend_number, condition_number, batch_number, simulation_id);
	}
	catch (std::exception &e)
	{
		ERROR_LOGGER << e.what(); env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
	}
	return (jlong)simulation_id;
 }

jobject Java_TRN4JAVA_Basic_00024Simulation_decode(JNIEnv *env, jclass _jclass, jlong simulation_id)
{
	TRACE_LOGGER;
	try
	{
		unsigned short condition_number;
		unsigned short frontend_number;
		unsigned int batch_number;


		TRN4CPP::Simulation::decode((unsigned long long)simulation_id, frontend_number, condition_number, batch_number);

		jclass identifier_class = env->FindClass("TRN4JAVA/Simulation$Identifier");
		jmethodID constructor = env->GetMethodID(identifier_class, "<init>", "void(V)");
		jobject identifier = env->NewObject(identifier_class, constructor);


		auto condition_number_field = env->GetFieldID(env->GetObjectClass(identifier), "condition_number", "S");
		if (condition_number_field == 0)
			throw std::invalid_argument("Can't find field condition_number");
		auto frontend_number_field = env->GetFieldID(env->GetObjectClass(identifier), "frontend_number", "S");
		if (frontend_number_field == 0)
			throw std::invalid_argument("Can't find field frontend_number");
		auto bundle_size_field = env->GetFieldID(env->GetObjectClass(identifier), "batch_number", "I");
		if (bundle_size_field == 0)
			throw std::invalid_argument("Can't find field batch_number");

		env->SetShortField(identifier, frontend_number_field, (jshort)frontend_number);
		env->SetShortField(identifier, condition_number_field, (jshort)condition_number);
		env->SetIntField(identifier, bundle_size_field, (jint)batch_number);

		return identifier;
	}
	catch (std::exception &e)
	{
		ERROR_LOGGER << e.what(); env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
	}
	return NULL;
}

jlong Java_TRN4JAVA_Basic_00024Simulation_00024Evaluation_encode(JNIEnv *env, jclass jclazz, jobject identifier)
{
	TRACE_LOGGER;

	unsigned long long evaluation_id;

	try
	{
		auto trial_number_field = env->GetFieldID(env->GetObjectClass(identifier), "trial_number", "S");
		if (trial_number_field == 0)
			throw std::invalid_argument("Can't find field trial_number");
		auto train_number_field = env->GetFieldID(env->GetObjectClass(identifier), "train_number", "S");
		if (train_number_field == 0)
			throw std::invalid_argument("Can't find field train_number");
		auto test_number_field = env->GetFieldID(env->GetObjectClass(identifier), "test_number", "S");
		if (test_number_field == 0)
			throw std::invalid_argument("Can't find field test_number");
		auto repeat_number_field = env->GetFieldID(env->GetObjectClass(identifier), "repeat_number", "S");
		if (repeat_number_field == 0)
			throw std::invalid_argument("Can't find field repeat_number");

		unsigned short trial_number = (unsigned short)env->GetShortField(identifier, trial_number_field);
		unsigned short train_number = (unsigned short)env->GetShortField(identifier, train_number_field);
		unsigned short test_number = (unsigned short)env->GetShortField(identifier, test_number_field);
		unsigned short repeat_number = (unsigned short)env->GetShortField(identifier, repeat_number_field);


		TRN4CPP::Simulation::Evaluation::encode(trial_number, train_number, test_number, repeat_number, evaluation_id);
	}
	catch (std::exception &e)
	{
		ERROR_LOGGER << e.what(); env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
	}
	return (jlong)evaluation_id;
}

jobject Java_TRN4JAVA_Basic_00024Simulation_00024Evaluation_decode(JNIEnv *env, jclass _jclass, jlong evaluation_id)
{
	TRACE_LOGGER;
	try
	{
		unsigned short trial_number;
		unsigned short train_number;
		unsigned short test_number;
		unsigned short repeat_number;


		TRN4CPP::Simulation::Evaluation::decode((unsigned long long)evaluation_id, trial_number, train_number, test_number, repeat_number);

		jclass identifier_class = env->FindClass("TRN4JAVA/Simulation$Evaluation$Identifier");
		jmethodID constructor = env->GetMethodID(identifier_class, "<init>", "void(V)");
		jobject identifier = env->NewObject(identifier_class, constructor);


		auto trial_number_field = env->GetFieldID(env->GetObjectClass(identifier), "trial_number", "S");
		if (trial_number_field == 0)
			throw std::invalid_argument("Can't find field trial_number");
		auto train_number_field = env->GetFieldID(env->GetObjectClass(identifier), "train_number", "S");
		if (train_number_field == 0)
			throw std::invalid_argument("Can't find field train_number");
		auto test_number_field = env->GetFieldID(env->GetObjectClass(identifier), "test_number", "S");
		if (test_number_field == 0)
			throw std::invalid_argument("Can't find field test_number");
		auto repeat_number_field = env->GetFieldID(env->GetObjectClass(identifier), "repeat_number", "S");
		if (repeat_number_field == 0)
			throw std::invalid_argument("Can't find field repeat_number");

		env->SetShortField(identifier, trial_number_field, (jshort)trial_number);
		env->SetShortField(identifier, train_number_field, (jshort)train_number);
		env->SetIntField(identifier, test_number_field, (jint)test_number);
		env->SetIntField(identifier, repeat_number_field, (jint)repeat_number);

		return identifier;
	}
	catch (std::exception &e)
	{
		ERROR_LOGGER << e.what(); env->ThrowNew(env->FindClass("java/lang/Exception"), e.what());
	}
	return NULL;
}
