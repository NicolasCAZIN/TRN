/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
/* Header for class TRN4JAVA_Api */

#ifndef _Included_TRN4JAVA_Api
#define _Included_TRN4JAVA_Api
#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     TRN4JAVA_Api
 * Method:    initialize_local
 * Signature: (II)V
 */
JNIEXPORT void JNICALL Java_TRN4JAVA_Api_initialize_1local
  (JNIEnv *, jclass, jint, jint);

/*
 * Class:     TRN4JAVA_Api
 * Method:    initialize_remote
 * Signature: (Ljava/lang/String;I)V
 */
JNIEXPORT void JNICALL Java_TRN4JAVA_Api_initialize_1remote
  (JNIEnv *, jclass, jstring, jint);

/*
 * Class:     TRN4JAVA_Api
 * Method:    initialize_distributed
 * Signature: ([Ljava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_TRN4JAVA_Api_initialize_1distributed
  (JNIEnv *, jclass, jobjectArray);

/*
 * Class:     TRN4JAVA_Api
 * Method:    allocate
 * Signature: (I)V
 */
JNIEXPORT void JNICALL Java_TRN4JAVA_Api_allocate
  (JNIEnv *, jclass, jint);

/*
 * Class:     TRN4JAVA_Api
 * Method:    deallocate
 * Signature: (I)V
 */
JNIEXPORT void JNICALL Java_TRN4JAVA_Api_deallocate
  (JNIEnv *, jclass, jint);

/*
 * Class:     TRN4JAVA_Api
 * Method:    train
 * Signature: (ILjava/lang/String;Ljava/lang/String;Ljava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_TRN4JAVA_Api_train
  (JNIEnv *, jclass, jint, jstring, jstring, jstring);

/*
 * Class:     TRN4JAVA_Api
 * Method:    test
 * Signature: (ILjava/lang/String;Ljava/lang/String;Ljava/lang/String;I)V
 */
JNIEXPORT void JNICALL Java_TRN4JAVA_Api_test
  (JNIEnv *, jclass, jint, jstring, jstring, jstring, jint);

/*
 * Class:     TRN4JAVA_Api
 * Method:    declare_sequence
 * Signature: (ILjava/lang/String;Ljava/lang/String;[FI)V
 */
JNIEXPORT void JNICALL Java_TRN4JAVA_Api_declare_1sequence
  (JNIEnv *, jclass, jint, jstring, jstring, jfloatArray, jint);

/*
 * Class:     TRN4JAVA_Api
 * Method:    declare_batch
 * Signature: (ILjava/lang/String;Ljava/lang/String;[Ljava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_TRN4JAVA_Api_declare_1batch
  (JNIEnv *, jclass, jint, jstring, jstring, jobjectArray);

/*
 * Class:     TRN4JAVA_Api
 * Method:    setup_states
 * Signature: (ILTRN4JAVA/Api/Matrix;)V
 */
JNIEXPORT void JNICALL Java_TRN4JAVA_Api_setup_1states
  (JNIEnv *, jclass, jint, jobject);

/*
 * Class:     TRN4JAVA_Api
 * Method:    setup_weights
 * Signature: (ILTRN4JAVA/Api/Matrix;)V
 */
JNIEXPORT void JNICALL Java_TRN4JAVA_Api_setup_1weights
  (JNIEnv *, jclass, jint, jobject);

/*
 * Class:     TRN4JAVA_Api
 * Method:    setup_performances
 * Signature: (ILTRN4JAVA/Api/Performances;)V
 */
JNIEXPORT void JNICALL Java_TRN4JAVA_Api_setup_1performances
  (JNIEnv *, jclass, jint, jobject);

/*
 * Class:     TRN4JAVA_Api
 * Method:    configure_begin
 * Signature: (I)V
 */
JNIEXPORT void JNICALL Java_TRN4JAVA_Api_configure_1begin
  (JNIEnv *, jclass, jint);

/*
 * Class:     TRN4JAVA_Api
 * Method:    configure_end
 * Signature: (I)V
 */
JNIEXPORT void JNICALL Java_TRN4JAVA_Api_configure_1end
  (JNIEnv *, jclass, jint);

/*
 * Class:     TRN4JAVA_Api
 * Method:    configure_reservoir_widrow_hoff
 * Signature: (IIIIFFF)V
 */
JNIEXPORT void JNICALL Java_TRN4JAVA_Api_configure_1reservoir_1widrow_1hoff
  (JNIEnv *, jclass, jint, jint, jint, jint, jfloat, jfloat, jfloat);

/*
 * Class:     TRN4JAVA_Api
 * Method:    configure_loop_copy
 * Signature: (II)V
 */
JNIEXPORT void JNICALL Java_TRN4JAVA_Api_configure_1loop_1copy
  (JNIEnv *, jclass, jint, jint);

/*
 * Class:     TRN4JAVA_Api
 * Method:    configure_loop_spatial_filter
 * Signature: (IILTRN4JAVA/Api/Loop;LTRN4JAVA/Api/Loop;IIFFFF[FFFLjava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_TRN4JAVA_Api_configure_1loop_1spatial_1filter
  (JNIEnv *, jclass, jint, jint, jobject, jobject, jint, jint, jfloat, jfloat, jfloat, jfloat, jfloatArray, jfloat, jfloat, jstring);

/*
 * Class:     TRN4JAVA_Api
 * Method:    configure_loop_custom
 * Signature: (IILTRN4JAVA/Api/Loop;)V
 */
JNIEXPORT void JNICALL Java_TRN4JAVA_Api_configure_1loop_1custom
  (JNIEnv *, jclass, jint, jint, jobject);

/*
 * Class:     TRN4JAVA_Api
 * Method:    configure_scheduler_tiled
 * Signature: (II)V
 */
JNIEXPORT void JNICALL Java_TRN4JAVA_Api_configure_1scheduler_1tiled
  (JNIEnv *, jclass, jint, jint);

/*
 * Class:     TRN4JAVA_Api
 * Method:    configure_scheduler_snippets
 * Signature: (IIILjava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_TRN4JAVA_Api_configure_1scheduler_1snippets
  (JNIEnv *, jclass, jint, jint, jint, jstring);

/*
 * Class:     TRN4JAVA_Api
 * Method:    configure_scheduler_custom
 * Signature: (ILTRN4JAVA/Api/Scheduler;Ljava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_TRN4JAVA_Api_configure_1scheduler_1custom
  (JNIEnv *, jclass, jint, jobject, jstring);

/*
 * Class:     TRN4JAVA_Api
 * Method:    configure_readout_uniform
 * Signature: (IFFF)V
 */
JNIEXPORT void JNICALL Java_TRN4JAVA_Api_configure_1readout_1uniform
  (JNIEnv *, jclass, jint, jfloat, jfloat, jfloat);

/*
 * Class:     TRN4JAVA_Api
 * Method:    configure_readout_gaussian
 * Signature: (IFF)V
 */
JNIEXPORT void JNICALL Java_TRN4JAVA_Api_configure_1readout_1gaussian
  (JNIEnv *, jclass, jint, jfloat, jfloat);

/*
 * Class:     TRN4JAVA_Api
 * Method:    configure_readout_custom
 * Signature: (ILTRN4JAVA/Api/Initializer;)V
 */
JNIEXPORT void JNICALL Java_TRN4JAVA_Api_configure_1readout_1custom
  (JNIEnv *, jclass, jint, jobject);

/*
 * Class:     TRN4JAVA_Api
 * Method:    configure_feedback_uniform
 * Signature: (IFFF)V
 */
JNIEXPORT void JNICALL Java_TRN4JAVA_Api_configure_1feedback_1uniform
  (JNIEnv *, jclass, jint, jfloat, jfloat, jfloat);

/*
 * Class:     TRN4JAVA_Api
 * Method:    configure_feedback_gaussian
 * Signature: (IFF)V
 */
JNIEXPORT void JNICALL Java_TRN4JAVA_Api_configure_1feedback_1gaussian
  (JNIEnv *, jclass, jint, jfloat, jfloat);

/*
 * Class:     TRN4JAVA_Api
 * Method:    configure_feedback_custom
 * Signature: (ILTRN4JAVA/Api/Initializer;)V
 */
JNIEXPORT void JNICALL Java_TRN4JAVA_Api_configure_1feedback_1custom
  (JNIEnv *, jclass, jint, jobject);

/*
 * Class:     TRN4JAVA_Api
 * Method:    configure_recurrent_uniform
 * Signature: (IFFF)V
 */
JNIEXPORT void JNICALL Java_TRN4JAVA_Api_configure_1recurrent_1uniform
  (JNIEnv *, jclass, jint, jfloat, jfloat, jfloat);

/*
 * Class:     TRN4JAVA_Api
 * Method:    configure_recurrent_gaussian
 * Signature: (IFF)V
 */
JNIEXPORT void JNICALL Java_TRN4JAVA_Api_configure_1recurrent_1gaussian
  (JNIEnv *, jclass, jint, jfloat, jfloat);

/*
 * Class:     TRN4JAVA_Api
 * Method:    configure_recurrent_custom
 * Signature: (ILTRN4JAVA/Api/Initializer;)V
 */
JNIEXPORT void JNICALL Java_TRN4JAVA_Api_configure_1recurrent_1custom
  (JNIEnv *, jclass, jint, jobject);

/*
 * Class:     TRN4JAVA_Api
 * Method:    configure_feedforward_uniform
 * Signature: (IFFF)V
 */
JNIEXPORT void JNICALL Java_TRN4JAVA_Api_configure_1feedforward_1uniform
  (JNIEnv *, jclass, jint, jfloat, jfloat, jfloat);

/*
 * Class:     TRN4JAVA_Api
 * Method:    configure_feedforward_gaussian
 * Signature: (IFF)V
 */
JNIEXPORT void JNICALL Java_TRN4JAVA_Api_configure_1feedforward_1gaussian
  (JNIEnv *, jclass, jint, jfloat, jfloat);

/*
 * Class:     TRN4JAVA_Api
 * Method:    configure_feedforward_custom
 * Signature: (ILTRN4JAVA/Api/Initializer;)V
 */
JNIEXPORT void JNICALL Java_TRN4JAVA_Api_configure_1feedforward_1custom
  (JNIEnv *, jclass, jint, jobject);

#ifdef __cplusplus
}
#endif
#endif