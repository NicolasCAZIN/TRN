package TRN4JAVA;
import java.lang.RuntimeException;
import java.io.*;
public class Engine
{
	static
	{
			/*System.loadLibrary("vcomp120");
			System.loadLibrary("msvcr120");
			System.loadLibrary("msvcp120");
	

				System.loadLibrary("vcruntime140");
			System.loadLibrary("concrt140");
				System.loadLibrary("msvcp140");

	
			System.loadLibrary("vcomp140");
	
		
			System.loadLibrary("tbb");
			System.loadLibrary("tbbmalloc");

			System.loadLibrary("libimalloc");
			System.loadLibrary("libiomp5md");
			System.loadLibrary("mkl_core");
			System.loadLibrary("mkl_intel_thread");
			System.loadLibrary("mkl_sequential");
			System.loadLibrary("mkl_tbb_thread");
			System.loadLibrary("mkl_def");
			System.loadLibrary("mkl_avx");
			System.loadLibrary("mkl_avx2");
			System.loadLibrary("mkl_avx512");
			System.loadLibrary("mkl_avx512_mic");

			System.loadLibrary("mkl_mc");
			System.loadLibrary("mkl_mc3");
			System.loadLibrary("mkl_rt");

			System.loadLibrary("mkl_vml_avx");
			System.loadLibrary("mkl_vml_avx2");
			System.loadLibrary("mkl_vml_avx512");
			System.loadLibrary("mkl_vml_avx512_mic");
			System.loadLibrary("mkl_vml_cmpt");
			System.loadLibrary("mkl_vml_def");
			System.loadLibrary("mkl_vml_mc");
			System.loadLibrary("mkl_vml_mc2");
			System.loadLibrary("mkl_vml_mc3");
	




			System.loadLibrary("hdf5");

			System.loadLibrary("icudt56");
			System.loadLibrary("icuin56");
			System.loadLibrary("icuio56");
			System.loadLibrary("icuuc56");
			System.loadLibrary("zlib1");
			System.loadLibrary("libexpat");

	
			System.loadLibrary("libmwfl");
			System.loadLibrary("libmwfoundation_usm");
			System.loadLibrary("libmwi18n");
			System.loadLibrary("libmwresource_core");
	
			System.loadLibrary("libut");
			System.loadLibrary("libmat");
			System.loadLibrary("libmx");




			System.loadLibrary("cudart64_90");
			System.loadLibrary("cublas64_90");
			System.loadLibrary("curand64_90");

			System.loadLibrary("boost_chrono-vc120-mt-1_56");
			System.loadLibrary("boost_date_time-vc120-mt-1_56");
			System.loadLibrary("boost_filesystem-vc120-mt-1_56");
			System.loadLibrary("boost_log-vc120-mt-1_56");
			System.loadLibrary("boost_regex-vc120-mt-1_56");
			System.loadLibrary("boost_serialization-vc120-mt-1_56");
			System.loadLibrary("boost_signals-vc120-mt-1_56");
			System.loadLibrary("boost_thread-vc120-mt-1_56");
			System.loadLibrary("boost_system-vc120-mt-1_56");

	
			System.loadLibrary("boost_filesystem-vc140-mt-1_62");
			System.loadLibrary("boost_iostreams-vc140-mt-1_62");
			System.loadLibrary("boost_mpi-vc140-mt-1_62");
			System.loadLibrary("boost_program_options-vc140-mt-1_62");
			System.loadLibrary("boost_serialization-vc140-mt-1_62");
			System.loadLibrary("boost_system-vc140-mt-1_62");
			System.loadLibrary("boost_zlib-vc140-mt-1_62");
			System.loadLibrary("boost_bzip2-vc140-mt-1_62");

			System.loadLibrary("Backend");
			System.loadLibrary("GPU");
			System.loadLibrary("CPU");

			System.loadLibrary("Helper");
			System.loadLibrary("Core");
			System.loadLibrary("Initializer");
			System.loadLibrary("Loop");
			System.loadLibrary("Measurement");
			System.loadLibrary("Mutator");
			System.loadLibrary("Reservoir");
			System.loadLibrary("Scheduler");
			System.loadLibrary("Simulator");
			System.loadLibrary("Model");

			System.loadLibrary("Network");

			System.loadLibrary("Engine");
			System.loadLibrary("Remote");
			System.loadLibrary("Distributed");
			System.loadLibrary("Local");
		
			System.loadLibrary("ViewModel");
			System.loadLibrary("TRN4CPP");
			System.loadLibrary("TRN4JAVA");*/
	}

	public static native void initialize();
	public static native void uninitialize();


	public static class Backend
	{
		public static class Local
		{
			public static native void initialize(final int[] indices);
		}

		public static class Remote
		{
			public static native void initialize(final String host, final int port);
		}
	}
}
