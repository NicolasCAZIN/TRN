package TRN4JAVA;

public class Engine
{
	public static native void initialize();
	public static native void uninitialize();

	public static class Events
	{
		public static abstract class Configured
		{
			public abstract void callback(final long id);
			public static native void install(final Configured configured);
		}

		public static abstract class Trained
		{
			public abstract void callback(final long id);
			public static native void install(final Trained trained);
		}

		public static abstract class Primed
		{
			public abstract void callback(final long id);
			public static native void install(final Primed primed);
		}

		public static abstract class Tested
		{
			public abstract void callback(final long id);
			public static native void install(final Tested tested);
		}

		public static abstract class Ack
		{
			public abstract void callback(final long id, final long number, final boolean success, final String cause);
			public static native void install(final Ack ack);
		}

		public static abstract class Processor
		{
			public abstract void callback(final int rank, final String host, final int index, final String name);
			public static native void install(final Processor processor);
		}
		public static abstract class Completed
		{
			public abstract void callback();
			public static native void install(final Completed completed);
		}
		public static abstract class Allocated
		{
			public abstract void callback(final long id, final int rank);
			public static native void install(final Allocated allocated);
		}

		public static abstract class Deallocated
		{
			public abstract void callback(final long id, final int rank);
			public static native void install(final Deallocated deallocated);
		}
	}

	public static class Execution
	{
		public static native void run();
	}

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
