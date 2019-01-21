package TRN4JAVA;

public class					Basic
{
	public static class Logging
	{
		public static class Severity
		{
			public static class Trace
			{
				public static native void setup();
			}	
			public static class Debug
			{
				public static native void setup();
			}	
			public static class Information
			{
				public static native void setup();
			}	
			public static class Warning
			{
				public static native void setup();
			}	
			public static class Error
			{
				public static native void setup();
			}	
		}
	}

	public static class Engine
	{
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

			public static class Distributed
			{
				public static native void initialize(final String args[]);
			}
		}
	}

	public static class Simulation
	{
		public static class Identifier
		{
			public final short frontend_number;
			public final short condition_number;
			public final int batch_number;

			public Identifier(final short frontend_number, final short condition_number, final int batch_number)
			{
				this.frontend_number = frontend_number;
				this.condition_number = condition_number;
				this.batch_number = batch_number;
			}
		}

		public static native long	encode(final Identifier identifier);
		public static native Identifier	decode(final long simulation_id);

		public static class Evaluation
		{
			public static class Identifier
			{
				public final short trial_number;
				public final short train_number;
				public final short test_number;
				public final short repeat_number;

				public Identifier(final short trial_number, final short train_number, final short test_number,	final short repeat_number)
				{
					this.trial_number = trial_number;
					this.train_number = train_number;
					this.test_number = test_number;
					this.repeat_number = repeat_number;
				}
			}
			public static native long	encode(final Identifier identifier);
			public static native Identifier	decode(final long simulation_id);
		}
	}
}


