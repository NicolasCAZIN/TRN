package TRN4JAVA;

public class Custom
{
	public static class Plugin
	{
		public static native void initialize(final String library_path, final String name, final java.util.Map<String, String> arguments);
	}

	public static class Simulation
	{
		public static abstract class	Loop
		{
			public abstract void		callback(final longrid, final long trial, final long evaluation, final float prediction[], final long rows, final long cols);
			public native void			notify(final longrid, final long trial, final long evaluation, final float perception[], final long rows, final long cols);
	
			public static class				Stimulus
			{
				public static native void	install(final Loop loop);
			}

			public static class				Position
			{
				public static native void	install(final Loop loop);
			}
		}

		public static abstract class	Scheduler
		{
			public abstract void		callback(final longrid, final long seed, final long trial, final float elements[], final long rows, final long cols, final int offsets[], final int durations[]);
			public native void			notify(final longrid, final long trial, final int offsets[], final int durations[]);

			public static native void	install(Scheduler scheduler);

			public static abstract class	Mutator
			{
				public abstract void		callback(final longrid, final long seed, final long trial, final int offsets[], final int durations[]);
				public native void			notify(final longrid, final long trial, final int offsets[], final int durations[]);

				public static native void	install(final Mutator mutator);
			}
		}

		public static class	Reservoir
		{
			public static abstract class	Weights
			{
				public abstract void		callback(final longrid, final long seed, final long batch_size, final long rows, final long cols);
				public native void			notify(final longrid, final float weights[], final long batch_size, final long rows, final long cols);	
			}

			public static abstract class	Feedforward 
			{
				static native void			install(final Weights weights);		
			}

			public static abstract class	Feedback
			{
				static native void			install(final Weights weights);		
			}

			public static abstract class	Recurrent
			{
				static native void			install(final Weights weights);		
			}

			public static abstract class	Readout
			{
				static native void			install(final Weights weights);		
			}
		}
	}
}
