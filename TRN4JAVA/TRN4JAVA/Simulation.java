package TRN4JAVA;

public class					Simulation
{
	public static native void	declare(final String label, final float sequence[], final long rows, final long cols, final String tag);
	public static native void	compute(final String scenario_filename);

	public static abstract class	Loop
	{
	    public abstract void		callback(final long id, final long trial, final long evaluation, final float prediction[], final long rows, final long cols);
		public native void			notify(final long id, final long trial, final long evaluation, final float perception[], final long rows, final long cols);
		
		public static abstract class	Stimulus
		{
			public static native void	install(Loop loop);
		}

		public static abstract class	Position
		{
			public static native void	install(Loop loop);
		}
	}

	public static abstract class	Scheduler
	{
		public abstract void		callback(final long id, final long seed, final long trial, final float elements[], final long rows, final long cols, final int offsets[], final int durations[]);
		public native void			notify(final long id, final long trial, final int offsets[], final int durations[]);

		public static native void	install(Scheduler scheduler);

		public static abstract class	Mutator
		{
			public abstract void		callback(final long id, final long seed, final long trial, final int offsets[], final int durations[]);
			public native void			notify(final long id, final long trial, final int offsets[], final int durations[]);

			public static native void	install(Mutator mutator);
		}
	}

	public static class	Reservoir
	{
		public static abstract class	Weights
		{
			public abstract void		callback(final long id, final long seed, final long batch_size, final long rows, final long cols);
			public native void			notify(final long id, final float weights[], final long batch_size, final long rows, final long cols);	

			public static abstract class	Feedforward
			{
				static native void			install(Weights weights);		
			}

			public static abstract class	Feedback
			{
				static native void			install(Weights weights);		
			}

			public static abstract class	Recurrent
			{
				static native void			install(Weights weights);		
			}

			public static abstract class	Readout
			{
				static native void			install(Weights weights);		
			}
		}
	}

	public static class Measurement
	{
		public static abstract class	Raw
		{	
			public abstract void		callback(final long id, final long trial, final long evaluation, final float primed[], final float predicted[], final float expected[], final long preamble, final long batch_size, final long rows, final long cols);
		}

		public static abstract class	Processed
		{	
			public abstract void		callback(final long id, final long trial, final long evaluation, final float values[], final long rows, final long cols);
		}


		public static class	Readout 
		{
			public static abstract class	Raw
			{
				public static native void	install(Measurement.Raw raw);
			}

			public static abstract class	MeanSquareError
			{
				public static native void	install(Measurement.Processed mean_square_error);
			}

			public static abstract class	FrechetDistance
			{
				public static native void	install(Measurement.Processed frechet_distance);
			}
		}

		public static class	Position
		{
			public static abstract class	Raw
			{
				public static native void	install(Measurement.Raw raw);
			}

			public static abstract class	MeanSquareError
			{
				public static native void	install(Measurement.Processed mean_square_error);
			}

			public static abstract class	FrechetDistance
			{
				public static native void	install(Measurement.Processed frechet_distance);
			}
		}
	}

	public static class					Recording
	{
		public static abstract class	States
		{
			public abstract void		callback(final long id, final String phase, final String label, final long batch, final long trial, final long evaluation, final float samples[], final long rows, final long cols);
			public static native void	install(States states);
		}

		public static abstract class	Weights
		{
			public abstract void		callback(final long id, final String phase, final String label, final long batch, final long trial, final float weights[], final long rows, final long cols);
			public static native void	install(Weights weights);
		}

		public static abstract class	Performances
		{
			public abstract void		callback(final long id, final long trial, final long evaluation, final String phase, final float cycles_per_second, final float gflops_per_second);
			public static native void	install(Performances performances);
		}

		public static abstract class	Scheduling
		{
			public abstract void		callback(final long id, final long trial, final int[] offsets, final int[] durations);
			public static native void	install(Scheduling scheduling);
		}
	}
}