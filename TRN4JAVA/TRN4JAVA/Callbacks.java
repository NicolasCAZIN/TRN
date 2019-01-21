package TRN4JAVA;

public class Callbacks
{
	public static class Plugin
	{
		public static native void initialize(final String library_path, final String name, final java.util.Map<String, String> arguments);
	}

	public static class Simulation
	{
		public static class Measurement
		{
			public static abstract class	Raw
			{	
				public abstract void		callback(final long simulation_id, final long evaluation_id, final float primed[], final float predicted[], final float expected[], final long preamble, final long batch_size, final long rows, final long cols);
			}

			public static abstract class	Processed
			{	
				public abstract void		callback(final long simulation_id, final long evaluation_id, final float values[], final long rows, final long cols);
			}

			public static class	Readout 
			{
				public static class	Raw
				{
					public static native void	install(final TRN4JAVA.Callbacks.Simulation.Measurement.Raw raw);
				}

				public static class	MeanSquareError
				{
					public static native void	install(final TRN4JAVA.Callbacks.Simulation.Measurement.Processed mean_square_error);
				}

				public static class	FrechetDistance
				{
					public static native void	install(final TRN4JAVA.Callbacks.Simulation.Measurement.Processed frechet_distance);
				}
			}

			public static class	Position
			{
				public static class	Raw
				{
					public static native void	install(final TRN4JAVA.Callbacks.Simulation.Measurement.Raw raw);
				}

				public static class	MeanSquareError
				{
					public static native void	install(final TRN4JAVA.Callbacks.Simulation.Measurement.Processed mean_square_error);
				}

				public static class	FrechetDistance
				{
					public static native void	install(final TRN4JAVA.Callbacks.Simulation.Measurement.Processed frechet_distance);
				}
			}
		}

		public static class					Recording
		{
			public static abstract class	States
			{
				public abstract void		callback(final long simulation_id, final long evaluation_id,final String phase, final String label, final long batch, final float samples[], final long rows, final long cols);
			
				public static native void	install(final TRN4JAVA.Callbacks.Simulation.Recording.States states);
			}

			public static abstract class	Weights
			{
				public abstract void		callback(final long simulation_id, final long evaluation_id, final String phase, final String label, final long batch, final float weights[], final long rows, final long cols);

				public static native void	install(final TRN4JAVA.Callbacks.Simulation.Recording.Weights weights);
			}

			public static abstract class	Performances
			{
				public abstract void		callback(final long simulation_id, final long evaluation_id, final String phase, final float cycles_per_second, final float gflops_per_second);

				public static native void	install(final TRN4JAVA.Callbacks.Simulation.Recording.Performances performances);
			}

			public static abstract class	Scheduling
			{
				public abstract void		callback(final long simulation_id, final long evaluation_id,final int[] offsets, final int[] durations);

				public static native void	install(final TRN4JAVA.Callbacks.Simulation.Recording.Scheduling scheduling);
			}
		}
	}
}
