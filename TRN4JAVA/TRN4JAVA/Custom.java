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
			public abstract void		callback(final long simulation_id, final long evaluation_id, final float prediction[], final long rows, final long cols);
			public native void			notify(final long simulation_id, final long evaluation_id, final float perception[], final long rows, final long cols);
	
	
			public static native void	install(final Loop loop);
			
		}
		public static abstract class	Encoder
		{
			public abstract void		callback(final long simulation_id, final long evaluation_id, final float predicted_position[], final long rows, final long cols);
			public native void			notify(final long simulation_id, final long evaluation_id, final float estimated_position[], final float perceived_stimulus[], final long rows, final long cols);

			public static native void	install(final Encoder encoder);
		}
		public static abstract class	Scheduler
		{
			public abstract void		callback(final long simulation_id, final long evaluation_id,final long seed, final float elements[], final long rows, final long cols, final int offsets[], final int durations[]);
			public native void			notify(final long simulation_id, final long evaluation_id,final int offsets[], final int durations[]);

			public static native void	install(Scheduler scheduler);

			public static abstract class	Mutator
			{
				public abstract void		callback(final long simulation_id,  final long evaluation_id, final long seed, final int offsets[], final int durations[]);
				public native void			notify(final long simulation_id,  final long evaluation_id, final int offsets[], final int durations[]);

				public static native void	install(final Mutator mutator);
			}
		}

		public static class	Reservoir
		{
			public static abstract class	Weights
			{
				public abstract void		callback(final long simulation_id, final long seed, final long batch_size, final long rows, final long cols);
				public native void			notify(final long simulation_id, final float weights[], final long batch_size, final long rows, final long cols);	
			}

			public static abstract class	Feedforward 
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
