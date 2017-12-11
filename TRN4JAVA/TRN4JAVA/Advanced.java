package TRN4JAVA;


public class Advanced
{
	public static class Engine
	{
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
	}
	public static class Simulation
	{
		public static class	Loop
		{
			public static class				Custom
			{
				public static native void	configure(final long id, final long batch_size, final long stimulus_size, 
													  final TRN4JAVA.Custom.Simulation.Loop stimulus);
			}

			public static class				SpatialFilter
			{
				public static native void	configure( final long id, final long batch_size, final long stimulus_size, final long seed,
														final TRN4JAVA.Custom.Simulation.Loop position, final TRN4JAVA.Custom.Simulation.Loop stimulus,
													  final long rows, final long cols, 
												
													  final float x_min, final float x_max, final float y_min, final float y_max, 
													  final float response[], final float sigma, final float radius, final float scale, final String tag);
			}
		}

		public static class	Scheduler
		{
			public static class			Custom
			{
				public static native void	configure(final long id, final long seed, final TRN4JAVA.Custom.Simulation.Scheduler scheduler, final String tag);
			}

			public static class	Mutator
			{
				public static class				Custom
				{
					public static native void	configure(final long id, final long seed, final TRN4JAVA.Custom.Simulation.Scheduler.Mutator mutator);
				}
			}
		}

		public static class	Reservoir
		{
			public static class	Weights
			{
				public static class	Feedforward 
				{
					public static class			Custom
					{
						public static native void	configure(final long id, final TRN4JAVA.Custom.Simulation.Reservoir.Weights weights);
					}
			
				}
				public static class	Feedback
				{
					public static class				Custom
					{
						public static native void	configure(final long id, final TRN4JAVA.Custom.Simulation.Reservoir.Weights weights);
					}
				}

				public static class	Recurrent
				{
					public static class				Custom
					{
						public static native void	configure(final long id, final TRN4JAVA.Custom.Simulation.Reservoir.Weights weights);
					}
				}

				public static class	Readout
				{
					public static class				Custom
					{
						public static native void	configure(final long id, final TRN4JAVA.Custom.Simulation.Reservoir.Weights weights);
					}
				}
			}
		}

		public static class Measurement
		{
			public static class	Readout 
			{
				public static class	Raw
				{
					public static native void	configure(final long id, final long batch_size, final TRN4JAVA.Callbacks.Simulation.Measurement.Raw raw);
				}

				public static class	MeanSquareError
				{
					public static native void	configure(final long id, final long batch_size, final TRN4JAVA.Callbacks.Simulation.Measurement.Processed mean_square_error);
				}

				public static class	FrechetDistance
				{
					public static native void	configure(final long id, final long batch_size, final TRN4JAVA.Callbacks.Simulation.Measurement.Processed frechet_distance);
				}
			}

			public static class	Position
			{
				public static class	Raw
				{
					public static native void	configure(final long id, final long batch_size, final TRN4JAVA.Callbacks.Simulation.Measurement.Raw raw);
				}

				public static class	MeanSquareError
				{
					public static native void	configure(final long id, final long batch_size, final TRN4JAVA.Callbacks.Simulation.Measurement.Processed mean_square_error);
				}

				public static class	FrechetDistance
				{
					public static native void	configure(final long id, final long batch_size, final TRN4JAVA.Callbacks.Simulation.Measurement.Processed frechet_distance);
				}
			}
		}

		public static class					Recording
		{
			public static  class	States
			{
				public static native void	configure(final long id, final TRN4JAVA.Callbacks.Simulation.Recording.States states, final boolean train, final boolean prime, final boolean generate);
			}

			public static  class	Weights
			{
				public static native void	configure(final long id, final TRN4JAVA.Callbacks.Simulation.Recording.Weights weights, final boolean initialize, final boolean train);
			}

			public static  class	Performances
			{
				public static native void	configure(final long id, final TRN4JAVA.Callbacks.Simulation.Recording.Performances performances, final boolean train, final boolean prime, final boolean generate);
			}

			public static  class	Scheduling
			{
				public static native void	configure(final long id, final TRN4JAVA.Callbacks.Simulation.Recording.Scheduling scheduling);
			}
		}
	}
}