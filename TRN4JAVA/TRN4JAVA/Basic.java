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
		public static classidentifier
		{
			public short frontend_number;
			public short condition_number;
			public int bundle_size;
		}

		public static native long	encode(finalidentifieridentifier);
		public static nativeidentifier	decode(final longrid);
	}
}


/*

package TRN4JAVA;

public class					Simulation
{
	public static classidentifier
	{
		public short frontend_number;
		public short condition_number;
		public int bundle_size;
	}

	public static native long	encode(finalidentifieridentifier);
	public static nativeidentifier	decode(final longrid);

	public static native void	declare(final String label, final float sequence[], final long rows, final long cols, final String tag);
	public static native void	compute(final String scenario_filename);

	public static native void	allocate(final longrid);
	public static native void	deallocate(final longrid);
	public static native void	train(final longrid, final String label, final String incoming, final String expected);
	public static native void	test(final longrid, final String sequence, final String incoming, final String expected, final int preamble, final boolean autonomous, final int supplementary_generations);
	public static native void	declare_sequence(final longrid, final String label,  final String tag, final float sequence[], final long observations);
	public static native void	declare_set(final longrid, final String label, final String tag, final String labels[]);
	public static native void	configure_begin(final longrid);
	public static native void	configure_end(final longrid);

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

		public static class				Copy
		{
			public static native void	configure(final longrid, final long batch_size, final long stimulus_size);
		}
		
		public static class				Custom
		{
			public static native void	configure(final longrid, final long batch_size, final long stimulus_size);
			public static native void	configure(final longrid, final long batch_size, final long stimulus_size, 
												  final Loop stimulus);

		}

		public static class				SpatialFilter
		{

			public static class Advanced
{
			public static native void	configure( final longrid, final long batch_size, final long stimulus_size, final long seed,
											final Loop position, final Loop stimulus,
												  final long rows, final long cols, 
												
												  final float x_min, final float x_max, final float y_min, final float y_max, 
												  final float response[], final float sigma, final float radius, final float scale, final String tag);
}
			public static native void	configure(final longrid, final long batch_size, final long stimulus_size, final long seed,
												  final long rows, final long cols, 
												  final float x_min, final float x_max, final float y_min, final float y_max, 
												  final float response[], final float sigma, final float radius, final float scale, final String tag);
		}
	}

	public static abstract class	Scheduler
	{
		public abstract void		callback(final longrid, final long seed, final long trial, final float elements[], final long rows, final long cols, final int offsets[], final int durations[]);
		public native void			notify(final longrid, final long trial, final int offsets[], final int durations[]);

		public static native void	install(Scheduler scheduler);

		public static class			Tiled
		{
			public static native void	configure(final longrid, final int epochs);
		}

		public static class			Snippets
		{
			public static native void	configure(final longrid, final long seed, final int snippets_size, final int time_budget, final String tag);
		}

		public static class			Custom
		{
			public static native void	configure(final longrid, final long seed, final String tag);
			public static native void	configure(final longrid, final long seed, final Scheduler scheduler, final String tag);
		}

		public static abstract class	Mutator
		{
			public abstract void		callback(final longrid, final long seed, final long trial, final int offsets[], final int durations[]);
			public native void			notify(final longrid, final long trial, final int offsets[], final int durations[]);

			public static native void	install(final Mutator mutator);

			public static class				Shuffle
			{
				public static native void	configure(final longrid, final long seed);
			}

			public static class				Reverse
			{
				public static native void	configure(final longrid, final long seed, final float rate, final long size);
			}

			public static class				Custom
			{
				public static native void	configure(final longrid, final long seed);
				public static native void	configure(final longrid, final long seed, final Mutator mutator);
			}
		}
	}
public class Plugin
{

	public static class Custom
	{
		public static native void initialize(final String library_path, final String name, final java.util.Map<String, String> arguments);
	}
	public static class Callbacks
	{
		public static native void initialize(final String library_path, final String name, final java.util.Map<String, String> arguments);
	}
}
	public static class	Reservoir
	{
		public static class WidrowHoff
		{
			public static native void	configure(final longrid, final long stimulus_size, final long prediction_size, final long reservoir_size, final float leak_rate, final float initial_state_scale, final float lerning_rate, final long seed, final long batch_size);
		}

		public static abstract class	Weights
		{
			public abstract void		callback(final longrid, final long seed, final long batch_size, final long rows, final long cols);
			public native void			notify(final longrid, final float weights[], final long batch_size, final long rows, final long cols);	

			public static abstract class	Feedforward 
			{
				static native void			install(final Weights weights);		

				public static class				Gaussian
				{
					public static native void	configure(final longrid, final float mu, final float sigma);
				}

				public static class				Uniform
				{
					public static native void	configure(final longrid, final float a, final float b, final float sparsity);
				}

				public static class				Custom
				{
					public static native void	configure(final longrid);
					public static native void	configure(final longrid, final Weights weights);
				}
			}

			public static abstract class	Feedback
			{
				static native void			install(final Weights weights);		

				public static class				Gaussian
				{
					public static native void	configure(final longrid, final float mu, final float sigma);
				}

				public static class				Uniform
				{
					public static native void	configure(final longrid, final float a, final float b, final float sparsity);
				}

				public static class				Custom
				{
					public static native void	configure(final longrid);
					public static native void	configure(final longrid, final Weights weights);
				}
			}

			public static abstract class	Recurrent
			{
				static native void			install(final Weights weights);		

				public static class				Gaussian
				{
					public static native void	configure(final longrid, final float mu, final float sigma);
				}

				public static class				Uniform
				{
					public static native void	configure(final longrid, final float a, final float b, final float sparsity);
				}

				public static class				Custom
				{
					public static native void	configure(final longrid);
					public static native void	configure(final longrid, final Weights weights);
				}
			}

			public static abstract class	Readout
			{
				static native void			install(final Weights weights);		

				public static class				Gaussian
				{
					public static native void	configure(final longrid, final float mu, final float sigma);
				}

				public static class				Uniform
				{
					public static native void	configure(final longrid, final float a, final float b, final float sparsity);
				}

				public static class				Custom
				{
					public static native void	configure(final longrid);
					public static native void	configure(final longrid, final Weights weights);
				}
			}
		}
	}

	public static class Measurement
	{
		public static abstract class	Raw
		{	
			public abstract void		callback(final longrid, final long trial, final long evaluation, final float primed[], final float predicted[], final float expected[], final long preamble, final long batch_size, final long rows, final long cols);
		}

		public static abstract class	Processed
		{	
			public abstract void		callback(final longrid, final long trial, final long evaluation, final float values[], final long rows, final long cols);
		}


		public static class	Readout 
		{
			public static class	Raw
			{
				public static native void	install(final Measurement.Raw raw);

				public static native void	configure(final longrid, final long batch_size);
				public static native void	configure(final longrid, final long batch_size, final Measurement.Raw raw);
			}

			public static class	MeanSquareError
			{
				public static native void	install(final Measurement.Processed mean_square_error);

				public static native void	configure(final longrid, final long batch_size);
				public static native void	configure(final longrid, final long batch_size, final Measurement.Processed mean_square_error);
			}

			public static class	FrechetDistance
			{
				public static native void	install(final Measurement.Processed frechet_distance);

				public static native void	configure(final longrid, final long batch_size);
				public static native void	configure(final longrid, final long batch_size, final Measurement.Processed frechet_distance);
			}
		}

		public static class	Position
		{
			public static class	Raw
			{
				public static native void	install(final Measurement.Raw raw);

				public static native void	configure(final longrid, final long batch_size);
				public static native void	configure(final longrid, final long batch_size, final Measurement.Raw raw);
			}

			public static class	MeanSquareError
			{
				public static native void	install(final Measurement.Processed mean_square_error);

				public static native void	configure(final longrid, final long batch_size);
				public static native void	configure(final longrid, final long batch_size, final Measurement.Processed mean_square_error);
			}

			public static class	FrechetDistance
			{
				public static native void	install(final Measurement.Processed frechet_distance);

				public static native void	configure(final longrid, final long batch_size);
				public static native void	configure(final longrid, final long batch_size, final Measurement.Processed frechet_distance);
			}
		}
	}

	public static class					Recording
	{
		public static abstract class	States
		{
			public abstract void		callback(final longrid, final String phase, final String label, final long batch, final long trial, final long evaluation, final float samples[], final long rows, final long cols);
			
			public static native void	install(final States states);
			public static native void	configure(final longrid, final boolean train, final boolean prime, final boolean generate);
			public static native void	configure(final longrid, final States states, final boolean train, final boolean prime, final boolean generate);
		}

		public static abstract class	Weights
		{
			public abstract void		callback(final longrid, final String phase, final String label, final long batch, final long trial, final float weights[], final long rows, final long cols);

			public static native void	install(final Weights weights);
			public static native void	configure(final longrid, final boolean initialize, final boolean train);
			public static native void	configure(final longrid, final Weights weights, final boolean initialize, final boolean train);
		}

		public static abstract class	Performances
		{
			public abstract void		callback(final longrid, final long trial, final long evaluation, final String phase, final float cycles_per_second, final float gflops_per_second);

			public static native void	install(final Performances performances);
			public static native void	configure(final longrid, final boolean train, final boolean prime, final boolean generate);
			public static native void	configure(final longrid, final Performances performances, final boolean train, final boolean prime, final boolean generate);
		}

		public static abstract class	Scheduling
		{
			public abstract void		callback(final longrid, final long trial, final int[] offsets, final int[] durations);

			public static native void	install(final Scheduling scheduling);
			public static native void	configure(final longrid);
			public static native void	configure(final longrid, final Scheduling scheduling);
		}
	}
}

*/