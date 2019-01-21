package TRN4JAVA;

public class Extended
{
	public static class Engine
	{
		public static class				Execution
		{
			public static native void	run();
		}
	}

	public static class				Simulation
	{
		public static native void	allocate(final long simulation_id);
		public static native void	deallocate(final long simulation_id);
		public static native void	train(final long simulation_id, final long evaluation_id, final String label, final String incoming, final String expected);
		public static native void	test(final long simulation_id, final long evaluation_id, final String sequence, final String incoming, final String expected, final int preamble, final boolean autonomous, final int supplementary_generations);
		public static native void	declare_sequence(final long simulation_id, final String label,  final String tag, final float sequence[], final long observations);
		public static native void	declare_set(final long simulation_id, final String label, final String tag, final String labels[]);
		public static native void	configure_begin(final long simulation_id);
		public static native void	configure_end(final long simulation_id);

		public static  class	Loop
		{
			public static class				Copy
			{
				public static native void	configure(final long simulation_id, final long batch_size, final long stimulus_size);
			}

			public static class				Custom
			{
				public static native void	configure(final long simulation_id, final long batch_size, final long stimulus_size);
			}

			public static class				SpatialFilter
			{
				public static native void	configure(final long simulation_id, final long batch_size, final long stimulus_size, final String tag);
			}
		}
		public static class	Encoder
		{
			public static class Model
			{
				public static native void  	configure(final long simulation_id, final long batch_size, final long stimulus_size,
					final float cx[], final float cy[], final float width[]);
			};

			public static class				Custom
			{
				public static native void	configure(final long simulation_id, final long batch_size, final long stimulus_size);
			}
		}

		public static class	Decoder
		{
			public static class	Linear
			{
				public static native void  	configure(final long simulation_id, final long batch_size, final long stimulus_size,
					final float cx[], final float cy[]);
			}

			public static class Kernel
			{
				public static class Map
				{
					public static native void  	configure(
													final long simulation_id, final long batch_size, final long stimulus_size,
													
													final long rows, final long cols, 
													final float x_min, final float x_max, final float y_min, final float y_max, 
													final float sigma, final float radius,final float angle, final float scale, final long seed,final float response[]);
				}

				public static class Model
				{
					public static native void  	configure(
						final long simulation_id, final long batch_size, final long stimulus_size,
													
													final long rows, final long cols, 
													final float x_min, final float x_max, final float y_min, final float y_max, 
													final float sigma, final float radius,final float angle, final float scale, final long seed,
													final float cx[], final float cy[], final float width[]);
					
				}
			}
				
		}
		public static class	Scheduler
		{	
			public static class			Tiled
			{
				public static native void	configure(final long simulation_id, final int epochs);
			}

			public static class			Snippets
			{
				public static native void	configure(final long simulation_id, final long seed, final int snippets_size, final int time_budget, final float learn_reverse_rate, final float generate_reverse_rate, final float learning_rate, final float discount, final String tag);
			}

			public static class			Custom
			{
				public static native void	configure(final long simulation_id, final long seed, final String tag);
			}

			public static class	Mutator
			{
				public static class				Shuffle
				{
					public static native void	configure(final long simulation_id, final long seed);
				}

				public static class				Reverse
				{
					public static native void	configure(final long simulation_id, final long seed, final float rate, final long size);
				}

				public static class				Custom
				{
					public static native void	configure(final long simulation_id, final long seed);
				}
			}
		}

		public static class	Reservoir
		{
			public static class WidrowHoff
			{
				public static native void	configure(final long simulation_id, final long stimulus_size, final long prediction_size, final long reservoir_size, final float leak_rate, final float initial_state_scale, final float lerning_rate, final long seed, final long batch_size, final long mini_batch_size);
			}


			public static class Weights
			{
				public static class	Feedforward 
				{
					public static class				Gaussian
					{
						public static native void	configure(final long simulation_id, final float mu, final float sigma, final float sparsity);
					}

					public static class				Uniform
					{
						public static native void	configure(final long simulation_id, final float a, final float b, final float sparsity);
					}

					public static class				Custom
					{
						public static native void	configure(final long simulation_id);
					}
				}

				public static class	Recurrent
				{
					public static class				Gaussian
					{
						public static native void	configure(final long simulation_id, final float mu, final float sigma, final float sparsity);
					}

					public static class				Uniform
					{
						public static native void	configure(final long simulation_id, final float a, final float b, final float sparsity);
					}

					public static class				Custom
					{
						public static native void	configure(final long simulation_id);
					}
				}

				public static class	Readout
				{
					public static class				Gaussian
					{
						public static native void	configure(final long simulation_id, final float mu, final float sigma, final float sparsity);
					}

					public static class				Uniform
					{
						public static native void	configure(final long simulation_id, final float a, final float b, final float sparsity);
					}

					public static class				Custom
					{
						public static native void	configure(final long simulation_id);
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
					public static native void	configure(final long simulation_id, final long batch_size);
				}

				public static class	MeanSquareError
				{
					public static native void	configure(final long simulation_id, final long batch_size);
				}

				public static class	FrechetDistance
				{
					public static native void	configure(final long simulation_id, final long batch_size, final String norm, final String aggregator);
				}
			}

			public static class	Position
			{
				public static class	Raw
				{
					public static native void	configure(final long simulation_id, final long batch_size);
				}

				public static class	MeanSquareError
				{
					public static native void	configure(final long simulation_id, final long batch_size);
				}

				public static class	FrechetDistance
				{
					public static native void	configure(final long simulation_id, final long batch_size);
				}
			}
		}

		public static class					Recording
		{
			public static  class			States
			{
				public static native void	configure(final long simulation_id, final boolean train, final boolean prime, final boolean generate);
			}

			public static abstract class	Weights
			{
				public static native void	configure(final long simulation_id, final boolean initialize, final boolean train);
			}

			public static abstract class	Performances
			{
				public static native void	configure(final long simulation_id, final boolean train, final boolean prime, final boolean generate);
			}

			public static abstract class	Scheduling
			{
				public static native void	configure(final long simulation_id);
			}
		}
	}
}
