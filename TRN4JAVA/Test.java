import java.util.List;
import java.util.ArrayList;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

import java.awt.Font;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.XYSeries;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.axis.*;
import org.jfree.data.xy.XYSeriesCollection;
import org.jfree.ui.ApplicationFrame;
import org.jfree.ui.RefineryUtilities;

import TRN4JAVA.Engine;
import TRN4JAVA.Engine.Backend.Local;
import TRN4JAVA.Simulation;

public class Test
{
	static
	{
		System.loadLibrary("vcomp120");
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

	
	
			System.loadLibrary("boost_chrono-vc140-mt-1_62");
			System.loadLibrary("boost_date_time-vc140-mt-1_62");
			System.loadLibrary("boost_filesystem-vc140-mt-1_62");
			System.loadLibrary("boost_thread-vc140-mt-1_62");
			System.loadLibrary("boost_program_options-vc140-mt-1_62");
			System.loadLibrary("boost_serialization-vc140-mt-1_62");
			System.loadLibrary("boost_system-vc140-mt-1_62");
			System.loadLibrary("boost_zlib-vc140-mt-1_62");
			System.loadLibrary("boost_bzip2-vc140-mt-1_62");
			System.loadLibrary("boost_log-vc140-mt-1_62");
			System.loadLibrary("boost_regex-vc140-mt-1_62");
			System.loadLibrary("boost_log_setup-vc140-mt-1_62");
			System.loadLibrary("boost_iostreams-vc140-mt-1_62");
			System.loadLibrary("boost_mpi-vc140-mt-1_62");
			
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
			System.loadLibrary("TRN4JAVA");
	}
	
	public static class PlaceCell
	{
		private final static float THRESHOLD = 0.2f;
		
		private final float K;
		private final float x;
		private final float y;
		
		public PlaceCell(float x, float y, float radius)
		{
			this.x = x;
			this.y = y;
			this.K = (float)Math.log(THRESHOLD) / (radius * radius);
			
	
		}
		
		public float activation(float x, float y)
		{
			float dx = x - this.x;
			float dy = y - this.y;
			
			return (float)Math.exp((dx*dx + dy*dy) * K);
		}
	}
	
	
	
	public static class Arena extends TRN4JAVA.Simulation.Measurement.Raw
	{
		@Override
		public void		callback(final long id, final long trial, final long evaluation, final float primed[], final float predicted [], final float expected[], final long preamble, final long batch_size, final long rows, final long cols)
		{
			/*System.out.println(id);
			System.out.println(trial);
			System.out.println(evaluation);
			System.out.println(batch_size);
			System.out.println(rows);
			System.out.println(cols);*/

			final XYSeriesCollection collection = new XYSeriesCollection();
			for (long batch = 0; batch < batch_size; batch++)
			{
				final XYSeries series = new XYSeries("Rat #" + Long.toString(batch + 1), false);
				for (long row = 0; row < rows; row++)
				{ 
					int offset = (int)(batch * rows * cols + row * cols);
					float x = predicted[offset+ 0];
					float y = predicted[offset+ 1];
					series.add(x, y);
				}
				
				collection.addSeries(series);
				
			
			}
			
			final ApplicationFrame frame = new ApplicationFrame("Plot");
			final JFreeChart chart = ChartFactory.createXYLineChart(
				"Arena",
				"X", 
				"Y", 
				collection,
				PlotOrientation.VERTICAL,
				true,
				true,
				false
			);
			XYPlot xyPlot = (XYPlot) chart.getPlot();
			ValueAxis domainAxis = xyPlot.getDomainAxis();
			ValueAxis rangeAxis = xyPlot.getRangeAxis();

			domainAxis.setRange(-1.0, 1.0);
			//domainAxis.setTickUnit(new NumberTickUnit(0.1));
			rangeAxis.setRange(-1.0, 1.0);
			//rangeAxis.setTickUnit(new NumberTickUnit(0.1));
			final ChartPanel chartPanel = new ChartPanel(chart);
			chartPanel.setPreferredSize(new java.awt.Dimension(1000, 1000));
			frame.setContentPane(chartPanel);
			
			frame.pack();
			RefineryUtilities.centerFrameOnScreen(frame);
			frame.setVisible(true);
		}
	}
	
	public static class Rat 
	{
		public static class Position extends TRN4JAVA.Simulation.Loop
		{
			private Rat rat;
			public Position(Rat rat)
			{
				this.rat = rat;
			}
			@Override
			
			public void callback(final long id, final long trial, final long evaluation, final float prediction[], final long rows, final long cols)
			{
				new Thread(new Runnable() {
           public void run() {
               			rat.move(id, trial, evaluation, prediction, rows, cols);          
    }
}).start();
	
			}
		};
		
		public static class Stimulus extends TRN4JAVA.Simulation.Loop
		{
			private Rat rat;
			public Stimulus(Rat rat)
			{
				this.rat = rat;
			}
			@Override
			public void callback(final long id, final long trial, final long evaluation, final float prediction[], final long rows, final long cols)
			{

			}
		};
		
		public final Position position;
		public final Stimulus stimulus;
		private final List<PlaceCell> place_cells;
	
		public Rat(final List<PlaceCell> place_cells)
		{
			position = new Position(this);
			stimulus = new Stimulus(this);
			this.place_cells = place_cells;
		}
		
		protected void move(final long id, final long trial, final long evaluation, final float location[], final long rows, final long cols)
		{
			assert(cols == 2);
			float activation[] = new float[(int)rows * place_cells.size()];
			for (int batch = 0; batch < rows; batch++)
			{
				for (int place_cell = 0; place_cell < place_cells.size(); place_cell++)
				{
					float x = location[batch * 2 + 0];
					float y = location[batch * 2 + 1];
					activation[batch * place_cells.size() + place_cell] = place_cells.get(place_cell).activation(x, y);
				}
			}
		
			position.notify(id, trial, evaluation, location, rows, cols);
			stimulus.notify(id, trial, evaluation, activation, rows, place_cells.size());
		}
	};
	
	
	public final static void main(String args[])
	{
		try
		{
			java.lang.Object object;
			
			if (args.length != 2)
				throw new java.lang.RuntimeException("experiment and placecells filename must me provided");
			String experiment = args[0];
			String placecells = args[1];
			
			BufferedReader br = new BufferedReader(new FileReader(placecells));
			String line = br.readLine();
			List<PlaceCell> place_cells = new ArrayList<PlaceCell>();
			
			while ((line = br.readLine()) != null) 
			{
				String[] tokens = line.split("\t");
	
				float x = Float.parseFloat(tokens[3]);
				float y = Float.parseFloat(tokens[4]);
				float radius = Float.parseFloat(tokens[5]);
				place_cells.add(new PlaceCell(x, y, radius));
			}
		
			
		
			Rat rat = new Rat(place_cells);
			Arena arena = new Arena();
			int indices[]={0};
			final long ID = 1;
			final int STIMULUS_SIZE = place_cells.size();
			final int PREDICTION_SIZE = STIMULUS_SIZE;
			final int RESERVOIR_SIZE = 1024;
			final int ROWS = 400;
			final int COLS = 400;
			final int BATCH_SIZE = 1;
			final float X_MIN = -1.0f;
			final float X_MAX = 1.0f;
			final float Y_MIN = -1.0f;
			final float Y_MAX = 1.0f;
			final float SIGMA = 1.0f;
			final float RADIUS = 0.2f;
			final float SCALE = 1e-7f;
			final float LEAK_RATE = 0.45f;
			final float LEARNING_RATE = 0.1f/RESERVOIR_SIZE;
			final float INITIAL_STATE_SCALE = 1e-2f;
			final int SNIPPETS_SIZE = 10;
			final int TIME_BUDGET = 20000;
			final long SEED = 123456789;
			final String POSITION_TAG = "POS";
			final String REWARD_TAG = "REW";
			final String INCOMING_TAG = "INC";
			final String EXPECTED_TAG = "EXP";

			final String TRAJECTORY = "spiral";
			final String TRAINING_SEQUENCES[] = {TRAJECTORY};
			final String TRAINING_SET = "simple_training_set";
			final int PREAMBLE = 10;
			final int TRIALS = 2;
			final int T = 130;
			
			final float X_RANGE = X_MAX - X_MIN;
			final float Y_RANGE = Y_MAX - Y_MIN;
			
			float trajectory[] = new float[T * 2];
			float activation[] = new float[T * STIMULUS_SIZE];
			
			for (int t = 0; t < T; t++)
			{
				float p = 2.0f * (float)Math.PI * (t/(float)(T - 1));
				float x = 0.15f * p * (float)Math.sin(p);
				float y = 0.15f * p * (float)Math.cos(p);
				
				trajectory[t * 2 + 0] = x;
				trajectory[t * 2 + 1] = y;
				for (int place_cell = 0; place_cell < STIMULUS_SIZE; place_cell++)
				{
					activation[t * STIMULUS_SIZE + place_cell] = place_cells.get(place_cell).activation(x, y);
				}
			}
			
			float response[] = new float[ROWS * COLS * STIMULUS_SIZE];
			for (int place_cell = 0; place_cell < STIMULUS_SIZE; place_cell++)
			{
				for (int row = 0; row < ROWS; row++)
				{
					float y = ((row )/ (float)(ROWS - 1))*Y_RANGE + Y_MIN;
					for (int col = 0; col < COLS; col++)
					{
						float x = (col / (float)(COLS - 1))*X_RANGE + X_MIN;
						response[place_cell * ROWS * COLS + row * COLS + col]= place_cells.get(place_cell).activation(x, y);
					}
				}
			}
			float incoming[] = new float[(T - 1) * STIMULUS_SIZE];
			float expected[] = new float[(T - 1) * STIMULUS_SIZE];
			float position[] = new float[(T - 1) * 2];
			float reward[] = new float[(T - 1) * 1];
			
			System.arraycopy(activation, 0 * STIMULUS_SIZE, incoming, 0, (T - 1) * STIMULUS_SIZE);
			System.arraycopy(activation, 1 * STIMULUS_SIZE, expected, 0, (T - 1) * STIMULUS_SIZE);
			System.arraycopy(trajectory, 0 * STIMULUS_SIZE, position, 0, (T - 1) * 2);
			java.util.Arrays.fill(reward, 1.0f);
			
			/*
			*/
			TRN4JAVA.Engine.Events.Processor.install(new TRN4JAVA.Engine.Events.Processor()
			{
				@Override
				public void callback(final int rank, final String host, final int index, final String name)
				{
					System.out.println("Processor rank " + rank + " hosted by " + host + " will use device #" + index + " named " + name);
				}
			});
			
			class Handler extends TRN4JAVA.Engine.Events.Deallocated 
			{
				private java.util.concurrent.BlockingQueue<Long> queue = new java.util.concurrent.ArrayBlockingQueue<Long>(1);
				private int expected;
				
				Handler(int expected)
				{
					this.expected = expected;
				}
				@Override
				public void callback(final long id, final int rank)
				{
					System.out.println("Simulation " + id + " deallocated from processor rank " + rank);
					
					try
					{
						queue.put(id);
					}
					catch (InterruptedException ie)
					{
						ie.printStackTrace();
					}
				}
				
				public void wait_condition()
				{
					while (expected > 0)
					{
						try
						{
							long id = queue.take();
							expected--;
						}
						catch (InterruptedException ie)
						{
							ie.printStackTrace();
						}
						
					}
				}
			}
			
			
			
			
			
			Handler deallocated_handler = new Handler(1);
		
			TRN4JAVA.Engine.Events.Deallocated.install(deallocated_handler);
			TRN4JAVA.Engine.Events.Trained.install(new TRN4JAVA.Engine.Events.Trained()
			{
				@Override
				public void callback(final long id)
				{
					System.out.println("Simulation " + id + " is trained");
				}
			});
			TRN4JAVA.Engine.Events.Tested.install(new TRN4JAVA.Engine.Events.Tested()
			{
				@Override
				public void callback(final long id)
				{
					System.out.println("Simulation " + id + " is tested");
				}
			});
			TRN4JAVA.Engine.Events.Primed.install(new TRN4JAVA.Engine.Events.Primed()
			{
				@Override
				public void callback(final long id)
				{
					System.out.println("Simulation " + id + " is primed");
				}
			});
			TRN4JAVA.Engine.Events.Completed.install(new TRN4JAVA.Engine.Events.Completed()
			{
				@Override
				public void callback()
				{
					System.out.println("Simulations completed");
				}
			});
			TRN4JAVA.Engine.Events.Allocated.install(new TRN4JAVA.Engine.Events.Allocated()
			{
				@Override
				public void callback(final long id, final int rank)
				{
					System.out.println("Simulation " + id + " allocated on processor rank " + rank);
				}
			});
			/*TRN4JAVA.Engine.Events.Ack.install(new TRN4JAVA.Engine.Events.Ack()
			{
				@Override
				public void callback(final long id, final long number, final boolean success, final String cause)
				{
					System.out.println("Simulation " + id + " ack for message " + number + " success " + success + " cause " + cause);
				}
			});*/
			TRN4JAVA.Engine.Backend.Local.initialize(indices);

			TRN4JAVA.Simulation.allocate(ID);
			
			TRN4JAVA.Simulation.configure_begin(ID);
			
			
			
			TRN4JAVA.Simulation.Scheduler.Snippets.configure(ID, SEED, SNIPPETS_SIZE, TIME_BUDGET, REWARD_TAG);
			TRN4JAVA.Simulation.Reservoir.WidrowHoff.configure(ID, STIMULUS_SIZE, PREDICTION_SIZE, RESERVOIR_SIZE, LEAK_RATE, INITIAL_STATE_SCALE, LEARNING_RATE, SEED, BATCH_SIZE);
			TRN4JAVA.Simulation.Reservoir.Weights.Feedforward.Uniform.configure(ID, -1.0f, 1.0f, 0.0f);
			TRN4JAVA.Simulation.Reservoir.Weights.Feedback.Uniform.configure(ID, -1.0f, 1.0f, 0.0f);
			TRN4JAVA.Simulation.Reservoir.Weights.Recurrent.Uniform.configure(ID, -1.0f/(float)Math.sqrt(RESERVOIR_SIZE), 1.0f/(float)Math.sqrt(RESERVOIR_SIZE), 0.0f);
			TRN4JAVA.Simulation.Reservoir.Weights.Readout.Uniform.configure(ID, -1e-3f, 1e-3f, 0.0f);
			TRN4JAVA.Simulation.Loop.SpatialFilter.configure(ID, BATCH_SIZE, STIMULUS_SIZE, SEED, rat.position, rat.stimulus,
				ROWS, COLS,			
				X_MIN, X_MAX, Y_MIN, Y_MAX, 
				response, SIGMA, RADIUS, SCALE,  POSITION_TAG);
				
				
				
			TRN4JAVA.Simulation.Recording.Performances.configure(ID, new TRN4JAVA.Simulation.Recording.Performances()
			{
				@Override	
				public void		callback(final long id, final long trial, final long evaluation, final String phase, final float cycles_per_second, final float gflops_per_second)
				{
					System.out.println("id = " + id + ", trial = " + trial + ", evaluation = " + evaluation + ", phase = " + phase + ", cycles_per_second = " + cycles_per_second + ", Gflops/s = " + gflops_per_second);
				}

			}, true, true, true);

			TRN4JAVA.Simulation.Measurement.Position.Raw.configure(ID, BATCH_SIZE, arena);
			TRN4JAVA.Simulation.configure_end(ID);
		
			
			TRN4JAVA.Simulation.declare_sequence(ID, TRAJECTORY, INCOMING_TAG, incoming, T - 1);
			TRN4JAVA.Simulation.declare_sequence(ID, TRAJECTORY, EXPECTED_TAG, expected, T - 1);
			TRN4JAVA.Simulation.declare_sequence(ID, TRAJECTORY, POSITION_TAG, position, T - 1);
			TRN4JAVA.Simulation.declare_sequence(ID, TRAJECTORY, REWARD_TAG, reward, T - 1);
			
			TRN4JAVA.Simulation.declare_set(ID, TRAINING_SET, INCOMING_TAG, TRAINING_SEQUENCES);
			TRN4JAVA.Simulation.declare_set(ID, TRAINING_SET, EXPECTED_TAG, TRAINING_SEQUENCES);
			TRN4JAVA.Simulation.declare_set(ID, TRAINING_SET, POSITION_TAG, TRAINING_SEQUENCES);
			TRN4JAVA.Simulation.declare_set(ID, TRAINING_SET, REWARD_TAG, TRAINING_SEQUENCES);
			
			for (int trial = 0; trial < TRIALS; trial++)
			{
			
				TRN4JAVA.Simulation.train(ID, TRAINING_SET, INCOMING_TAG, EXPECTED_TAG);
				TRN4JAVA.Simulation.test(ID, TRAJECTORY, INCOMING_TAG, EXPECTED_TAG, PREAMBLE, true, 0);
			}
			TRN4JAVA.Simulation.deallocate(ID);
	
			deallocated_handler.wait_condition();
			//TRN4JAVA.Engine.Execution.run();
			
			TRN4JAVA.Engine.uninitialize();
		}
		catch (Exception e)
		{
			System.err.println("Exception " + e);
		}
	}
	
}
	

