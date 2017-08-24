package TRN4JAVA;

import java.io.*;
import java.nio.file.*;

import java.awt.*;
import java.awt.image.*;
import javax.swing.*;
import java.util.Arrays;

public class Api
{
	static 
	{
		try
		{
			Path dir = Files.createTempDirectory("TRN4JAVA");
			dir.toFile().deleteOnExit();
		
			//loadLibraryFromJar(dir, "libimalloc");
			/*loadLibraryFromJar(dir, "msvcp140");
			loadLibraryFromJar(dir, "vccorlib140");
			loadLibraryFromJar(dir, "vcruntime140");
			loadLibraryFromJar(dir, "concrt140");*/
			loadLibraryFromJar(dir, "libimalloc");
			loadLibraryFromJar(dir, "impi");
			loadLibraryFromJar(dir, "impimt");
			loadLibraryFromJar(dir, "boost_serialization-vc140-mt-1_62");
			loadLibraryFromJar(dir, "boost_system-vc140-mt-1_62");
			loadLibraryFromJar(dir, "boost_chrono-vc140-mt-1_62");
			loadLibraryFromJar(dir, "boost_date_time-vc140-mt-1_62");
			loadLibraryFromJar(dir, "boost_thread-vc140-mt-1_62");
		
			loadLibraryFromJar(dir, "boost_mpi-vc140-mt-1_62");
		
			loadLibraryFromJar(dir,"cudart"+System.getProperty("sun.arch.data.model")+"_80");
			//loadLibraryFromJar(dir,"cublas"+System.getProperty("sun.arch.data.model")+"_80");
			//loadLibraryFromJar(dir,"curand"+System.getProperty("sun.arch.data.model")+"_80");
			//loadLibraryFromJar(dir,"Qt5Core");
			loadLibraryFromJar(dir, "Backend");
			loadLibraryFromJar(dir,"Core");
			loadLibraryFromJar(dir,"Simulator");
			loadLibraryFromJar(dir,"Loop");
			loadLibraryFromJar(dir,"Scheduler");
			loadLibraryFromJar(dir,"Initializer");
			loadLibraryFromJar(dir,"Reservoir");

			loadLibraryFromJar(dir,"CPU");
			loadLibraryFromJar(dir,"GPU");
	
			loadLibraryFromJar(dir,"Model");
			loadLibraryFromJar(dir,"Engine");
			loadLibraryFromJar(dir,"Local");
			loadLibraryFromJar(dir,"Network");
			loadLibraryFromJar(dir,"Remote");
			loadLibraryFromJar(dir,"Distributed");
			loadLibraryFromJar(dir,"ViewModel");
			loadLibraryFromJar(dir,"TRN4CPP");
			loadLibraryFromJar(dir,"Qt5Core");
			loadLibraryFromJar(dir,"TRN4JAVA");
		}
		catch (IOException e)
		{
		  e.printStackTrace(); 
		}
	}


	public static void loadLibraryFromJar(Path dir, String name) throws IOException 
	{

        int bits = Integer.parseInt(System.getProperty("sun.arch.data.model"));       // JRE architecture i.e 64 bit or 32 bit JRE
		String platform = "";
		switch (bits)
		{
		case 32 :
			platform = "Win32";
			break;
		case 64 :
			platform = "x64";
			break;
		}
        
		String path = "/" + platform + "/" + System.mapLibraryName(name);
		System.out.println("JRE architecture is " + System.getProperty("sun.arch.data.model") +" bits");
		 System.out.println(  "OS Architecture : " + System.getProperty("os.arch"));

        System.out.println("OS Name : " + System.getProperty("os.name"));

        System.out.println("OS Version : " + System.getProperty("os.version"));

        System.out.println("Data Model : " + System.getProperty("sun.arch.data.model"));
			System.out.println("JVM loading " + path);
        if (!path.startsWith("/")) 
		{
            throw new IllegalArgumentException("The path has to be absolute (start with '/').");
        }
 
        // Obtain filename from path
        String[] parts = path.split("/");
        String filename = (parts.length > 1) ? parts[parts.length - 1] : null;
 
        // Split filename to prexif and suffix (extension)
        String prefix = "";
        String suffix = null;
        if (filename != null) {
            parts = filename.split("\\.", 2);
            prefix = parts[0];
            suffix = (parts.length > 1) ? "."+parts[parts.length - 1] : null; // Thanks, davs! :-)
        }
 

        // Check if the filename is okay
        if (filename == null || prefix.length() < 3) {
            throw new IllegalArgumentException("The filename has to be at least 3 characters long.");
        }
 
        // Prepare temporary file


	
        File temp = new File(dir.toString(), filename);

 
		if (!temp.createNewFile()) {
            throw new FileNotFoundException("File " + temp.getAbsolutePath() + " does not exist.");
        }
	
        // Prepare buffer for data copying
        byte[] buffer = new byte[8192];
        int readBytes;
 
        // Open and check input stream
        InputStream is = TRN4JAVA.Api.class.getResourceAsStream(path);
        if (is == null) {
            throw new FileNotFoundException("File " + path + " was not found inside JAR.");
        }
 
        // Open output stream and copy data between source file in JAR and the temporary file
        OutputStream os = new FileOutputStream(temp);
        try {
            while ((readBytes = is.read(buffer)) != -1) {
                os.write(buffer, 0, readBytes);
            }
        } finally {
            // If read/write fails, close streams safely before throwing an exception
            os.close();
            is.close();
        }
 
        // Finally, load the library
		String to_load = temp.getAbsolutePath();
		System.load(to_load);
		System.out.println("JVM has loaded " + to_load);
    }

	public static abstract class	Processor
	{
		public  abstract void		callback(final int rank, final String host, final int index, final String name);
	}
	public static abstract class	Association
	{
		public  abstract void		callback(final int id, final int rank);
	}
	public static abstract class	Matrix
	{
		 public  abstract void		callback(final int id, final String label, final float elements[], final int rows, final int cols);
	}
	public static abstract class	Measurement
	{
		 public  abstract void		callback(final int id, final float elements[], final int rows, final int cols);
	}
	public static abstract class	Result
	{
		 public  abstract void		callback(final int id, final float predicted[], final float expected[], final int batch_size, final int rows, final int cols);
	}
	public static abstract class	Performances
	{
		 public  abstract void		callback(final int id, final String phase, final int batch_size, final int cycles, final float gflops, final float seconds);
	}
	public static abstract class	Loop
	{
	    public abstract void		callback(final int id, final float prediction[], final int rows, final int cols);
		public native void			notify(final int id, final float perception[], final int rows, final int cols);
	}
	public static abstract class	Scheduler
	{
		public abstract void		callback(final int id, final float elements[], final int rows, final int cols, final int offsets[], final int durations[]);
		public native void			notify(final int id, final int offsets[], final int durations[]);
	}
	public static abstract class	Initializer
	{
		public abstract void		callback(final int id, final int matrices, final int rows, final int cols);
		public native void			notify(final int id, final float weights[], final int matrices, final int rows, final int cols);
	}

	public static native void		install_processor(Processor processor);
	public static native void		install_allocation(Association association);
	public static native void		install_deallocation(Association association);

	// index = 0 for CPU
	// index > 0 for NVIDIA GPU
	public static native void		initialize_local(final int indexes[]);
	public static native void		initialize_remote(final String host, final int port);
	public static native void		initialize_distributed(final String args[]);
	public static native void		uninitialize();

	public static native void		allocate(final int id);
	public static native void		deallocate(final int id);

	public static native void		train(final int id, final String sequence, final String incoming, final String expected);
	public static native void		test(final int id, final String sequence, final String incoming, final String expected, final int preamble);

	public static native void		declare_sequence(final int id, final String label, final String tag, final float sequence[], final int observations);
	public static native void		declare_batch(final int id, final String label, final String tag, final String labels[]);

	public static native void		setup_states(final int id, Matrix states);
	public static native void		setup_weights(final int id, Matrix weights);
	public static native void		setup_performances(final int id, Performances performances);
	
	public static native void		configure_begin(final int id);
	public static native void		configure_end(final int id);
	
	public static native void		configure_measurement_readout_mean_square_error(final int id, final int batch_size, Measurement measurement);
	public static native void		configure_measurement_readout_frechet_distance(final int id, final int batch_size, Measurement measurement);
	public static native void		configure_measurement_readout_custom(final int id, final int batch_size, Result result);

	public static native void		configure_measurement_position_mean_square_error(final int id, final int batch_size, Measurement measurement);
	public static native void		configure_measurement_position_frechet_distance(final int id, final int batch_size, Measurement measurement);
	public static native void		configure_measurement_position_custom(final int id, final int batch_size, Result result);

	public static native void		configure_reservoir_widrow_hoff(final int id, final int stimulus_size, final int prediction_size, final int reservoir_size, final float leak_rate, final float initial_state_scale, final float learning_rate, final long seed, final int batch_size);
	
	public static native void		configure_loop_copy(final int id, final int batch_size, final int stimulus_size);
	public static native void		configure_loop_spatial_filter(final int id, final int batch_size, final int stimulus_size, Loop position, Loop stimulus, 
	final int rows, final int cols, final float x_min, final float x_max, 	final float y_min, final float y_max, final float response[], final float sigma, final float radius, final String tag);
	public static native void		configure_loop_custom(final int id, final int batch_size, final int stimulus_size, Loop stimulus);

	public static native void		configure_scheduler_tiled(final int id, final int epochs);
	public static native void		configure_scheduler_snippets(final int id, final int snippets_size, final int time_budget, final String tag);
	public static native void		configure_scheduler_custom(final int id, Scheduler scheduler, final String tag);

	public static native void		configure_readout_uniform(final int id, final float a, final float b, final float sparsity);
	public static native void		configure_readout_gaussian(final int id, final float mu, final float sigma);
	public static native void		configure_readout_custom(final int id, Initializer initializer);

	public static native void		configure_feedback_uniform(final int id, final float a, final float b, final float sparsity);
	public static native void		configure_feedback_gaussian(final int id, final float mu, final float sigma);
	public static native void		configure_feedback_custom(final int id, Initializer initializer);

	public static native void		configure_recurrent_uniform(final int id, final float a, final float b, final float sparsity);
	public static native void		configure_recurrent_gaussian(final int id, final float mu, final float sigma);
	public static native void		configure_recurrent_custom(final int id, Initializer initializer);

	public static native void		configure_feedforward_uniform(final int id, final float a, final float b, final float sparsity);
	public static native void		configure_feedforward_gaussian(final int id, final float mu, final float sigma);
	public static native void		configure_feedforward_custom(final int id, Initializer initializer);

	public static void display(String label, float samples[], final int rows, final int cols)
	{ 
		final int WIDTH=cols;
		final int HEIGHT=rows;

		JFrame frame = new JFrame(label);
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		 
		final BufferedImage img = new BufferedImage(WIDTH, HEIGHT, BufferedImage.TYPE_INT_RGB);
		Graphics2D g = (Graphics2D)img.getGraphics();
		for(int j = 0; j < HEIGHT; j++)
		{
			for(int i = 0; i < WIDTH; i++)
			{	
				float c = samples[j*WIDTH+i];
				if (c > 1.0f)
				 c = 1.0f;
				 else if (c < -1.0f)
				 c = -1.0f;
				float red = c >= 0.0f ? c : 0.0f;
				float green = c < 0.0f ? -c : 0.0f;
				g.setColor(new Color(red, green, 1.0f));
				g.fillRect(i, j, 1, 1);
			 }
		 }
		 JPanel panel = new JPanel() 
		 {
			@Override
			protected void paintComponent(Graphics g)
			{
				Graphics2D g2d = (Graphics2D)g;
				g2d.clearRect(0, 0, getWidth(), getHeight());
				g2d.drawImage(img, 0, 0, this);
			 }
		  };
		  panel.setPreferredSize(new Dimension(WIDTH, HEIGHT));
		  frame.getContentPane().add(panel);
		  frame.pack();
	      frame.setVisible(true);
	  }
  



	public static void main(String[] args)
	{
	//    RobotLoop robot_loop = new TRN4JAVA().new RobotLoop();

		final int BATCH_SIZE = 100;
		final int INDEX[] = {1, 2};
		final int SEED = 12345;
		final int ID = 42;
		final int RESERVOIR_SIZE = 1024;
		final int STIMULUS_SIZE = 256;
		final float LEAK_RATE = 1.0f;
		final float INITIAL_STATE_SCALE = 1e-2f;
		final float LEARNING_RATE = 1e-3f;
		final int EPOCHS = 1000;
		final int OBSERVATIONS = 130;
		float stimulus[]= new float[OBSERVATIONS * STIMULUS_SIZE];
		Arrays.fill(stimulus, 0.0f);
		for (int row = 0; row < OBSERVATIONS; row++)
		{
			stimulus[row * STIMULUS_SIZE + row] = 1.0f;
		}
		display("DATA", stimulus, OBSERVATIONS, STIMULUS_SIZE);

		float incoming[] = Arrays.copyOfRange(stimulus, 0, (OBSERVATIONS-1)*STIMULUS_SIZE);
		float expected[] = Arrays.copyOfRange(stimulus, STIMULUS_SIZE, OBSERVATIONS*STIMULUS_SIZE);
		float reward[] = new float[OBSERVATIONS-1];
		Arrays.fill(reward, 0.0f);

		TRN4JAVA.Api.initialize_local(INDEX);

		TRN4JAVA.Api.allocate(ID);
		TRN4JAVA.Api.configure_begin(ID);
		TRN4JAVA.Api.configure_reservoir_widrow_hoff(ID, STIMULUS_SIZE, STIMULUS_SIZE, RESERVOIR_SIZE, LEAK_RATE, INITIAL_STATE_SCALE, LEARNING_RATE, SEED, BATCH_SIZE);
		TRN4JAVA.Api.setup_performances(ID, new Performances()
		{
				@Override
				 public void callback(final int id, String phase, final int batch_size, final int cycles, final float gflops, final float seconds)
				 {
					System.out.println("Performances callback : phase = " + phase + ", gflops / s = " + batch_size * gflops / seconds);
				 }
		});

		/*TRN4JAVA.setup_weights(ID, new Matrix()
		{
				@Override
				 public void callback(String label, final float samples[], final int rows, final int cols)
				 {
				     display(label, samples, rows, cols);
					System.out.println("Weights callback : label = " + label + ", rows = " + rows + ", cols = " + cols);
				 }
		});*/
		/*TRN4JAVA.Api.setup_states(ID, new Matrix()
		{
				@Override
				 public void callback(final int id, String label,  final float samples[], final int rows, final int cols)
				 {
				  display(label, samples, rows, cols);
					System.out.println("States callback : label = " + label + ", rows = " + rows + ", cols = " + cols);
				 }
		});*/


	TRN4JAVA.Api.configure_loop_custom(ID, BATCH_SIZE, STIMULUS_SIZE, new Loop()
	{
		@Override
		public void callback(final int id, final float prediction[], final int rows, final int cols)
		{
			float thresholded[] = new float[prediction.length];
			for (int k = 0; k < prediction.length; k++)
			{
				if (prediction[k] > 0.5f)
					thresholded[k] = 1.0f;
				else
					thresholded[k] = 0.0f;
			}
			
			notify(id, thresholded, rows, cols);
		}
	});
		//TRN4JAVA.configure_loop_copy(ID, STIMULUS_SIZE);

		// you can provide a CSC simulator class that implements the interface or that holds a reference to a delegate class that belongs to the csc
		
		//TRN4JAVA.configure_scheduler_tiled(ID, EPOCHS);
		TRN4JAVA.Api.configure_scheduler_custom(ID, new Scheduler()
		{
			@Override
			public void callback(final int id, final float elements[], final int rows, final int cols, final int offsets[], final int durations[])
			{
				assert(offsets.length == durations.length);
				 int tiled_offsets[] = new int[EPOCHS * offsets.length];
				 int tiled_durations[] = new int[EPOCHS * offsets.length];
				 for (int l = 0; l < offsets.length; l++)
				 {
					 for (int e = 0; e < EPOCHS; e++)
					 {
						 tiled_offsets[l * EPOCHS + e] = offsets[l];
						 tiled_durations[l * EPOCHS + e] = durations[l];
					 }
				 }
				
				notify(id, tiled_offsets, tiled_durations);
			}
		},
		"INC"
		);
		TRN4JAVA.Api.configure_feedforward_uniform(ID, -1.0f, 1.0f, 0.0f);
		TRN4JAVA.Api.configure_recurrent_gaussian(ID, 0.0f, 0.5f/(float)Math.sqrt(RESERVOIR_SIZE));
		TRN4JAVA.Api.configure_feedback_uniform(ID, -1.0f, 1.0f, 0.0f);
		//TRN4JAVA.configure_readout_uniform(ID, -1e-2f, 1e-2f, 0.0f);
		
		TRN4JAVA.Api.configure_readout_custom(ID, new Initializer()
		{
			@Override
			public void callback(final int id, final int matrices,  final int rows, final int cols)
			{
				System.out.println("readout custom " + rows + " x " + cols);
				float weights[] = new float[matrices*rows * cols];
			
				 for (int k = 0; k < matrices*rows * cols; k++)
					weights[k] = (float)(Math.random() * 0.02 - 0.01);
				notify(id, weights, matrices, rows, cols);
			}
		});
		TRN4JAVA.Api.configure_end(ID);


		TRN4JAVA.Api.declare_sequence(ID, "ramp", "INC", incoming,  OBSERVATIONS-1);
		TRN4JAVA.Api.declare_sequence(ID, "ramp", "EXP", expected,  OBSERVATIONS-1);
	//TRN4JAVA.Api.declare_sequence(ID, "ramp", "REW", reward,  OBSERVATIONS-1);
		
		String labels[] = {"ramp"};
		
		TRN4JAVA.Api.declare_batch(ID, "target", "INC", labels);
		TRN4JAVA.Api.declare_batch(ID, "target", "EXP", labels);
		//TRN4JAVA.Api.declare_batch(ID, "target", "REW", labels);		
		TRN4JAVA.Api.train(ID, "target", "INC", "EXP");
		TRN4JAVA.Api.test(ID, "ramp", "INC", "EXP", 10);
		TRN4JAVA.Api.deallocate(ID);

		TRN4JAVA.Api.uninitialize();
	}
}