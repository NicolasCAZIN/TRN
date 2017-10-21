package TRN4JAVA;

public class Simulation
{
	public static native void		declare(final String label, final float sequence[], final long rows, final long cols, final String tag);
	public static native void		compute(final String scenario_filename);

	public static abstract class Loop
	{
	    public abstract void		callback(final long id, final long trial, final long evaluation, final float prediction[], final long rows, final long cols);
		public native void			notify(final long id, final long trial, final long evaluation, final float perception[], final long rows, final long cols);

		public static class Stimulus
		{
			public static native void install(Loop loop);
		}

		public static class Position
		{
			public static native void install(Loop loop);
		}
	}
}