classdef TN4MATLAB_toto < handle

	properties (Access = private)
	 id_ %id of the session.
	end

	methods (Static)
		function install_processor(callback)
			TRN4MATLAB('install_processor', callback)
		end
	end

end
