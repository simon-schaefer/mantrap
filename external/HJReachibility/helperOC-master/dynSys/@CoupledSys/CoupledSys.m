classdef CoupledSys < DynSys
  properties    
    % Acceleartion
    aRange
      
    % Disturbance
    dRange
    
    % Dimensions that are active
    dims
  end
  
  methods
    function obj = CoupledSys(x, aRange, dRange, dims)
      % obj = CoupledSys(x, wMax, speed, dMax, dims)
      %     Two-Agent coupled system 
      %
      % Dynamics:
      %    \dot{x}_1 = vx + d1
      %    \dot{x}_2 = vy + d2
      %    \dot{x}_3 = ax
      %    \dot{x}_4 = ay
      %         u \in [-aMax, aMax]
      %         d \in [-dMax, dMax]
      %
      % Inputs:
      %   x      - state: [xpos; ypos]
      %   aRange - maximum acceleration
      %   velocity - 2D velocity vector 
      %   dMax   - disturbance  nds
      %
      % Output:
      %   obj       - a CoupledSys object
      
      if numel(x) ~= obj.nx
        error('Initial state does not have right dimension!');
      end
      
      if ~iscolumn(x)
        x = x';
      end
      
      if nargin < 2
        aRange = [-1 1];
      end
      
      if nargin < 3
        dRange = {[0; 0]; [0; 0]};
      end
      
      if nargin < 4
        dims = 1:4;
      end

      if numel(aRange) < 2
        aRange = [-aRange; aRange];
      end
      
      if ~iscell(dRange)
        dRange = {-dRange, dRange};
      end
      
      % Basic vehicle properties
      obj.pdim = [find(dims == 1) find(dims == 2)]; % Position dimensions
      obj.vdim = [find(dims == 3) find(dims == 4)]; % Velocity dimensions
      obj.nx = length(dims);
      obj.nu = 2;
      obj.nd = 2;
      
      obj.x = x;
      obj.xhist = obj.x;
      
      obj.dRange = dRange;
      obj.aRange = aRange;
      obj.dims = dims;
    end
    
  end % end methods
end % end classdef
