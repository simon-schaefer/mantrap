function dx = dynamics(obj, ~, x, u, d)
% Dynamics of the Coupled System
      %    \dot{x}_1 = vx + d1
      %    \dot{x}_2 = vy + d2
      %    \dot{x}_3 = ax
      %    \dot{x}_4 = ay
%   Control: u = ax, ay;

if nargin < 5
  d = [0; 0];
end

if iscell(x)
  dx = cell(length(obj.dims), 1);
  for i = 1:length(obj.dims)
    dx{i} = dynamics_cell_helper(x, u, d, obj.dims, obj.dims(i));
  end
else
  dx = zeros(obj.nx, 1);
  
  dx(1) = x(3) + d(1);
  dx(2) = x(4) + d(2);
  dx(3) = u(1);
  dx(4) = u(2);
end
end

function dx = dynamics_cell_helper(x, u, d, dims, dim)

switch dim
  case 1
    dx = x{dims==3} + d{1};
  case 2
    dx = x{dims==4} + d{2};
  case 3
    dx = u{dims==1};
  case 4
    dx = u{dims==2};
  otherwise
    error('Only dimension 1-4 are defined for dynamics of CoupledSys!')
end
end