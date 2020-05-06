function uOpt = optCtrl(obj, ~, ~, deriv, uMode)
% uOpt = optCtrl(obj, t, x, deriv, uMode)

%% Input processing
if nargin < 5
  uMode = 'min';
end

if ~iscell(deriv)
  deriv = num2cell(deriv);
end

uOpt = cell(obj.nu, 1);

%% Optimal control
if strcmp(uMode, 'max')
  uOpt{1} = (deriv{obj.dims==1}>=0)*(obj.aRange(2)) + (deriv{obj.dims==1}<0)*obj.aRange(1);
  uOpt{2} = (deriv{obj.dims==2}>=0)*(obj.aRange(2)) + (deriv{obj.dims==2}<0)*obj.aRange(1);

elseif strcmp(uMode, 'min')
  uOpt{1} = (deriv{obj.dims==1}>=0)*(obj.aRange(1)) + (deriv{obj.dims==1}<0)*obj.aRange(2);
  uOpt{2} = (deriv{obj.dims==2}>=0)*(obj.aRange(1)) + (deriv{obj.dims==2}<0)*obj.aRange(2);
else
  error('Unknown uMode!')
end

end