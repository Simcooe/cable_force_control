function outputFile = make_model_new_visual(shellOpacity, innerOpacity, showModel)
%MAKE_MODEL_NEW_VISUAL Create a colored, transparent view of model_new.
%   make_model_new_visual
%   make_model_new_visual(0.15, 0.35)
%   shellOpacity / innerOpacity are OPACITY: 0 = invisible, 1 = opaque.
%   A new SLX copy is created each time; existing files are not overwritten.
%   This function sets graphic properties only; it does not run simulation.

if nargin < 1, shellOpacity = 0.15; end
if nargin < 2, innerOpacity = 0.35; end
if nargin < 3, showModel = true; end
validateattributes(shellOpacity, {'numeric'}, ...
    {'real','finite','scalar','>=',0,'<=',1});
validateattributes(innerOpacity, {'numeric'}, ...
    {'real','finite','scalar','>=',0,'<=',1});

root = fileparts(mfilename('fullpath'));
previousFolder = pwd;
restoreFolder = onCleanup(@() cd(previousFolder)); %#ok<NASGU>
cd(root);
sourceFile = fullfile(root, 'model_new.slx');
assert(isfile(sourceFile), 'Cannot find %s', sourceFile);

model = 'model_new_visual';
suffix = 1;
while isfile(fullfile(root, [model '.slx'])) || bdIsLoaded(model)
    suffix = suffix + 1;
    model = sprintf('model_new_visual_%02d', suffix);
end
outputFile = fullfile(root, [model '.slx']);
% Initialize Simscape before creating the copy, so a local UI/service startup
% failure does not leave an unconfigured output file.
load_system('sm_lib');
copyfile(sourceFile, outputFile);
load_system(outputFile);
sys = [model '/x__25_07_22_V1_STEP_1_RIGID'];

% The actual cable graph in model_new.slx confirms this mapping:
% SID 16609 -> Spool1; 16610 -> Spool2; 16611 -> Spool3; 16612 -> Spool4.
ropeSids = [16609 16610 16611 16612];
colors = [0.05 0.35 1.00; ...  % rope 1: blue
          0.00 0.65 0.35; ...  % rope 2: green
          1.00 0.45 0.00; ...  % rope 3: orange
          0.65 0.15 0.90];     % rope 4: purple
colorNames = {'blue','green','orange','purple'};
for k = 1:4
    block = Simulink.ID.getHandle(sprintf('%s:%d', model, ropeSids(k)));
    set_param(block, 'GraphicType', 'PitchLine', ...
        'GraphicVisPropType', 'SimpleVisualProperties', ...
        'GraphicDiffuseColor', mat2str(colors(k,:), 4), ...
        'GraphicOpacity', '1');
    assert(strcmp(get_param(block, 'GraphicOpacity'), '1'));
    fprintf('Spool%d cable: %-6s RGB=%s\n', k, colorNames{k}, ...
        get_param(block, 'GraphicDiffuseColor'));
end

% Main shells: fixed upstream structure and the bending-side outer sleeve.
shells = {'Solid47','Solid46'};
for k = 1:numel(shells)
    setOpacity([sys '/' shells{k}], shellOpacity);
end

% Sliding structure and the surrounding guide supports remain distinguishable.
% These names were checked in the original model_new.slx, not inferred from
% smiData indices (its opacity indices differ from its mass indices).
innerParts = {'Solid34','L1','L2','L3','L4','L5', ...
              'R0','R5','r1','r2','r3','r4'};
for k = 1:numel(innerParts)
    setOpacity([sys '/' innerParts{k}], innerOpacity);
end

% Keep internal moving details legible through the transparent structure.
details = {'Solid35','Solid36','Solid37','Solid38','Solid40', ...
           'Solid41','Solid42','Solid43','Solid44','Solid51', ...
           'l1-','l11','l111','l2-','l22','l222','l3','l33', ...
           'l4','l5','r0','r00','r1-','r2-','r3-','r33','r4-', ...
           'r5','rr2'};
for k = 1:numel(details)
    setOpacity([sys '/' details{k}], 1);
end

save_system(model);
fprintf('Saved visualization copy: %s\n', outputFile);
fprintf('Shell opacity = %.3g; inner/support opacity = %.3g\n', ...
    shellOpacity, innerOpacity);
fprintf('Update the diagram (Ctrl+D) to refresh Mechanics Explorer.\n');
fprintf('The model uses relative CAD paths: work from %s\n', root);

if showModel
    open_system(model);
    open_system(sys);
else
    close_system(model, 0);
end
end

function setOpacity(block, value)
set_param(block, 'GraphicVisPropType', 'SimpleVisualProperties', ...
    'GraphicOpacity', num2str(value, '%.6g'));
assert(abs(str2double(get_param(block, 'GraphicOpacity')) - value) < 1e-6);
end
