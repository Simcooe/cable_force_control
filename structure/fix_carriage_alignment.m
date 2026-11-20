function outputFile = fix_carriage_alignment()
% Correct Transform46 in a new copy of the latest Solid5 source assembly.
% Original SLX/autosave/STEP files are retained. No dimensions are changed.
root = fileparts(mfilename('fullpath'));
sourceFile = fullfile(root,'model_new_visual.slx.autosave');
outputFile = fullfile(root,'model_new_visual_aligned.slx');
assert(isfile(sourceFile),'Latest source autosave is missing.');
assert(~isfile(outputFile),'Aligned copy already exists; inspect it before creating another.');
oldFolder = pwd;
folderCleanup = onCleanup(@() cd(oldFolder)); %#ok<NASGU>
cd(root);
cache = fullfile(root,'four_cord_sim','.cache','matlab_carriage_alignment');
oldConfig = Simulink.fileGenControl('getConfig');
fileCleanup = onCleanup(@() Simulink.fileGenControl('set', ...
    'CacheFolder',oldConfig.CacheFolder,'CodeGenFolder',oldConfig.CodeGenFolder)); %#ok<NASGU>
Simulink.fileGenControl('set','CacheFolder',cache,'CodeGenFolder',cache,'createDir',true);
run(fullfile(root,'model_new_DataFile.m'));
assignin('base','smiData',smiData);
assignin('base','Sig',Sig);
load_system('sm_lib');
copyfile(sourceFile,outputFile);
model = 'model_new_visual_aligned';
load_system(outputFile);
modelCleanup = onCleanup(@() close_system(model,0)); %#ok<NASGU>
block = Simulink.ID.getHandle([model ':17365']);
assert(strcmp(get_param(block,'Name'),'Transform46'));
assert(strcmp(get_param(block,'TranslationMethod'),'Cartesian'));
before = get_param(block,'TranslationCartesianOffset');
assert(isequal(str2num(before),[5.54 1.25 2.1])); %#ok<ST2NM>
set_param(block,'TranslationCartesianOffset','[5.54 1.25 2.2]');
originalExplorer = get_param(model,'SimMechanicsOpenEditorOnUpdate');
set_param(model,'SimMechanicsOpenEditorOnUpdate','off');
set_param(model,'SimulationCommand','update');
set_param(model,'SimMechanicsOpenEditorOnUpdate',originalExplorer);
assert(strcmp(get_param(block,'TranslationCartesianOffset'),'[5.54 1.25 2.2]'));
save_system(model,outputFile);
fprintf('CARRIAGE_ALIGNMENT_SOURCE_PASS\n%s\n%s: %s -> [5.54 1.25 2.2] mm\n', ...
    outputFile,getfullname(block),before);
end
