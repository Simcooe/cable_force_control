function export_simscape()
% Measure the original assembly at t=0 and export its CAD reference frames.
root = fileparts(fileparts(mfilename('fullpath')));
source = fullfile(fileparts(root),'structure');
model = 'model_new_visual_aligned';
assert(~bdIsLoaded(model),'Run in a fresh MATLAB session to protect open edits.');
oldpwd = pwd; restoreDir = onCleanup(@()cd(oldpwd));
run(fullfile(source,'model_new_DataFile.m'));
assignin('base','smiData',smiData); assignin('base','Sig',Sig);
cache = tempname; mkdir(cache);
oldPath=path; restorePath=onCleanup(@()path(oldPath));
addpath(source); cd(cache);
copyfile(fullfile(source,[model '.slx']),fullfile(cache,[model '.slx']));
settings = Simulink.fileGenControl('getConfig');
restoreCache = onCleanup(@()Simulink.fileGenControl('setConfig','config',settings));
Simulink.fileGenControl('set','CacheFolder',cache,'CodeGenFolder',cache,'createDir',true);
load_system(model); closeSource = onCleanup(@()close_system(model,0));
set_param(model,'SimMechanicsOpenEditorOnUpdate','off');
blocks = find_system(model,'LookUnderMasks','all','FollowLinks','off','Type','Block');
% Bind every CAD asset to the current project before compiling the copy.
% The source SLX contains one historical absolute filename.
for i=1:numel(blocks)
    if strcmp(sourceType(blocks{i}),'File Solid')
        [~,stem,ext]=fileparts(get_param(blocks{i},'ExtGeomFileName'));
        local=fullfile(source,[stem ext]);
        assert(isfile(local),'Missing project CAD file %s',local);
        set_param(blocks{i},'ExtGeomFileName',local);
    end
end
world = find_system(model,'SearchDepth',1,'Name','World');
world = world{1};
% Read absolute joint positions and actual cable contact angles from the
% unchanged mechanical network, adding only signal readouts in memory.
signals = struct;
for i=1:numel(blocks)
    b=blocks{i}; type=sourceType(b); sid=get_param(b,'SID');
    if endsWith(type,' Joint')
        unit='rad'; if contains(type,'Prismatic'), unit='m'; end
        signals=measure(signals,b,sid,'position','SensePosition',unit);
    elseif strcmp(type,'Pulley')
        signals=measure(signals,b,sid,'wrap','SenseWrapAngle','rad');
        signals=measure(signals,b,sid,'a','SensePulleyAngleA','rad');
        signals=measure(signals,b,sid,'b','SensePulleyAngleB','rad');
    elseif strcmp(type,'Belt-Cable Spool')
        signals=measure(signals,b,sid,'a','SenseSpoolAngleA','rad');
    end
end
fprintf('Sampling original Simscape model at t=0\n');
out=sim(model,'StopTime','0','ReturnWorkspaceOutputs','on');
names=fieldnames(signals);
for i=1:numel(names)
    ts=out.get(names{i}); signals.(names{i})=double(getdatasamples(ts,1));
end
% KinematicsSolver evaluates named frames at those measured joint positions.
origin=measurementOrigin(model,world);
ks=simscape.multibody.KinematicsSolver(model);
joints=jointPositionVariables(ks); addTargetVariables(ks,joints.ID);
q=zeros(height(joints),1);
for i=1:height(joints)
    if endsWith(joints.BlockPath(i),'/Measurement origin'), continue; end
    sid=get_param(char(joints.BlockPath(i)),'SID');
    q(i)=signals.(['s' sid '_position']);
    if joints.Unit(i)=="deg", q(i)=rad2deg(q(i)); end
end
% A separate grounded fixture gives the F1-to-CAD transform for solids whose
% R port is hidden. Original assembly ports and connections stay intact.
fixture='baseline_frame_calibration'; new_system(fixture);
closeFixture=onCleanup(@()close_system(fixture,0));
wf=add_block(sprintf('sm_lib/Frames and\nTransforms/World Frame'),[fixture '/World']);
wh=get_param(wf,'PortHandles');
solver=add_block(sprintf('nesl_utility/Solver\nConfiguration'),[fixture '/Solver']);
sh=get_param(solver,'PortHandles'); add_line(fixture,wh.RConn(1),sh.RConn(1));
poses=struct; parts=struct; hidden={}; frames={};
for i=1:numel(blocks)
    b=blocks{i}; type=sourceType(b); sid=get_param(b,'SID'); key=['s' sid];
    if any(strcmp(type,{'File Solid','Revolved Solid'}))
        port='R';
        if strcmp(get_param(b,'DoExposeReferenceFrame'),'off')
            port='F1'; copy=[fixture '/' key];
            h=add_block(b,copy); set_param(h,'DoExposeReferenceFrame','on');
            ph=get_param(h,'PortHandles'); add_line(fixture,wh.RConn(1),ph.RConn(1));
            hidden(end+1,:)={key,copy};
        end
        fields={'GraphicDiffuseColor','GraphicOpacity'};
        if strcmp(type,'File Solid')
            fields=[fields {'Mass','CenterOfMass','MomentsOfInertia','ProductsOfInertia'}];
        else
            fields=[fields {'RevolutionCrossSection','Density'}];
        end
        params=struct;
        for f=fields
            params.(f{1})=eval(get_param(b,f{1}));
        end
        parts.(key)=params;
        frames(end+1,:)={key,[b '/' port]};
    elseif endsWith(type,' Joint')
        frames(end+1,:)={[key '_base'],[b '/B']};
        frames(end+1,:)={key,[b '/F']};
    elseif any(strcmp(type,{'Pulley','Belt-Cable Spool','Belt-Cable End'}))
        frames(end+1,:)={key,[b '/R']};
    end
end
fprintf('Measuring %d assembly frames and %d local CAD offsets\n',size(frames,1),size(hidden,1));
poses=framePoses(ks,repmat({origin},size(frames,1),1),frames(:,2),frames(:,1),q);
originCal=measurementOrigin(fixture,[fixture '/World']);
ksCal=simscape.multibody.KinematicsSolver(fixture);
calJoint=jointPositionVariables(ksCal); addTargetVariables(ksCal,calJoint.ID);
calKeys=[strcat(hidden(:,1),'f');strcat(hidden(:,1),'r')];
calFrames=[strcat(hidden(:,2),'/F1');strcat(hidden(:,2),'/R')];
locals=framePoses(ksCal,repmat({originCal},size(calKeys)),calFrames,calKeys,0);
for i=1:size(hidden,1)
    key=hidden{i,1}; poses.(key)=poses.(key)*(locals.([key 'f'])\locals.([key 'r']));
end
snapshot=struct('model',model,'matlab',version('-release'),'poses_m',poses,'parts',parts,'signals',signals);
snapshot.slx_sha256=fileHash(fullfile(source,[model '.slx']));
snapshot.parameters_sha256=fileHash(fullfile(source,'model_new_DataFile.m'));
cad={};
for i=1:numel(blocks)
    if strcmp(sourceType(blocks{i}),'File Solid')
        [~,stem,ext]=fileparts(get_param(blocks{i},'ExtGeomFileName'));
        cad{end+1}=[stem ext];
    end
end
cad=unique(cad);
snapshot.cad=cellfun(@(f)struct('file',f,'sha256',fileHash(fullfile(source,f))),cad,'UniformOutput',false);
outfile=fullfile(root,'model','simscape.json');
temp=[tempname(fileparts(outfile)) '.json']; fid=fopen(temp,'w','n','UTF-8');
fprintf(fid,'%s\n',jsonencode(snapshot)); fclose(fid); movefile(temp,outfile,'f');
fprintf('Exported %s\n',outfile);
end

function origin=measurementOrigin(model,world)
% KinematicsSolver requires independently movable measurement frames.
% A slider fixed at coordinate zero supplies a coincident world reference.
joint=[model '/Measurement origin'];
j=add_block(sprintf('sm_lib/Joints/Prismatic\nJoint'),joint);
p=get_param(j,'PortHandles'); w=get_param(world,'PortHandles');
add_line(model,w.RConn(1),p.LConn(1));
block=add_block('sm_lib/Body Elements/Brick Solid',[model '/Measurement mass']);
h=get_param(block,'PortHandles'); add_line(model,p.RConn(1),h.RConn(1));
origin=[joint '/F'];
end

function t=sourceType(b)
t=''; p=get_param(b,'ObjectParameters');
if isfield(p,'ReferenceBlock')
    ref=get_param(b,'ReferenceBlock');
    if startsWith(ref,'sm_lib/')
        fields=strsplit(ref,'/'); t=strtrim(regexprep(fields{end},'\s+',' '));
    end
end
end

function s=measure(s,b,sid,key,option,unit)
set_param(b,option,'on'); ph=get_param(b,'PortHandles');
% This model has one joint position output. Pulley outputs are enabled in
% their block order (wrap, A, B), each appearing after the preceding one.
port=ph.RConn(end);
assert(endsWith(get_param(port,'ConnectionType'),'.output'));
sys=get_param(b,'Parent'); name=['s' sid '_' key];
c=add_block(sprintf('nesl_utility/PS-Simulink\nConverter'),[sys '/read_' name],'Unit',unit);
cp=get_param(c,'PortHandles'); add_line(sys,port,cp.LConn(1));
h=add_block('simulink/Sinks/To Workspace',[sys '/log_' name],'VariableName',name,'SaveFormat','Timeseries');
hp=get_param(h,'PortHandles'); add_line(sys,cp.Outport(1),hp.Inport(1)); s.(name)=0;
end

function result=framePoses(ks,bases,followers,keys,targets)
result=struct;
for first=1:12:numel(keys)
    idx=first:min(first+11,numel(keys));
    batch=frameBatch(ks,bases(idx),followers(idx),keys(idx),targets);
    for j=idx, result.(keys{j})=batch.(keys{j}); end
    clearOutputVariables(ks); clearFrameVariables(ks);
end
end

function result=frameBatch(ks,bases,followers,keys,targets)
for i=1:numel(keys)
    addFrameVariables(ks,['t' keys{i}],'translation',bases{i},followers{i},'LengthUnit','m');
    addFrameVariables(ks,['r' keys{i}],'rotation',bases{i},followers{i},'AngleUnit','rad');
end
vars=frameVariables(ks); addOutputVariables(ks,vars.ID);
[v,status,ok]=solve(ks,targets);
assert(status>=0 && all(ok),'Kinematic solve failed or joint targets missed.');
ids=string(outputVariables(ks).ID); result=struct;
for i=1:numel(keys)
    key=keys{i}; xyz=v(startsWith(ids,['t' key '.'])); a=v(startsWith(ids,['r' key '.']));
    assert(numel(a)==3 && numel(xyz)==3);
    cx=cos(a(1)); sx=sin(a(1)); cy=cos(a(2)); sy=sin(a(2)); cz=cos(a(3)); sz=sin(a(3));
    R=[1 0 0;0 cx -sx;0 sx cx]*[cy 0 sy;0 1 0;-sy 0 cy]*[cz -sz 0;sz cz 0;0 0 1];
    result.(key)=[R xyz(:);0 0 0 1];
end
end

function hash=fileHash(path)
fid=fopen(path,'rb'); assert(fid>=0,'Cannot read %s',path); closer=onCleanup(@()fclose(fid));
md=java.security.MessageDigest.getInstance('SHA-256');
md.update(fread(fid,Inf,'*uint8')); hash=lower(reshape(dec2hex(typecast(md.digest(),'uint8'),2).',1,[]));
end
