% this code is inspired by VOCevaldet in the PASVAL VOC devkit
% Note: this function has been significantly optimized since ILSVRC2013
function Make_LOC_FRCNN_Data()
% Format : 
% # image_index
% img_path (relative path)
% num_roi
% label x1 y1 x2 y2 difficult
clear;clc;

CLASS = {'Animal', 'Music', 'Bike', 'Baby', 'Boy', 'Fire', 'Skier'};
Classes = cell(7,1);
Classes{1} = {'n01443537', 'n01503061', 'n01639765', 'n01662784', 'n01674464', 'n01726692', 'n01770393', ...
'n01784675', 'n01882714', 'n01910747', 'n01944390', 'n01990800', 'n02062744', 'n02076196', ...
'n02084071', 'n02118333', 'n02129165', 'n02129604', 'n02131653', 'n02165456', 'n02206856', ...
'n02219486', 'n02268443', 'n02274259', 'n02317335', 'n02324045', 'n02342885', 'n02346627', ...
'n02355227', 'n02374451', 'n02391049', 'n02395003', 'n02398521', 'n02402425', 'n02411705', ...
'n02419796', 'n02437136', 'n02444819', 'n02445715', 'n02454379', 'n02484322', 'n02503517', ...
'n02509815', 'n02510455', 'Animal'};
Classes{2} = { ...
'n02672831', 'n02787622', 'n02803934', 'n02804123', 'n03249569', 'n03372029', 'n03467517', 'n03800933', ...
'n03838899', 'n03928116', 'n04141076', 'n04536866', 'Music'}; ...
Classes{3} = {'Bike', 'n02834778', 'n02835271', 'n03792782', 'n04126066'};
Classes{4} = {'Baby', 'n10353016'};
Classes{5} = {'Boy', 'n09871229', 'n09871867', 'n10078719'};
Classes{6} = {'n03343560', 'Fire', 'FIre', 'n03346135', 'n10091450', 'n10091564', 'n10091861', 'n14891255'};
Classes{7} = {'Skier', 'n04228054'};

%XML_Dir = './LOC/train_bbox';
%IMG_Dir = './LOC/train_img';
XML_Dir = './LOC/frame_bbox';
IMG_Dir = './LOC/frame_img';
assert( ~isempty(XML_Dir) && exist(XML_Dir,'dir') );
ImageSet = './LOC/LOC_Split/trecvid_5_manual_frame_val.txt';
Save_Name = './trecvid_5_manual.val';
%ImageSet = './LOC/LOC_Split/trecvid_val_Animal_Music.txt';
%Save_Name = './Animal_Music.val';

[ pic ] = textread(ImageSet,'%s');
num_imgs = length(pic);
t = tic;
Fid = fopen(Save_Name,'w');
fprintf('total imgs : %d\n', num_imgs);
for i=1:num_imgs
    rec = VOCreadxml(fullfile(XML_Dir,[pic{i},'.xml']));
    fprintf(Fid,'# %d\n%s\n',i-1, rec.annotation.filename);
    if ~isfield(rec.annotation,'object')
        fprintf(Fid,'0\n');
    else
        fprintf(Fid,'%-3d\n',length(rec.annotation.object));
        for j=1:length(rec.annotation.object)
            obj = rec.annotation.object(j);
            label = Get_Label(CLASS, Classes, obj.name);
            b = obj.bndbox;
            box = str2double({b.xmin b.ymin b.xmax b.ymax});
            fprintf(Fid, '%-3d %-5.0f %-5.0f %-5.0f %-5.0f %s\n', label, box, getfield(obj,'difficult') );
        end
    end
    if (rem(i,1000) == 0), fprintf('Current %4d / %4d, cost : %5.1f s\n', i, num_imgs, toc(t)); end
end
fprintf('Process %6d xmls in %.2f min\n', num_imgs , toc(t)/60);

end

function label = Get_Label(CLASS, Classes, name)
    assert(length(CLASS) == length(Classes));
    for i = 1:length(CLASS)
        class_names = Classes{i};
        for j = 1:length(class_names)
            if(strcmp(class_names{j},name)==1)
                label = i;
                return;
            end
        end
    end
    assert( false , ['Does not find matched class name : ', name]);
end
