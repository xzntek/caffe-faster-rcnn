function rec = VOCreadxml(path)

if length(path)>5&&strcmp(path(1:5),'http:')
    xml=urlread(path)';
else
    f=fopen(path,'r');
    xml=fread(f,'*char')';
    fclose(f);
end

addition = '<?xml version="1.0" ?>';
if ( strcmp(addition, xml(1:min(end,length(addition)))) == 1 )
    L = length(addition);
    xml = xml(L+1:end);
end

rec=VOCxml2struct(xml);
