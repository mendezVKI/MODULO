
load('Data.mat');
mkdir('Ex_2_Synthetic');
destination = 'Ex_2_Synthetic';

fileID = fopen('MESH.txt','wt');
fprintf(fileID,'#x #y\n');

for i=1:256
    for j=1:256
        fprintf(fileID,'%2.6f \t %2.6f\n',Xg(1,j),Yg(i,j));
    end
end
fclose(fileID);

movefile('MESH.txt',destination)

for i=1:512
   name = ['Res_',num2str(i,'%05.f'),'.txt'];
   fileID = fopen(name,'wt');
   fprintf(fileID,'#u\n');
   fprintf(fileID,'%2.6g\n',D(:,i));
   fclose(fileID);
   movefile(name,destination)
end

