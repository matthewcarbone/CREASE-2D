function fastcalcIq_sphere(path,infilename,outfilename)
alltime=tic;
opts = delimitedTextImportOptions("NumVariables", 12);
opts.DataLines = [10, Inf];
opts.Delimiter = " ";
opts.VariableNames = ["id", "type", "x", "y", "z", "a", "b", "c", "qw", "qx", "qy", "qz"];
opts.VariableTypes = ["double", "double", "double", "double", "double", "double","double", "double", "double", "double", "double", "double"];
opts.ExtraColumnsRule = "ignore";
opts.EmptyLineRule = "read";
opts.ConsecutiveDelimitersRule = "join";
opts.LeadingDelimitersRule = "ignore";
data = readtable([path infilename], opts);
clear opts
%Read box dimensions
opts = delimitedTextImportOptions("NumVariables", 2);
opts.DataLines = [6, 8];
opts.Delimiter = " ";
opts.VariableNames = ["lo", "hi"];
opts.VariableTypes = ["double", "double"];
opts.ConsecutiveDelimitersRule = "join";
opts.LeadingDelimitersRule = "ignore";
boxdim = readtable([path infilename], opts);
clear opts
%Generate random displacements within the box
numruns=10;
boxlen=mean([boxdim.hi(1)-boxdim.lo(1) boxdim.hi(2)-boxdim.lo(2) boxdim.hi(3)-boxdim.lo(3)]);
boxrad=boxlen/2;
originpositions=rand(numruns,3)*boxrad;
%Setup q,theta grid
nq=501;
ntheta=91;
qmin_exponent=-2;
qmax_exponent=3;
qgrid = logspace(qmin_exponent,qmax_exponent,nq)'*ones(1,ntheta);
%qgrid = logspace(-1,-0.9,nq)'*ones(1,ntheta);
thetagrid = ones(nq,1)*linspace(0,pi/2,ntheta);
qmag=qgrid;
qmag(:,:,2)=qmag(:,:,1);
dir1grid=cos(thetagrid);
dir2grid=sin(thetagrid);
dir1grid(:,:,2)=-dir1grid(:,:,1);
dir2grid(:,:,2)=dir2grid(:,:,1);
dir1value=reshape(dir1grid,[],1);
dir2value=reshape(dir2grid,[],1);
numpoints=nq*ntheta*2;
qmagvalue=reshape(qmag,[],1);
boxformfactorxy=3*(sin(qmagvalue.*boxrad)-(qmagvalue.*boxrad).*cos(qmagvalue.*boxrad))./(qmagvalue.*boxrad).^3;
boxformfactoryz=3*(sin(qmagvalue.*boxrad)-(qmagvalue.*boxrad).*cos(qmagvalue.*boxrad))./(qmagvalue.*boxrad).^3;
boxformfactorxz=3*(sin(qmagvalue.*boxrad)-(qmagvalue.*boxrad).*cos(qmagvalue.*boxrad))./(qmagvalue.*boxrad).^3;

finalresultqxqy=0;
finalresultqyqz=0;
finalresultqxqz=0;

%run calculations for random displacements of the origin
for i=1:numruns
    XYZ=[data.x-originpositions(i,1) data.y-originpositions(i,2) data.z-originpositions(i,3)];
    XYZ=mod(XYZ+boxrad,boxlen)-boxrad;
    datamap=(sum(XYZ.^2,2)<boxrad^2);
    XYZ=XYZ(datamap,:);
    TYPE=data.type(datamap);
    ID=data.id(datamap);
    quatW=data.qw(datamap);
    quatX=data.qx(datamap);
    quatY=data.qy(datamap);
    quatZ=data.qz(datamap);
    ax_vecs=[...
        2*(quatX.*quatZ+quatW.*quatY),...
        2*(quatY.*quatZ-quatW.*quatX),...
        quatW.^2-quatX.^2-quatY.^2+quatZ.^2];
    ax_vecs_mag=sqrt(sum(ax_vecs.^2,2));
    ax_vecs=ax_vecs./(ax_vecs_mag*ones(1,3));
    ax_lens=[data.a(datamap) data.b(datamap) data.c(datamap)];
    Numbeads=length(XYZ(:,1));
    boxvol=4/3*pi*boxrad^3;
    ellipsoidalvol=4/3*pi*prod(ax_lens,2);
    Ampxy = 0;
    Ampyz = 0;
    Ampxz = 0;
    chunksize = 256;
    nchunks=ceil(Numbeads/chunksize);
    nchunks=pow2(ceil(log2(nchunks)));
    chunksize=ceil(Numbeads/nchunks);
    disp(['For ' outfilename ' working on run ' num2str(i) ' of ' num2str(numruns) ' runs.']);
    disp(['Total Chunks = ' num2str(nchunks)]);
    numpaddedvals=nchunks*chunksize-Numbeads;
    lastvalidchunk=nchunks-floor(numpaddedvals/chunksize);
    XYZ=padarray(XYZ,numpaddedvals,'post');
    ellipsoidalvol=padarray(ellipsoidalvol,numpaddedvals,'post');
    ax_vecs=padarray(ax_vecs,numpaddedvals,'post');
    ax_lens=padarray(ax_lens,numpaddedvals,'post');
    chunkXYZ=pagetranspose(reshape(XYZ',3,chunksize,nchunks));
    chunkellipsoidalvol=reshape(ellipsoidalvol,chunksize,1,nchunks);
    chunkax_vecs=pagetranspose(reshape(ax_vecs',3,chunksize,nchunks));
    chunkax_lens=pagetranspose(reshape(ax_lens',3,chunksize,nchunks));
    parfor n=1:nchunks
        innerlooptime=tic;
        subXYZ=chunkXYZ(:,:,n);
        subellipsoidalvol=ones(numpoints,1)*chunkellipsoidalvol(:,:,n)';
        axvec=chunkax_vecs(:,:,n);
        axlen=chunkax_lens(:,:,n);
        if n<lastvalidchunk
            currentchunksize=chunksize;
        elseif n == lastvalidchunk
            currentchunksize=Numbeads-(lastvalidchunk-1)*chunksize;
            subXYZ=subXYZ(1:currentchunksize,:);
            subellipsoidalvol=subellipsoidalvol(:,1:currentchunksize);
            axvec=axvec(1:currentchunksize,:);
            axlen=axlen(1:currentchunksize,:);
        else
            disp(['Skipping Chunk#=' num2str(n) '.']);
            continue;
        end
        ellipsoidalvol=ones(numpoints,1)*4/3*pi*prod(axlen,2)';
        cos_alphaxy=(dir1value*axvec(:,1)'+dir2value*axvec(:,2)');
        cos_alphayz=(dir1value*axvec(:,2)'+dir2value*axvec(:,3)');
        cos_alphaxz=(dir1value*axvec(:,1)'+dir2value*axvec(:,3)');
        sin_alphaxy=sin(acos(cos_alphaxy));
        sin_alphayz=sin(acos(cos_alphayz));
        sin_alphaxz=sin(acos(cos_alphaxz));
        qradxy = (qmagvalue*ones(1,currentchunksize)).*sqrt((sin_alphaxy.*(ones(numpoints,1)*axlen(:,1)')).^2+(cos_alphaxy.*(ones(numpoints,1)*axlen(:,3)')).^2);
        qradyz = (qmagvalue*ones(1,currentchunksize)).*sqrt((sin_alphayz.*(ones(numpoints,1)*axlen(:,1)')).^2+(cos_alphayz.*(ones(numpoints,1)*axlen(:,3)')).^2);
        qradxz = (qmagvalue*ones(1,currentchunksize)).*sqrt((sin_alphaxz.*(ones(numpoints,1)*axlen(:,1)')).^2+(cos_alphaxz.*(ones(numpoints,1)*axlen(:,3)')).^2);
        formfacxy = 3*(sin(qradxy)-qradxy.*cos(qradxy))./qradxy.^3;
        formfacyz = 3*(sin(qradyz)-qradyz.*cos(qradyz))./qradyz.^3;
        formfacxz = 3*(sin(qradxz)-qradxz.*cos(qradxz))./qradxz.^3;
        qposxy=(qmagvalue*ones(1,currentchunksize)).*(dir1value*subXYZ(:,1)'+dir2value*subXYZ(:,2)');
        qposyz=(qmagvalue*ones(1,currentchunksize)).*(dir1value*subXYZ(:,2)'+dir2value*subXYZ(:,3)');
        qposxz=(qmagvalue*ones(1,currentchunksize)).*(dir1value*subXYZ(:,1)'+dir2value*subXYZ(:,3)');
        resultxy=sum(subellipsoidalvol.*(exp(complex(0,-1)*qposxy).*formfacxy-boxformfactorxy*ones(1,currentchunksize)),2);
        resultyz=sum(subellipsoidalvol.*(exp(complex(0,-1)*qposyz).*formfacyz-boxformfactoryz*ones(1,currentchunksize)),2);
        resultxz=sum(subellipsoidalvol.*(exp(complex(0,-1)*qposxz).*formfacxz-boxformfactorxz*ones(1,currentchunksize)),2);
        Ampxy=Ampxy+resultxy;
        Ampyz=Ampyz+resultyz;
        Ampxz=Ampxz+resultxz;
        disp(['Chunk#=' num2str(n) '. The time elapsed is ' num2str(toc(innerlooptime)) ' seconds.']);
    end
    resultqxqy = log10(reshape(Ampxy.*conj(Ampxy),nq,ntheta,2)/boxvol);
    resultqyqz = log10(reshape(Ampyz.*conj(Ampyz),nq,ntheta,2)/boxvol);
    resultqxqz = log10(reshape(Ampxz.*conj(Ampxz),nq,ntheta,2)/boxvol);
    finalresultqxqy = finalresultqxqy + resultqxqy;
    finalresultqyqz = finalresultqyqz + resultqyqz;
    finalresultqxqz = finalresultqxqz + resultqxqz;
end
finalresultqxqy=finalresultqxqy/numruns;
finalresultqyqz=finalresultqyqz/numruns;
finalresultqxqz=finalresultqxqz/numruns;
disp(['The time elapsed is ' num2str(toc(alltime)) ' seconds.']);
%save(outfilename,'finalresultqxqy','finalresultqxqz','finalresultqyqz','qgrid','thetagrid');
dataxy=[finalresultqxqy(:,1:end-1,1) fliplr(finalresultqxqy(:,:,2))];
datayz=[finalresultqyqz(:,1:end-1,1) fliplr(finalresultqyqz(:,:,2))];
dataxz=[finalresultqxqz(:,1:end-1,1) fliplr(finalresultqxqz(:,:,2))];
%dataq=[qgrid(:,1:end-1) qgrid];
%datatheta=[thetagrid(:,1:end-1) pi-fliplr(thetagrid)];
writematrix(dataxy,[outfilename '_dataxy.txt']);
writematrix(datayz,[outfilename '_datayz.txt']);
writematrix(dataxz,[outfilename '_dataxz.txt']);
%writematrix(dataq(:,1),'datagrid_q.txt');
%writematrix(datatheta(1,:),'datagrid_theta.txt');
