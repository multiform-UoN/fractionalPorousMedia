function [N] = f_N(csi)

% Shape functions
N1=1/2*(1-csi);
N2=1/2*(1+csi);
N=[N1',N2'];

end

