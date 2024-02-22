function [N] = f_N_2D(csi,eta)

% Shape functions
N(1).N=1/4*(1-csi)*(1-eta);
N(2).N=1/4*(1+csi)*(1-eta);
N(3).N=1/4*(1+csi)*(1+eta);
N(4).N=1/4*(1-csi)*(1+eta);

end