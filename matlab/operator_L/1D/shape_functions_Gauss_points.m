function [N,dN]=shape_functions_Gauss_points(csi)

% Computation of shape functions (and derivatives) at Gauss points
n_gauss=length(csi);
for n=1:n_gauss
    N(:,n)=f_N(csi(n));
    dN(:,n)=f_dN(csi(n));
end

end

