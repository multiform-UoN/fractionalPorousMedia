function [gauss]=test_functions_Gauss_points_2D(gauss)

% Computation of test functions (and derivatives) at Gauss points
n_gauss=length(gauss);
for n=1:n_gauss
    gauss(n).W=f_W_2D(gauss(n).csi,gauss(n).eta);
    gauss(n).dW=f_dW_2D(gauss(n).csi,gauss(n).eta);
end

end