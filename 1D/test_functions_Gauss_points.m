function [W,dW]=test_functions_Gauss_points(csi,beta)

% Computation of test functions (and derivatives) at Gauss points
n_gauss=length(csi);
for n=1:n_gauss
    W(:,n)=f_W(csi(n),beta);
    dW(:,n)=f_dW(csi(n),beta);
end

end

