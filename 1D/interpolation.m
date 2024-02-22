function [u_e]=interpolation(n,u,A,n_e)

% Interpolation of scalar variable
csi=linspace(-1,+1,n_e);
for i=1:n_e
    Ni=f_N(csi(i));
    un=u(A(n,:));
    u_e(i)=Ni*un;
end

end

