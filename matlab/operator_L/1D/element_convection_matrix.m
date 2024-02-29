function [C]=element_convection_matrix(a,dof_el,n_gauss,dN,W,w,J)

% Element convection matrix
C=zeros(dof_el,dof_el);
for i=1:dof_el
    for j=1:dof_el
        for nn=1:n_gauss
            C(i,j)=C(i,j)+(W(i,nn)*dN(j,nn))*w(nn);
        end
        C(i,j)=a*C(i,j);
    end
end

end

