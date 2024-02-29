function [M]=element_mass_matrix(dof_el,n_gauss,N,W,w,J)

% Element mass matrix
M=zeros(dof_el,dof_el);
for i=1:dof_el
    for j=1:dof_el
        for nn=1:n_gauss
            M(i,j)=M(i,j)+(W(i,nn)*N(j,nn))*w(nn);
        end
        M(i,j)=(M(i,j))*J;
    end
end

end

