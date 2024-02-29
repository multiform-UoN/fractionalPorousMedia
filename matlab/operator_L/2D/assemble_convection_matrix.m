function [C]=assemble_convection_matrix(el,dof,n_el,dof_el,A)

% Assemblage of convection matrix
C=zeros(dof,dof);
for n=1:n_el
    for i=1:dof_el
        for j=1:dof_el
            C(A(n,i),A(n,j))=C(A(n,i),A(n,j))+el(n).C(i,j);
        end
    end
end

end

