function [A]=afference_matrix(n_el,dof_el)

% Afference matrix
A=zeros(n_el,dof_el);
for i=1:n_el
    for j=1:dof_el
        A(i,j)=(i-1)*(dof_el-1)+j; 
    end
end

end

