function [f]=assemble_load_vector(el,dof,n_el,dof_el,A)

% Assemblage of load vector
f=zeros(dof,1);
for n=1:n_el
    for i=1:dof_el
        f(A(n,i),1)=f(A(n,i),1)+el(n).f(i,1);
    end
end

end

