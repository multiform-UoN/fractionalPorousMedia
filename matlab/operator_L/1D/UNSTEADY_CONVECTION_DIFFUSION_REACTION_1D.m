%% 1D Unsteady convection-diffusion-reaction problem
% Based on "Finite Element Methods for flow problems" of
% Jean Donea and Antonio Huerta
% Andrea La Spina

%% Equation

% u_t+a*u_x-v*u_xx+sigma*u=s(x)           in [x_i,x_f]x]t_i,t_f[
% u(x,0)=u0(x)                            on [x_i,x_f]
% u(x1,t)=u_1(t);                         on x1 for t in [t_i,t_f]
% u(x2,t)=u_2(t);                         on x2 for t in [t_i,t_f]

%% Initialization

initialization

%% Model parameters

x_i=0;                                  % Initial point
x_f=1;                                  % Final point
a=1;                                    % Convection velocity
sigma=0;                                % Reaction coefficient
s_fun=@(x) 0;                           % Source term
u_0_fun=@(x) x*0;                       % Initial condition
% dof_constrained_string='[1,dof]';       % Degree of freedom constrained
% bound_cond_fun={@(t) 0,@(t) 0};         % Boundary conditions
dof_constrained_string='[1]';           % Degree of freedom constrained
bound_cond_fun={@(t) 1};            % Boundary conditions
Pe=1;                                % Péclet number
Courant=.2;                             % Courant number
t_i=0;                                  % Initial time
t_f=1;                                  % Final time
n_el=10;                               % Number of finite elements
n_gauss=2;                              % Number of Gauss points
polynomial_degree=1;                    % Shape functions polynomial degree
FE_type='Galerkin';                     % Type of FE (Galerkin or Upwind)
theta=1;                                % Theta for time integration scheme
                                        % ( 0  = Forward Euler)
                                        % (1/2 = Crank-Nicolson)
                                        % (2/3 = Galerkin)
                                        % ( 1  = Backward Euler)
dtvis = 1;

%% Derived parameters

L=x_f-x_i;                              % Domain length
n_np=polynomial_degree*n_el+1;          % Number of nodal points
n_eq=polynomial_degree*n_el-1;          % Number of equations
dof_el=polynomial_degree+1;             % Number of DOFs per element
dof=n_np;                               % Total number of DOFs
L_el=L/n_el;                            % Length of a finite element
h=L_el/polynomial_degree;               % Spatial step
x=x_i:h:x_f;                            % Space vector
dx_p=L/150;                             % Spatial step-analytical solution
x_p=x_i:dx_p:x_f;                       % Space vector-analytical solution
dx_e=L_el/10;                           % Spatial step-numerical interp.
x_e=0:dx_e:L_el;                        % Space vector-numerical interp.
n_e=length(x_e);                        % Number of points in space vector
s=feval(s_fun,x).*ones(1,length(x));    % Numerical value of source term
v=a*h/(2*Pe);                           % Diffusivity coefficient
dt=h/a*Courant;                         % Time step
T=t_i:dt:t_f;                           % Time vector

%% Evaluation of beta

if   strcmp(FE_type,'Galerkin')==1
    beta=0;
elseif strcmp(FE_type,'Upwind')==1
    beta=coth(Pe)-1/Pe;
end

%% Gauss parameters

[csi,w]=Gauss_parameters(n_gauss);

% Trasformation of coordinated for the Gauss integration points
for n=1:n_gauss
    x_gauss(n)=L_el/2*(1+csi(n));
end

% Jacobian of the transformation
J=h/2;

% Computation of shape and test functions (and derivatives) at Gauss points
[N,dN]=shape_functions_Gauss_points(csi);
[W,dW]=test_functions_Gauss_points(csi,beta);

%% Evaluate matrices and vectors

% Afference matrix
[A]=afference_matrix(n_el,dof_el);

% Element mass matrix
for n=1:n_el
    el(n).M=element_mass_matrix(dof_el,n_gauss,N,W,w,J);
end

% Element convection matrix
for n=1:n_el
    el(n).C=element_convection_matrix(a,dof_el,n_gauss,dN,W,w,J);
end

% Element diffusion matrix
for n=1:n_el
    el(n).K=element_diffusion_matrix(v,dof_el,n_gauss,dN,dW,w,J);
end

% Element load vector
for n=1:n_el
    el(n).s=s((n-1)*(dof_el-1)+1:n*(dof_el-1)+1);
    el(n).f=element_load_vector(el(n).s,dof_el,n_gauss,N,W,w,J);
end

% Element abscissae
for n=1:n_el
    el(n).x=x_i+(n-1)*L_el+x_e;
end

%% Assemblate matrices and vectors

% Assemblage of mass matrix
[M]=assemble_mass_matrix(el,dof,n_el,dof_el,A);

% Assemblage of convection matrix
[C]=assemble_convection_matrix(el,dof,n_el,dof_el,A);

% Assemblage of diffusion matrix
[K]=assemble_diffusion_matrix(el,dof,n_el,dof_el,A);

% Convection+Diffusion+Reaction matrix
D=C+K+sigma.*M;

% Assemblage of load vector
[f]=assemble_load_vector(el,dof,n_el,dof_el,A);

%% Boundary conditions

% Definition of the constrained DOFs
dof_constrained=eval(dof_constrained_string);
dof_free=dof-length(dof_constrained);
dof_constrained=sort(dof_constrained);
n_dof_constrained=length(dof_constrained);

% Evaluation of boundary conditions over time
for k=1:length(T)
    t=T(k);
    for n=1:n_dof_constrained
       constrain(n)=feval(bound_cond_fun{n},t);
    end
    u_p(:,k)=constrain';
end

% Mass matrix
[M_ff,M_fp,M_pf,M_pp]=constrain_matrix(M,dof_constrained);

% Convection matrix
[C_ff,C_fp,C_pf,C_pp]=constrain_matrix(C,dof_constrained);

% Diffusion matrix
[K_ff,K_fp,K_pf,K_pp]=constrain_matrix(K,dof_constrained);

% Convection+Diffusion matrix
[D_ff,D_fp,D_pf,D_pp]=constrain_matrix(D,dof_constrained);

% Load vector
[f_f,f_p]=constrain_vector(f,dof_constrained);

%% Initial conditions

u_0=feval(u_0_fun,x)';
u_0_f=constrain_vector(u_0,dof_constrained);

%% Unsteady convectio-diffusion-reaction solution

% M_ff = eye(n_el)/n_el;
% Time integration
u_f(1,:)=u_0_f;
for k=1:length(T)-1
    u_f(k+1,:)=(M_ff+dt*theta*D_ff)\...
        (...
        (M_ff-dt*(1-theta)*D_ff)*u_f(k,:)'...
        +dt*theta*(f_f-D_fp*u_p(:,k+1))...
        +dt*(1-theta)*(f_f-D_fp*u_p(:,k))...
        );
end

% Data for all dof
for k=1:length(T)
    [time(k).u]=data_all_dof(u_f(k,:)',u_p(:,k),dof_constrained);
end

% Interpolation of the solution
for n=1:n_el
    for k=1:length(T)
        el(n).time(k).u=interpolation(n,time(k).u,A,n_e);
    end
end

%% Animation

% Screen dimensions
scrsz=get(0,'ScreenSize');  % [pixel]
bar=64;                     % [pixel]

j=1;
figure('Color',[1 1 1],'Position',[0 0 scrsz(3) (scrsz(4)-bar)])
while j<=length(T)
    
    % Plot
    hold on
    plot(x,time(j).u,'LineWidth',2)
    title(['Solution - t = ', num2str(round(T(j)*100)/100),' sec'],...
           'FontSize',14)
    xlabel('x','FontSize',14)
    ylabel('u','FontSize',14)
    grid on
    grid minor
    xlim([x_i,x_f])
    ylim([0,1.1])
    
    % Time calibration
    j=j+dtvis;
    
    pause(0.0001);
end

%% Display in command window

disp('MODEL PARAMETERS')
disp('-------------------------------------------------------------------')
fprintf('Length\t\t\t\t=\t%.1f\n',L)
fprintf('Convection velocity\t\t=\t%.2f\n',a)
fprintf('Diffusion coefficient\t\t=\t%.2e\n',v)
fprintf('Reaction coefficient\t\t=\t%.2f\n',sigma)
fprintf('Source term\t\t\t=\t%s\n',char(s_fun))
fprintf('Initial condition\t\t=\t%s\n',char(u_0_fun))
fprintf('DOFs constrained\t\t=\t%s\n',dof_constrained_string)
fprintf('Boundary conditions\t\t=')
for n=1:n_dof_constrained
    fprintf('\t%s',char(bound_cond_fun{n}))
end
fprintf('\n')
fprintf('Polynomial degree\t\t=\t%d\n',polynomial_degree)
fprintf('Initial time\t\t\t=\t%.2f\n',t_i)
fprintf('Final time\t\t\t=\t%.2f\n',t_f)
fprintf('Time step\t\t\t=\t%.2e\n',dt)
fprintf('Péclet number\t\t\t=\t%.2f\n',Pe)
fprintf('Courant number\t\t\t=\t%.2f\n',Courant)
fprintf('Type of finite elements\t\t=\t%s\n',FE_type)
fprintf('Theta\t\t\t\t=\t%.2f\n',theta)
disp('-------------------------------------------------------------------')

disp(' ')

disp('FEM PARAMETERS')
disp('-------------------------------------------------------------------')
fprintf('Number of finite elements\t=\t%d\n',n_el)
fprintf('Length of a finite element\t=\t%.2f\n',L_el)
fprintf('Gauss integration points\t=\t%d\n',n_gauss)
fprintf('Number of nodes\t\t\t=\t%d\n',n_np)
fprintf('Number of DOF per element\t=\t%d\n',dof_el)
fprintf('Total number of DOF\t\t=\t%d\n',dof)
disp('-------------------------------------------------------------------')

disp(' ')

disp('ELEMENT MASS MATRIX (/det(J))')
disp('-------------------------------------------------------------------')
disp(el(1).M./J)
disp('-------------------------------------------------------------------')

disp(' ')

disp('ELEMENT CONVECTION MATRIX (/a)')
disp('-------------------------------------------------------------------')
disp(el(1).C./a)
disp('-------------------------------------------------------------------')

disp(' ')

disp('ELEMENT DIFFUSION MATRIX (/v*det(J))')
disp('-------------------------------------------------------------------')
disp(el(1).K./v.*J)
disp('-------------------------------------------------------------------')

disp(' ')

disp('ELEMENT LOAD VECTOR (/det(J))')
disp('-------------------------------------------------------------------')
disp(el(1).f./J)
disp('-------------------------------------------------------------------')

disp(' ')

% if dof<=10
% 
% disp('AFFERENCE MATRIX')
% disp('-------------------------------------------------------------------')
% disp(A)
% disp('-------------------------------------------------------------------')
% 
% disp(' ')
% 
% disp('GLOBAL MASS MATRIX (/det(J))')
% disp('-------------------------------------------------------------------')
% disp(M./J)
% disp('-------------------------------------------------------------------')
% 
% disp(' ')
% 
% disp('GLOBAL CONVECTION MATRIX (/a)')
% disp('-------------------------------------------------------------------')
% disp(C./a)
% disp('-------------------------------------------------------------------')
% 
% disp(' ')
% 
% disp('GLOBAL DIFFUSION MATRIX (/v*det(J))')
% disp('-------------------------------------------------------------------')
% disp(K./v.*J)
% disp('-------------------------------------------------------------------')
% 
% disp(' ')
% 
% disp('GLOBAL LOAD VECTOR (/det(J))')
% disp('-------------------------------------------------------------------')
% disp(f./J)
% disp('-------------------------------------------------------------------')
% 
% disp(' ')
% 
% end
% 
% disp('EVALUATION OF THE SIMULATION''S ACCURACY')
% disp('-------------------------------------------------------------------')
% fprintf('Real time =\t\t%.2f\t[sec]\n',time_real)
% fprintf('Simulation time =\t%.2f\t[sec]\n',time_simulation)
% fprintf('Relative error =\t%.2f\t[%%]\n',relative_error*100)
% disp('-------------------------------------------------------------------')