%% 1D Unsteady convection-diffusion-reaction problem
% Based on "Finite Element Methods for flow problems" of
% Jean Donea and Antonio Huerta

% Andrea La Spina
% https://it.linkedin.com/in/andrealaspina
% https://independent.academia.edu/AndreaLaSpina
% https://www.researchgate.net/profile/Andrea_La_Spina

%% Equation

% u_t+div(a*u)=0                        in W x ]t_i,t_f[
% u(x,0)=u0(x)                          on W at t_i
% u=u_D                                 on W_D x ]t_i,t_f[

%% Initialization

initialization

%% Input

% Model parameters --------------------------------------------------------
a_fun=@(x,y,h) [-y,x];                  % Convection velocity
v=1e-3;                                 % Diffusion coefficient
sigma=0.1;                              % Reaction coefficient
% -------------------------------------------------------------------------

% Space -------------------------------------------------------------------
x_i=-1/2;                               % Initial point (x)
x_f=+1/2;                               % Final point (x)
y_i=-1/2;                               % Initial point (y)
y_f=+1/2;                               % Final point (y)
% -------------------------------------------------------------------------

% Time --------------------------------------------------------------------
t_i=0;                                  % Initial time
t_f=2*pi;                               % Final time
dt=2*pi/100;                            % Time step
animation_time=10;                      % Animation time
% -------------------------------------------------------------------------

% FEM ---------------------------------------------------------------------
n_el_x=30;                              % Number of finite elements (x)
n_el_y=30;                              % Number of finite elements (y)
n_gauss_side=2;                         % Number of Gauss points per side
theta=1/2;                              % Theta for time integration scheme
                                        % ( 0  = Forward Euler)
                                        % (1/2 = Crank-Nicolson)
                                        % (2/3 = Galerkin)
                                        % ( 1  = Backward Euler)
% -------------------------------------------------------------------------

% Boundary and initial conditions -----------------------------------------
radius=0.2;                             % Radius of the hill
x_0=1/6;                                % Centre of the hill (x)
y_0=1/6;                                % Centre of the hill (y)
u_0_fun=@(x,y)...                       % Initial condition
        1/4*(1+cos(pi*(x-x_0)/radius))*...
        (1+cos(pi*(y-y_0)/radius))*...
        heaviside(x-(x_0-sqrt(radius^2-(y-y_0)^2)))*...
        heaviside(y-(y_0-sqrt(radius^2-(x-x_0)^2)))*...
        (1-heaviside(x-(x_0+sqrt(radius^2-(y-y_0)^2))))*...
        (1-heaviside(y-(y_0+sqrt(radius^2-(x-x_0)^2))));
dof_constrained_string=...              % Degree of freedom constrained
          ['[2:n_np_x-1,',...
           'n_np-n_np_x+2:n_np-1,'...
           '1:n_np_x:n_np-n_np_x+1,'...
           'n_np_x:n_np_x:n_np]'];       
bound_cond=0;                           % Boundary conditions
% -------------------------------------------------------------------------

%% Derived parameters

L_x=x_f-x_i;                            % Domain length (x)
L_y=y_f-y_i;                            % Domain length (y)
n_el=n_el_x*n_el_y;                     % Number of finite elements
n_gauss=n_gauss_side^2;                 % Number of Gauss points
n_np_x=n_el_x+1;                        % Number of nodal points (x)
n_np_y=n_el_y+1;                        % Number of nodal points (y)
n_np=n_np_x*n_np_y;                     % Number of nodal points
dof_el=4;                               % Number of DOFs per element
dof=n_np;                               % Total number of DOFs
L_el_x=L_x/n_el_x;                      % Length of a finite element (x)
L_el_y=L_y/n_el_y;                      % Length of a finite element (y)
h_x=L_el_x;                             % Spatial step (x)
h_y=L_el_y;                             % Spatial step (y)
h=sqrt(h_x*h_y);                        % Geometric average of spatial step
x=x_i:h_x:x_f;                          % Space vector (x)
y=y_i:h_y:y_f;                          % Space vector (y)
T=t_i:dt:t_f;                           % Time vector

%% Gauss parameters

gauss=[];
[gauss]=Gauss_parameters_2D(n_gauss_side,gauss);

% Trasformation of coordinated for the Gauss integration points
for n=1:n_gauss
    x_gauss(n)=L_el_x/2*(1+gauss(n).csi);
    y_gauss(n)=L_el_y/2*(1+gauss(n).eta);
end

% Jacobian of the transformation
J_mat=[h_y/2    0
         0    h_x/2];
J=det(J_mat);
J=sqrt(J);

% Computation of shape and test functions (and derivatives) at Gauss points
[gauss]=shape_functions_Gauss_points_2D(gauss);
[gauss]=test_functions_Gauss_points_2D(gauss);

%% Plot of geometry

if n_el<=100
    plot_geometry_2D(x,y,L_x,L_y,n_gauss,L_el_x,L_el_y,x_gauss,y_gauss);
end

%% Plot of shape and test functions

% Normalized domain
csi_plot=-1:0.2:1;
eta_plot=-1:0.2:1;

% Evaluate shape and test functions
N_plot=f_N_plot_2D(csi_plot,eta_plot);
plot_shape_functions_2D(csi_plot,eta_plot,N_plot,dof_el);

% Test functions
W_plot=f_W_plot_2D(csi_plot,eta_plot);
plot_test_functions_2D(csi_plot,eta_plot,W_plot,dof_el);

%% Evaluate matrices and vectors

% Afference matrix
[A]=afference_matrix_2D(n_np_x,n_np_y,dof_el);

% Element mass matrix
for n=1:n_el
    el(n).M=element_mass_matrix_2D(dof_el,gauss,J);
end

% Element convection matrix
for n=1:n_el
    [r,c]=row_column(n,n_el_x);
    x_c=x_i+L_el_x/2+L_el_x*(c-1);
    y_c=y_i+L_el_y*(n_el_y-1/2)-L_el_y*(r-1);
    a_c=feval(a_fun,x_c,y_c);
    el(n).C=element_convection_matrix_2D(a_c,dof_el,gauss,J);
end

% Element diffusion matrix
for n=1:n_el
    el(n).K=element_diffusion_matrix_2D(v,dof_el,gauss,J);
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

%% Boundary conditions

% Definition of the constrained DOFs
dof_constrained=eval(dof_constrained_string);
dof_free=dof-length(dof_constrained);
dof_constrained=sort(dof_constrained);
n_dof_constrained=length(dof_constrained);

% Evaluation of the derivative of the boundary conditions
bound_cond_der=0;

% Evaluation of boundary conditions over time
for k=1:length(T)
    t=T(k);
    for n=1:n_dof_constrained
       constrain(n)=bound_cond;
       constrain_der(n)=bound_cond_der;
    end
    u_p(:,k)=constrain';
    u_der_p(:,k)=constrain_der';
end

% Mass matrix
[M_ff,M_fp,M_pf,M_pp]=constrain_matrix(M,dof_constrained);

% Convection matrix
[C_ff,C_fp,C_pf,C_pp]=constrain_matrix(C,dof_constrained);

% Diffusion matrix
[K_ff,K_fp,K_pf,K_pp]=constrain_matrix(K,dof_constrained);

% Convection+Diffusion matrix
[D_ff,D_fp,D_pf,D_pp]=constrain_matrix(D,dof_constrained);

%% Initial conditions

for n=1:n_np
    [r,c]=row_column(n,n_np_x);
    xp=x_i+L_el_x*(c-1);
    yp=y_i+L_el_y*n_el_y-L_el_y*(r-1);
    u_0(n,1)=feval(u_0_fun,xp,yp);
end
u_0_f=constrain_vector(u_0,dof_constrained);

%% Unsteady convection

% Time integration
disp('TIME INTEGRATION')
disp('-------------------------------------------------------------------')
u_f(1,:)=u_0_f;
tic
for k=1:length(T)-1
    u_f(k+1,:)=(M_ff+dt*theta*D_ff)\...
        (...
        (M_ff-dt*(1-theta)*D_ff)*u_f(k,:)'...
        +dt*theta*(-M_fp*u_der_p(:,k+1)-D_fp*u_p(:,k+1))...
        +dt*(1-theta)*(-M_fp*u_der_p(:,k)-D_fp*u_p(:,k))...
        );
    fprintf('\nTime step %d/%d - Elapsed time %.2f sec',k,length(T)-1,toc)
end
disp(' ')
disp('-------------------------------------------------------------------')

% Data for all dof
for k=1:length(T)
    [time(k).u]=data_all_dof(u_f(k,:)',u_p(:,k),dof_constrained);
end

% Conversion of data from vector to matrix
for k=1:length(T)
    for j=1:n_np_y
        [time(k).u_matrix(:,j)]=time(k).u((n_np_y-j)*n_np_x+1:...
                                          (n_np_y-j)*n_np_x+n_np_x);
    end
end

%% Plots

% Grid matrices
[X,Y]=meshgrid(x,y);

% Evaluation of the convection field
for n=1:n_np
    [r,c]=row_column(n,n_np_x);
    xp=x_i+L_el_x*(c-1);
    yp=y_i+L_el_y*n_el_y-L_el_y*(r-1);
    a(n,:)=feval(a_fun,xp,yp);
end
a_x=a(:,1);
a_y=a(:,2);

for j=1:n_np_y
    [a_x_matrix(:,j)]=a_x((n_np_y-j)*n_np_x+1:(n_np_y-j)*n_np_x+n_np_x);
    [a_y_matrix(:,j)]=a_y((n_np_y-j)*n_np_x+1:(n_np_y-j)*n_np_x+n_np_x);
end
a_matrix=sqrt(a_x_matrix.^2+a_y_matrix.^2);

% Limit values
u_min=+inf;
u_max=-inf;
for k=1:length(T)
    if min(min(time(k).u_matrix))<u_min
        u_min=min(min(time(k).u_matrix));
    end
    if max(max(time(k).u_matrix))>u_max
        u_max=max(max(time(k).u_matrix));
    end
end

% Convection vector field
figure('Color',[1 1 1])
axes('FontSize',14)
quiver(X',Y',a_x_matrix,a_y_matrix,...
       'LineWidth',1,'Color',[1 0 0],'AutoScaleFactor',1.5)
title('Convection vector field','FontSize',14)
xlabel('x','FontSize',14)
ylabel('y','FontSize',14)
grid on
grid minor
xlim([x_i,x_f])
ylim([y_i,y_f])

% Convection field
figure('Color',[1 1 1])
axes('FontSize',14)
contourf(X',Y',a_matrix,'LineWidth',1)
title('Convection field','FontSize',14)
xlabel('x','FontSize',14)
ylabel('y','FontSize',14)
grid on
grid minor
xlim([x_i,x_f])
ylim([y_i,y_f])

% Initial conditions
figure('Color',[1 1 1])
axes('FontSize',14)
surf(X',Y',time(1).u_matrix)
title('Initial condition','FontSize',14)
xlabel('x','FontSize',14)
ylabel('y','FontSize',14)
zlabel('u_0(x,y)','FontSize',14)
grid on
grid minor
xlim([x_i,x_f])
ylim([y_i,y_f])
zlim([u_min,u_max])

% Final solution
figure('Color',[1 1 1])
axes('FontSize',14)
surf(X',Y',time(end).u_matrix)
title('Final solution','FontSize',14)
xlabel('x','FontSize',14)
ylabel('y','FontSize',14)
zlabel('u(x,y)','FontSize',14)
grid on
grid minor
xlim([x_i,x_f])
ylim([y_i,y_f])
zlim([u_min,u_max])

%% Animation

% Screen dimensions
scrsz=get(0,'ScreenSize');
bar=64;

% Simulation parameters
fact_ampl_sim=1;
fact_vel_sim=(t_f-t_i)/animation_time;
interval=0.0001;

i=1;
j=1;
tic

figure('Color',[1 1 1],'Position',[0 0 scrsz(3) (scrsz(4)-bar)])
while j<=length(T)
    
    % Plot
    surf(X',Y',time(j).u_matrix,'FaceColor','none')
    title(['Solution - t = ',num2str(round(T(j)*100)/100),' sec'],...
           'FontSize',14)
    xlabel('x','FontSize',14)
    ylabel('y','FontSize',14)
    zlabel('u(x,y)','FontSize',14)
    grid on
    grid minor
    xlim([x_i,x_f])
    ylim([y_i,y_f])
    zlim([u_min,u_max])
    
    % Time calibration
    i=i+1;
    time_real=toc;
    time_simulation=T(j)/fact_vel_sim;
    j=round(time_real/dt*fact_vel_sim);
    
    pause(interval);
end
relative_error=abs(time_simulation-time_real)/abs(time_real);

%% Display in command window

disp('MODEL PARAMETERS')
disp('-------------------------------------------------------------------')
fprintf('Length (x)\t\t\t=\t%.1f\n',L_x)
fprintf('Length (y)\t\t\t=\t%.1f\n',L_y)
fprintf('Convection velocity\t\t=\t%s\n',char(a_fun))
fprintf('Diffusion coefficient\t\t=\t%.3e\n',v)
fprintf('Initial condition\t\t=\t%s\n',char(u_0_fun))
fprintf('DOFs constrained\t\t=\t%s\n',dof_constrained_string)
fprintf('Boundary conditions\t\t=\t%.1f\n',bound_cond)
fprintf('Initial time\t\t\t=\t%.2f\n',t_i)
fprintf('Final time\t\t\t=\t%.2f\n',t_f)
fprintf('Time step\t\t\t=\t%.2e\n',dt)
fprintf('Theta\t\t\t\t=\t%.2f\n',theta)
disp('-------------------------------------------------------------------')

disp(' ')

disp('FEM PARAMETERS')
disp('-------------------------------------------------------------------')
fprintf('Number of finite elements (x)\t=\t%d\n',n_el_x)
fprintf('Number of finite elements (y)\t=\t%d\n',n_el_y)
fprintf('Length of a finite element (x)\t=\t%.2f\n',L_el_x)
fprintf('Length of a finite element (y)\t=\t%.2f\n',L_el_y)
fprintf('Gauss integration points\t=\t%d\n',n_gauss)
fprintf('Number of nodes\t\t\t=\t%d\n',n_np)
fprintf('Number of DOF per element\t=\t%d\n',dof_el)
fprintf('Total number of DOF\t\t=\t%d\n',dof)
disp('-------------------------------------------------------------------')

disp(' ')

disp('ELEMENT MASS MATRIX)')
disp('-------------------------------------------------------------------')
disp(el(1).M)
disp('-------------------------------------------------------------------')

disp(' ')

disp('ELEMENT CONVECTION MATRIX (element 1)')
disp('-------------------------------------------------------------------')
disp(el(1).C)
disp('-------------------------------------------------------------------')

disp(' ')

disp('ELEMENT DIFFUSION MATRIX')
disp('-------------------------------------------------------------------')
disp(el(1).K)
disp('-------------------------------------------------------------------')

disp(' ')

if dof<=10

disp('AFFERENCE MATRIX')
disp('-------------------------------------------------------------------')
disp(A)
disp('-------------------------------------------------------------------')

disp(' ')

disp('GLOBAL MASS MATRIX')
disp('-------------------------------------------------------------------')
disp(M)
disp('-------------------------------------------------------------------')

disp(' ')

disp('GLOBAL CONVECTION MATRIX')
disp('-------------------------------------------------------------------')
disp(C)
disp('-------------------------------------------------------------------')

disp(' ')

disp('GLOBAL DIFFUSION MATRIX')
disp('-------------------------------------------------------------------')
disp(K)
disp('-------------------------------------------------------------------')

disp(' ')

end

disp('EVALUATION OF THE SIMULATION''S ACCURACY')
disp('-------------------------------------------------------------------')
fprintf('Real time =\t\t%.2f\t[sec]\n',time_real)
fprintf('Simulation time =\t%.2f\t[sec]\n',time_simulation)
fprintf('Relative error =\t%.2f\t[%%]\n',relative_error*100)
disp('-------------------------------------------------------------------')