close all;
clear all;

% Spatial domain and mesh
a = 0 ; 
b = 1 ; 
N = 51 ; 
Dx = (b-a)/(N-1);

% Time domain and mesh 
t0 = 0 ; 
T = 1 ; 
h = 0.001 ; 

% Initial and boundary conditions
u0 = ones(N,1);
u0(1) = 0;
u0(end) = 1;


% Parameters
al = 0.5; % Fractional order
phi = 0.5*ones(N,1) ;  % Porosity
beta = 0.5*ones(N,1) ; % fractional porosity
c_diff = 0.01 ; % diffusion
c_advec = 0. ; % advection
FF = 0*eye(N,1) ; % forcing
bc = "dir"

% Advection diffusion operator
LL = c_diff*gallery('tridiag',N,1,-2,1)/Dx^2 - c_advec*gallery('tridiag',N,-1,1,0)/Dx;

if bc=="dir"     % Dirichlet Boundary conditions
    LL(1,:)=0;
    % LL(1,1)=1;
    LL(end,:)=0;
    % LL(end,end)=1;
    FF(1) = 0;
    FF(end) = 0;
elseif bc=="neu" % Homogeneous Neumann Boundary conditions
    tol = 1e5
    LL(1,:)=0;
    LL(1,1)=-tol;
    LL(1,2)=tol;
    LL(end,:)=0;
    LL(end,end)=-tol;
    LL(end,end-1)=tol;
    FF(1) = 0;
    FF(end) = 0;
elseif bc=="dirneu" % Dirichlet left, Neumann right
    LL(1,:)=0;
    % LL(1,1)=1;
    FF(1) = 0;
    tol = 1e5
    LL(end,:)=0;
    LL(end,end)=-tol;
    LL(end,end-1)=tol;
    FF(end) = 0;
end

[t, u] = Basset(al,phi,beta,LL,@(t) FF,t0,T,u0,h) ;

u_min = min(min(u)) ; 
u_max = max(max(u)) ;

%% Plot of the solution
% figure(1) 
% plot(Dx*(0:N-1),u(:,end)) 
% xlabel('x')
% axis([a,b,u_min,u_max]) ;
% title(sprintf('Solution at T = %f',T))

%% Polot of the solution over the time
figure(2)
K = size(u,2) ; 
F(K) = struct('cdata',[],'colormap',[]);
for k = 1 : 10 : K
    plot(Dx*(0:N-1),u(:,k),'.-')
    text(0.1,0.9,sprintf('t=%4.2f',t(k)), ...
        'Units','Normalized','FontSize',13) ;
    xlabel('x') ; ylabel('u(x,t)') ;
    axis([a-0.1,b+0.1,u_min,u_max]) ;
    %drawnow ;
    F(k) = getframe(gcf);
end

save("./results/out.mat","t","u")
