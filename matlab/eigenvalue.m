%% Setup
b = 1;
x=chebfun('x',[0,1]);
y=chebfun('y',[-b.^2/4+1e-5,100]);

%% Analytical eigenvalues
disc = sqrt(b.^2 + 4*y);
mu1 = -b + disc;
mu2 = - mu1 - 2*b;
f= mu1.*exp(mu1/2)-mu2.*exp(mu2/2);
l = roots(f);

y= chebfun('y',[-100,-b.^2/4-1e-5]);
disc = sqrt(-b.^2 - 4*y);
f = -b*sin(disc/2) + disc.*cos(disc/2);
l = max([roots(f);l]);

if (b.^2+4*l>0)
    discl = sqrt(b.^2 + 4*l);
    u = exp((-b+discl)*x/2) - exp((-b-discl)*x/2);
else
    discl = sqrt(-b.^2 - 4*l);
    u = exp(-x*b/2).*sin(x*disc(l)/2);
end

u = u/norm(u);

%% Eigenvalues with chebfun

L=chebop(@(u) -diff(u,2) + b*diff(u),[0,1]);
L.lbc='dirichlet';
L.rbc='neumann';

[vv,ll]=eigs(L,1);
vv = vv/norm(vv);


%% Compare chebfun and analytical
uu = exp(x*b/2).*sin(x*discl/2);
uu = uu/norm(uu);

disp([l,ll])

norm(u-vv)


%% Eigenvalues for different values of b
% Define the range of b values
b_values = 0:10:20;
num_b = length(b_values);

% Create a custom colormap that transitions from blue to red
colors = [linspace(0, 1, num_b)', zeros(num_b, 1), linspace(1, 0, num_b)'];

% Initialise figure
figure;
hold on;
xlabel('Real Part');
ylabel('Imaginary Part');
title('Eigenvalues for Different Values of b');
grid on;
axis equal;

% Loop through values of b with unique colors
for k = 1:num_b
    b = b_values(k);
    
    % Define the operator with current value of b
    L.op = @(u) diff(u, 2) + b * diff(u);
    
    % Compute the first 15 eigenvalues
    eigenvalues = eigs(L, 151) + 1e-10i;
    
    % Plot eigenvalues with the colour from the colormap
    plot(eigenvalues, 'x', 'Color', colors(k, :), 'DisplayName', sprintf('b = %d', b));
end

%

% Show legend to differentiate b values
legend show;


%% Eigenvalues discrete

i=1;
for k = 2:12
    m = 2^k+1;
    dx = 1.0/2^k;
    M = (diag(2*ones(m,1)) - diag(ones(m-1,1),1) - diag(ones(m-1,1),-1))/(dx*dx) + b*(diag(ones(m,1)) - diag(ones(m-1,1),-1))/dx;
    M(1,2) = 0;
    M(end,end-1)=-M(end,end);
    % [v,l] = eigs(inv(M),1);
    [v,l] = sipi(M,eye(size(M,2)),1000,1e-10,-1);
    disp(l)
    dof(i)    = m;
    lambda(i) = l;
    i = i+1;
end

%% test m=10

m = 11;
dx = 0.1;
M = (diag(2*ones(m,1)) - diag(ones(m-1,1),1) - diag(ones(m-1,1),-1))/(dx*dx) + b*(diag(ones(m,1)) - diag(ones(m-1,1),-1))/dx;
M(1,2) = 0;
M(end,end-1)=-M(end,end);
% [v,l] = eigs(inv(M),1);
[v,l] = sipi(M,eye(size(M,2)),1000,1e-10,-1);
disp(l)



%% Functions

function [b_k,eigenvalue] = sipi(L, M, num_iterations, tolerance, tau)
    % Compute the largest eigenvalue and eigenvector
    % using the inverse shifted power iteration method.
    %
    % Parameters:
    % L: Input matrix L (double or sparse).
    % M: Input matrix M (double or sparse).
    % num_iterations: Maximum number of iterations (default: 1000).
    % tolerance: Convergence tolerance (default: 1e-10).
    % tau: Shift parameter (default: 1.0).
    %
    % Returns:
    % eigenvalue: Dominant eigenvalue.
    % b_k: Corresponding eigenvector.

    if nargin < 3
        num_iterations = 1000;
    end
    if nargin < 4
        tolerance = 1e-10;
    end
    if nargin < 5
        tau = 1.0;
    end

    % Start with a random vector of 1s
    n = size(L, 2);
    b_k = ones(n, 1);
    b_k(1) = 0.0; % First component is set to zero
    eigenvalue_k = 0.0;

    % Construct the shifted matrix
    matrix = L - tau * M;
    disp(matrix)

    % Print the full matrix
    % disp(full(matrix));

    for i = 1:num_iterations
        % Solve the linear system
        b_k1 = matrix \ b_k;

        % Compute the Rayleigh quotient
        eigenvalue_k1 = (b_k1' * b_k) / (b_k' * b_k);

        fprintf('Iteration: %d, Eigenvalue: %f\n', i, eigenvalue_k1);

        % Check for convergence
        if abs(eigenvalue_k - eigenvalue_k1) < tolerance
            break;
        end

        % Update the eigenvector and eigenvalue
        b_k = b_k1 / norm(b_k1);
        eigenvalue_k = eigenvalue_k1;
    end

    % Compute the final eigenvalue
    eigenvalue = (1.0 / eigenvalue_k + tau);

end