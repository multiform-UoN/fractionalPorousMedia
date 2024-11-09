
b=-10;

%% Analytical eigenvalues
y=chebfun('y',[-b.^2/4+1e-5,100]);
disc = sqrt(b.^2 + 4*y);
mu1 = -b + disc;
mu2 = - mu1 - 2*b;
f= mu1.*exp(mu1/2)-mu2.*exp(mu2/2);
l = roots(f);

y= chebfun('y',[-100,-b.^2/4-1e-5]);
disc = sqrt(-b.^2 - 4*y);
f = -b*sin(disc/2) + disc.*cos(disc/2);
l = max([roots(f);l]);

x=chebfun('x',[0,1]);
if (b.^2+4*l>0)
    discl = sqrt(b.^2 + 4*l);
    u = exp((-b+discl)*x/2) - exp((-b-discl)*x/2);
else
    discl = sqrt(-b.^2 - 4*l);
    u = exp(-x*b/2).*sin(x*disc(l)/2);
end

u = u/norm(u);

%% Eigenvalues with chebfun

L=chebop(@(u) diff(u,2)+b*diff(u),[0,1]);
L.lbc='dirichlet';
L.rbc='neumann';

[v,ll]=eigs(L,1);
v = v/norm(v);

uu = exp(x*b/2).*sin(x*discl/2);
uu = uu/norm(uu);

disp([l,ll])

norm(u-v)


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