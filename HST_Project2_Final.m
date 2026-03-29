clear; clc; close all;

%% Parameters
alpha = 0.3;
beta  = 0.1;
gamma = 0.2;
rho   = 0.8;
delta = 0.2;
eta   = 0.5;
mu    = 0.3;
theta = 0.2;
lam   = 0.5;
omega = 0.3;
ks    = 0.3;

xf = log(201);   % fair-value log-price
ubar = [0;0];    % open-loop steady inputs


tanhsq = @(z) sech(z).^2;   % derivative of tanh is sech^2

% Nonlinear state update f(x,u). State x = [x1 x2 x3 x4 x5]^T, input u=[u1 u2]^T
f = @(x,u) [ ...
    x(1) + alpha*x(2) + gamma*x(4) - beta*(x(3)^2);                 % x1^+
    rho*tanh(x(1)-xf) - delta*x(2) + ks*u(2);                        % x2^+
    eta*(x(2)^2) - mu*x(3) + theta*u(1);                             % x3^+
    lam*(x(1)-x(5)) - omega*x(4);                                    % x4^+
    x(1)                                                             % x5^+
];

h = @(x) x(1);                 % output y = x1
C = [1 0 0 0 0];               % linear output matrix
D = zeros(1,2);

%% Equilibria (ū = 0)
% E0 (fundamental): x2=0 -> x1=xf, x3=0, x4=0, x5=x1
xbar0 = [xf; 0; 0; 0; xf];

% E+ (speculative): solve analytically from steady-state equations

x2p  = (alpha*mu^2/(beta*eta^2))^(1/3);
x1p  = xf + atanh( (x2p)/(rho/delta) );      % xf + atanh(x2/4)
x3p  = (eta/mu)*x2p^2;
x4p  = 0;
x5p  = x1p;
xbarp = [x1p; x2p; x3p; x4p; x5p];

fprintf('Equilibrium E0 = [x_f,0,0,0,x_f]^T with x_f = %.5f\n', xf);
fprintf('Equilibrium E+ ≈ [%.5f, %.5f, %.5f, 0, %.5f]^T\n', x1p, x2p, x3p, x5p);

%% Jacobians A,B at a general (x̄,ū) 
makeAB = @(xbar) deal( ...
[   1,                    alpha,               -2*beta*xbar(3),   gamma,     0;
    rho*tanhsq(xbar(1)-xf), -delta,            0,                 0,         0;
    0,                    2*eta*xbar(2),       -mu,               0,         0;
    lam,                  0,                    0,                -omega,    -lam;
    1,                    0,                    0,                 0,         0 ], ...
[   0,   0;
    0,   ks;
    theta, 0;
    0,   0;
    0,   0 ] );

[A0,B0] = makeAB(xbar0);
[Ap,Bp] = makeAB(xbarp);

Co_p = ctrb(Ap,Bp);  rankCo_p = rank(Co_p);
Ob_p = obsv(Ap,C);   rankOb_p = rank(Ob_p);
fprintf('rank(ctrb(Ap,Bp)) = %d (of 5)\n', rankCo_p);
fprintf('rank(obsv(Ap,C))  = %d (of 5)\n', rankOb_p);

%%  Open-loop eigenvalues
evA0 = eig(A0);
evAp = eig(Ap);
fprintf('\nOpen-loop eigenvalues at E0:\n'); disp(evA0.');
fprintf('Open-loop eigenvalues at E+:\n');  disp(evAp.');

%% Controllability / Observability
Co0 = ctrb(A0,B0);
rankCo0 = rank(Co0);
Ob0 = obsv(A0,C);
rankOb0 = rank(Ob0);
fprintf('rank(ctrb(A0,B0)) = %d (of 5)\n', rankCo0);
fprintf('rank(obsv(A0,C))  = %d (of 5)\n', rankOb0);

%% Controller design at E0

P_ctrl = [0.56+0.12i, 0.56-0.12i, -0.38, -0.58, -0.08];
K = place(A0,B0,P_ctrl);    
evAcl = eig(A0 - B0*K);
fprintf('\nPlaced closed-loop poles at E0:\n'); disp(evAcl.');

%% Observer design 

n  = size(A0,1);


Qn = 1e-3*eye(n);   
Rn = 1e-2;          

% Discrete-time 
[L, ~, evAobs] = dlqe(A0, eye(n), C, Qn, Rn);

fprintf('Observer error poles (A0 - L*C):\n'); disp(evAobs.');


%% Plot eigenvalues (open/closed)
figure('Name','Eigenvalue Map','Color','w'); hold on; grid on; axis equal;
th = linspace(0,2*pi,300); plot(cos(th), sin(th), '--k', 'LineWidth',1); % unit circle
plot(real(evA0),  imag(evA0),  'ro', 'DisplayName','Open-loop @E0');
plot(real(evAp),  imag(evAp),  'rx', 'DisplayName','Open-loop @E+');
plot(real(evAcl), imag(evAcl), 'bo', 'DisplayName','Closed-loop (A0-B0K)');
plot(real(evAobs),imag(evAobs),'gx', 'DisplayName','Observer (A0-LC)');
xlabel('Real'); ylabel('Imag'); title('Eigenvalues vs Unit Circle');
legend('Location','best');

%% Nonlinear simulation (closed loop)
% using LINEARIZED incremental observer (per lecture notes):
%   u(t)      = ū - K xhat_tilde(t)
%   xhat_t^+  = A0 xhat_t + B0 u_tilde + L*( y_tilde - C xhat_t - D u_tilde )
%   where xhat_t ≈ x(t)-xbar0, u_tilde = u - ū, y_tilde = y - ȳ, ȳ = x̄1

T = 250;                     
x  = [xf+0.8; 0.6; 0.6; 0.0; xf+0.8];  
xhat_til = zeros(5,1);        
u  = zeros(2,1);
ybar = xbar0(1);

% Optional practical guard: keep x3 >= 0 (market volume nonnegative)
guard_x3_nonneg = true;

X   = zeros(5,T+1);    X(:,1) = x;
XH  = zeros(5,T+1);    XH(:,1)= xbar0 + xhat_til;
U   = zeros(2,T);
Y   = zeros(1,T+1);    Y(1)   = h(x);

for t=1:T
    % incremental signals
    x_tilde = x - xbar0;
    y_tilde = h(x) - ybar;

    % control (observer-based, incremental)
    u_tilde = -K * xhat_til;
    u = ubar + u_tilde;

    % apply to plant (nonlinear), with optional nonnegativity guard on x3^+
    xnext = f(x,u);
    if guard_x3_nonneg && xnext(3) < 0
        xnext(3) = 0;
    end

    % observer update (linear incremental observer)
    xhat_til = A0*xhat_til + B0*u_tilde + L*( y_tilde - C*xhat_til - D*u_tilde );

    % log
    X(:,t+1)  = xnext;
    XH(:,t+1) = xbar0 + xhat_til;
    U(:,t)    = u;
    Y(t+1)    = h(xnext);

    % advance
    x = xnext;
end

%%  Plots: states vs targets 
t = 0:T;
figure('Name','States vs Target (Nonlinear Plant)','Color','w');
subplot(4,1,1);
plot(t, X(1,:), 'LineWidth',1.8); hold on; yline(xbar0(1),'--k','LineWidth',1.2);
ylabel('x_1 (log-price)'); grid on; title('States vs target (E0)');
subplot(4,1,2);
plot(t, X(2,:), 'LineWidth',1.8); hold on; yline(xbar0(2),'--k','LineWidth',1.2);
ylabel('x_2 (sentiment)'); grid on;
subplot(4,1,3);
plot(t, X(3,:), 'LineWidth',1.8); hold on; yline(xbar0(3),'--k','LineWidth',1.2);
ylabel('x_3 (spec vol)'); grid on;
subplot(4,1,4);
plot(t, X(4,:), 'LineWidth',1.8); hold on; yline(xbar0(4),'--k','LineWidth',1.2);
ylabel('x_4 (momentum)'); xlabel('time step'); grid on;

%%  Plots: true vs estimated 
figure('Name','True vs Estimated States','Color','w');
subplot(3,1,1);
plot(t, X(1,:),  'LineWidth',1.7); hold on;
plot(t, XH(1,:), '--', 'LineWidth',1.7);
ylabel('x_1 & \hat{x}_1'); grid on; title('True vs Estimated');
legend('x_1','\hat{x}_1','Location','best');

subplot(3,1,2);
plot(t, X(2,:),  'LineWidth',1.7); hold on;
plot(t, XH(2,:), '--', 'LineWidth',1.7);
ylabel('x_2 & \hat{x}_2'); grid on;
legend('x_2','\hat{x}_2','Location','best');

subplot(3,1,3);
plot(t, X(3,:),  'LineWidth',1.7); hold on;
plot(t, XH(3,:), '--', 'LineWidth',1.7);
ylabel('x_3 & \hat{x}_3'); xlabel('time step'); grid on;
legend('x_3','\hat{x}_3','Location','best');

%%  Plots: control inputs 
figure('Name','Control Inputs','Color','w');
plot(0:T-1, U(1,:), 'LineWidth',1.8); hold on;
plot(0:T-1, U(2,:), 'LineWidth',1.8);
yline(ubar(1),'--k'); yline(ubar(2),'--k');
xlabel('time step'); ylabel('u_1, u_2');
legend('u_1 (speculation control)','u_2 (sentiment control)','Location','best');
title('Control signals'); grid on;

%%  Print summary 
fprintf('\nSUMMARY:\n');
fprintf('- E0 open-loop unstable eigenvalues? max|lambda(A0)| = %.4f\n', max(abs(evA0)));
fprintf('- E+ open-loop unstable eigenvalues? max|lambda(Ap)| = %.4f\n', max(abs(evAp)));


% Add on for E_+  


fprintf('\n Add on for E_+\n');

%  Controller design (eigenvalue assignment) 
P_ctrl_p = [0.56+0.12i, 0.56-0.12i, -0.58, -0.38, -0.08];   % same targets as E0
Kp = place(Ap, Bp, P_ctrl_p);
evAcl_p = eig(Ap - Bp*Kp);
fprintf('E_+: Controller poles (Ap-Bp*Kp): '); disp(evAcl_p.');

%  Observer design (robust via discrete LQE) 
n  = size(Ap,1);
Qn = 1e-3*eye(n);       
Rn = 1e-2;             
[Lp, ~, evAobs_p] = dlqe(Ap, eye(n), C, Qn, Rn);
fprintf('E_+: Observer poles (Ap - Lp*C):   '); disp(evAobs_p.');
eig_obs_Ep = eig(Ap - Lp*C); 
disp('Observer eigenvalues at E+ (eig(Ap - Lp*C)) = ');
disp(eig_obs_Ep.');

%  Lyapunov certificate (closed loop at E_+) 
Qlyap = eye(n);
Pp = dlyap((Ap - Bp*Kp).', Qlyap);
mineigPp = min(eig(Pp));
fprintf('E_+: Lyapunov P min eig = %.4e (should be > 0)\n', mineigPp);

%  Nonlinear simulation around E_+ 
T2 = 250;
xbar = xbarp;               
A    = Ap;  B = Bp;   
ybar = xbar(1);

x   = xbar + [0.3; 0.2; 0.2; 0.0; 0.3];  
xhat_til = zeros(5,1);
u   = zeros(2,1);

guard_x3_nonneg = true;

X2   = zeros(5,T2+1);   X2(:,1) = x;
XH2  = zeros(5,T2+1);   XH2(:,1)= xbar + xhat_til;
U2   = zeros(2,T2);
Y2   = zeros(1,T2+1);   Y2(1)   = h(x);

for t=1:T2
    % incremental signals around E_+
    x_tilde = x - xbar;
    y_tilde = h(x) - ybar;

    % observer-based control (incremental)
    u_tilde = -Kp * xhat_til;
    u = ubar + u_tilde;

   
    xnext = f(x,u);
    if guard_x3_nonneg && xnext(3) < 0
        xnext(3) = 0;
    end

    % incremental observer around (A,B)
    xhat_til = A*xhat_til + B*u_tilde + Lp*( y_tilde - C*xhat_til - D*u_tilde );


    X2(:,t+1)  = xnext;
    XH2(:,t+1) = xbar + xhat_til;
    U2(:,t)    = u;
    Y2(t+1)    = h(xnext);

   
    x = xnext;
end

%  Figures (E_+) 
t2 = 0:T2;

% Eigenvalues (overlay on your existing map if desired)
figure('Name','Eigenvalues vs Unit Circle (E_+)','Color','w'); hold on; grid on; axis equal;
th = linspace(0,2*pi,300); plot(cos(th), sin(th), '--k', 'LineWidth',1);
plot(real(eig(Ap)),           imag(eig(Ap)),           'rx', 'DisplayName','Open-loop @E_+');
plot(real(evAcl_p),           imag(evAcl_p),           'bo', 'DisplayName','Closed-loop (Ap-BpKp)');
plot(real(evAobs_p),          imag(evAobs_p),          'gx', 'DisplayName','Observer (Ap-LpC)');
xlabel('Real'); ylabel('Imag'); title('Eigenvalues vs Unit Circle (E_+)');
legend('Location','best');

% States vs targets (E_+)
figure('Name','States vs Target (E_+)','Color','w');
subplot(4,1,1);
plot(t2, X2(1,:), 'LineWidth',1.8); hold on; yline(xbar(1),'--k','LineWidth',1.2);
ylabel('x_1 (log-price)'); grid on; title('States vs target (E_+)');
subplot(4,1,2);
plot(t2, X2(2,:), 'LineWidth',1.8); hold on; yline(xbar(2),'--k','LineWidth',1.2);
ylabel('x_2 (sentiment)'); grid on;
subplot(4,1,3);
plot(t2, X2(3,:), 'LineWidth',1.8); hold on; yline(xbar(3),'--k','LineWidth',1.2);
ylabel('x_3 (spec vol)'); grid on;
subplot(4,1,4);
plot(t2, X2(4,:), 'LineWidth',1.8); hold on; yline(xbar(4),'--k','LineWidth',1.2);
ylabel('x_4 (momentum)'); xlabel('time step'); grid on;

% True vs estimated (E_+) with LaTeX-safe labels
figure('Name','True vs Estimated States (E_+)','Color','w');
subplot(3,1,1);
plot(t2, X2(1,:),  'LineWidth',1.7); hold on;
plot(t2, XH2(1,:), '--', 'LineWidth',1.7);
ylabel('x_1 and \hat{x}_1','Interpreter','latex'); grid on; title('True vs Estimated (E_+)');
legend('x_1','\hat{x}_1','Interpreter','latex','Location','best');

subplot(3,1,2);
plot(t2, X2(2,:),  'LineWidth',1.7); hold on;
plot(t2, XH2(2,:), '--', 'LineWidth',1.7);
ylabel('x_2 and \hat{x}_2','Interpreter','latex'); grid on;
legend('x_2','\hat{x}_2','Interpreter','latex','Location','best');

subplot(3,1,3);
plot(t2, X2(3,:),  'LineWidth',1.7); hold on;
plot(t2, XH2(3,:), '--', 'LineWidth',1.7);
ylabel('x_3 and \hat{x}_3','Interpreter','latex'); xlabel('time step');
grid on; legend('x_3','\hat{x}_3','Interpreter','latex','Location','best');

% Controls (E_+)
figure('Name','Control Inputs (E_+)','Color','w');
plot(0:T2-1, U2(1,:), 'LineWidth',1.8); hold on;
plot(0:T2-1, U2(2,:), 'LineWidth',1.8);
yline(ubar(1),'--k'); yline(ubar(2),'--k');
xlabel('time step'); ylabel('u_1, u_2');
legend('u_1 (speculation control)','u_2 (sentiment control)','Location','best');
title('Control signals (E_+)'); grid on;

% Summary line
fprintf('Nonlinear sim to E_+: x1(T)=%.4f (target x_1^*=%.4f); x2^*=%.4f; x3^*=%.4f\n', ...
    X2(1,end), xbar(1), xbar(2), xbar(3));
