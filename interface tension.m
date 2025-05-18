%% 终极Young-Laplace求解器（解决局部最优问题）
clear all; close all; clc;

%% 物理参数设置
rho_w = 1000;       % 水密度 (kg/m³)
rho_o = 915.8;      % 油密度 (kg/m³)
gamma_ow = 31.8e-3; % 界面张力 (N/m)
g = 9.81;           % 重力加速度 (m/s²)
V_b = 1e-8;         % 水滴体积 (m³)

%% 特征参数计算
Lc = sqrt(gamma_ow / ((rho_w - rho_o) * g));
R_eq = (3*V_b/(4*pi))^(1/3);
Bo = ((rho_w - rho_o) * g * R_eq^2) / gamma_ow;

fprintf('系统参数:\n');
fprintf('  毛细长度: %.3f mm\n', Lc*1e3);
fprintf('  等效半径: %.3f mm\n', R_eq*1e3);
fprintf('  Bond数: %.4f\n', Bo);

%% 多阶段优化策略
theta_guess = deg2rad(90);
theta_range = [deg2rad(60), deg2rad(120)];

% 阶段1：全局搜索
fprintf('\n=== 阶段1：全局搜索 ===\n');
gs = GlobalSearch('Display', 'iter', 'NumStageOnePoints', 200, 'NumTrialPoints', 400);
problem = createOptimProblem('fmincon',...
    'objective', @(theta) yl_objective(theta, rho_w, rho_o, gamma_ow, g, V_b, Lc, R_eq),...
    'x0', theta_guess,...
    'lb', theta_range(1),...
    'ub', theta_range(2),...
    'options', optimoptions('fmincon', 'Display', 'off'));
[theta_sol, fval] = run(gs, problem);

% 阶段2：局部精修
fprintf('\n=== 阶段2：局部精修 ===\n');
options = optimoptions('fmincon',...
    'Algorithm', 'interior-point',...
    'Display', 'iter',...
    'MaxIterations', 100,...
    'StepTolerance', 1e-10,...
    'FunctionTolerance', 1e-10);
theta_sol = fmincon(@(theta) yl_objective(theta, rho_w, rho_o, gamma_ow, g, V_b, Lc, R_eq),...
            theta_sol, [], [], [], [], theta_range(1), theta_range(2), [], options);

%% 最终验证
[r, z] = solve_yl_shape(theta_sol, Lc, R_eq);
volume = compute_precise_volume(r, z);

fprintf('\n=== 最终结果 ===\n');
fprintf('  最优接触角: %.4f°\n', rad2deg(theta_sol));
fprintf('  计算体积: %.8e m³ (误差: %.6f%%)\n', volume, abs(volume-V_b)/V_b*100);

%% 增强可视化
plot_enhanced_results(r, z, theta_sol, gamma_ow, Lc);

%% 目标函数（带物理约束）
function error = yl_objective(theta, ~, ~, ~, ~, V_b, Lc, R_eq)
    [r, z, success] = solve_yl_shape(theta, Lc, R_eq);
    
    % 失败惩罚
    if ~success || isempty(r)
        error = 1e6 + 1e5*(theta - pi/2)^2;
        return;
    end
    
    % 体积计算
    volume = compute_precise_volume(r, z);
    vol_err = abs(volume - V_b)/V_b;
    
    % 几何约束
    height_err = (z(end)/Lc)^2;
    angle_err = (atan2(z(10)-z(1), r(10)-r(1)) - theta)^2;
    smoothness = sum(abs(diff(z,2)).^2)/length(z);
    
    % 加权目标
    error = vol_err + 1e3*height_err + 1e2*angle_err + 1e1*smoothness;
end

%% 稳健形状求解
function [r, z, success] = solve_yl_shape(theta, Lc, R_eq)
    success = false;
    
    % 自适应初始条件
    r0 = max(R_eq * 1e-3, 1e-6);
    z0 = r0 * tan(theta);
    
    % 动态积分范围
    s_max = min(20*Lc, 40*R_eq);
    
    % 增强ODE设置
    opts = odeset('RelTol', 1e-7, 'AbsTol', 1e-9,...
                 'MaxStep', Lc/100,...
                 'Events', @(s,y) yl_events(s,y,Lc));
    
    try
        % 使用ode15s处理刚性问题
        sol = ode15s(@(s,y) yl_system(s,y,Lc,R_eq,theta), [0 s_max], [r0; z0; theta], opts);
        
        % 提取有效解
        r = sol.y(1,:)';
        z = sol.y(2,:)';
        valid = (z >= -1e-6) & (r <= 15*R_eq);
        r = r(valid); z = z(valid);
        
        % 重采样保证质量
        if length(r) > 10
            [r, z] = resample_profile(r, z, 500);
            success = true;
        end
    catch
        r = []; z = [];
    end
end

%% Young-Laplace系统
function dyds = yl_system(~, y, Lc, R_eq, theta)
    r = y(1); z = y(2); phi = y(3);
    
    % 超稳定曲率计算
    if r < 1e-5
        if abs(phi) < 1e-3
            curvature = (2/R_eq)*sin(theta) - phi/r + phi^3/(6*r);
        else
            curvature = (2/R_eq)*sin(theta) - sin(phi)/r;
        end
    else
        curvature = (2/R_eq)*sin(theta) - sin(phi)/r;
    end
    
    dyds = [cos(phi);
            sin(phi);
            z/Lc^2 + curvature];
end

%% 事件检测
function [value,isterminal,direction] = yl_events(~,y,Lc)
    value = [y(2) + 1e-5*Lc;    % 触底
             y(1) - 25*Lc];      % 过度膨胀
    isterminal = [1; 1];
    direction = [-1; 1];
end

%% 超高精度体积计算
function vol = compute_precise_volume(r, z)
    if isempty(r) || length(r) < 3
        vol = 0;
        return;
    end
    
    % 确保单调性
    [r, idx] = unique(r);
    z = z(idx);
    
    % 自适应Simpson积分
    n = length(r)-1;
    h = diff(r);
    vol_segments = zeros(n,1);
    
    for i = 1:n
        r0 = r(i); r1 = r(i+1);
        z0 = z(i); z1 = z(i+1);
        rm = (r0 + r1)/2;
        zm = interp1(r, z, rm, 'spline');
        
        % 每个区间的Simpson积分
        vol_segments(i) = h(i)/6 * (r0^2*z0 + 4*rm^2*zm + r1^2*z1);
    end
    
    vol = pi * sum(vol_segments);
end

%% 轮廓重采样
function [r_new, z_new] = resample_profile(r, z, n)
    s = cumsum([0; sqrt(diff(r).^2 + diff(z).^2)]);
    s_new = linspace(0, s(end), n)';
    r_new = interp1(s, r, s_new, 'spline');
    z_new = interp1(s, z, s_new, 'spline');
end


%% 精确曲率计算
function [kappa, s] = compute_curvature(r, z)
    s = cumsum([0; sqrt(diff(r).^2 + diff(z).^2)]);
    
    % 使用5点中心差分
    drds = gradient(r, s);
    dzds = gradient(z, s);
    d2rds2 = gradient(drds, s);
    d2zds2 = gradient(dzds, s);
    
    kappa = (dzds.*d2rds2 - drds.*d2zds2) ./ (drds.^2 + dzds.^2).^(3/2);
    
    % 过滤异常值
    kappa(abs(kappa) > 1e4) = median(kappa);
end