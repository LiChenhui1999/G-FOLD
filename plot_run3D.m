function[] = plot_run3D(tv, r, v, u, m)

u_norms = norms(u);
u_dirs = rad2deg(atan2(sqrt(u(1,:).^2 + u(2,:).^2), u(3,:)));
T_vals = u_norms .* m;

% ---- Figure 1: Trajectory (standalone, 3D) ----
figure('Name', 'Trajectory');
hold on; grid on; axis equal;
h_path   = plot3(r(1,:), r(2,:), r(3,:), 'b-', 'LineWidth', 1.5);
h_thrust = quiver3(r(1,:), r(2,:), r(3,:), ...
                   u(1,:), u(2,:), u(3,:), 0.25, ...
                   'Color', [0.85 0.33 0.10]);
h_start  = plot3(r(1,1),   r(2,1),   r(3,1),   'go', 'MarkerSize', 8, ...
                 'MarkerFaceColor', 'g');
h_end    = plot3(r(1,end), r(2,end), r(3,end), 'rs', 'MarkerSize', 8, ...
                 'MarkerFaceColor', 'r');
xlabel('x (m)'); ylabel('y (m)'); zlabel('z (m)');
title('Trajectory (m)');
view(3);
legend([h_path h_thrust h_start h_end], ...
       {'Path', 'Thrust direction', 'Start', 'End'}, ...
       'Location', 'best');

% ---- Figure 2: Kinematic states (2x2 subplots) ----
figure('Name', 'Kinematic states');

subplot(2,2,1); hold on; grid on;
plot(tv, r(1,:), 'LineWidth', 1.2);
plot(tv, r(2,:), 'LineWidth', 1.2);
plot(tv, r(3,:), 'LineWidth', 1.2);
xlabel('Time (s)'); ylabel('Position (m)');
title('Position');
legend({'r_x', 'r_y', 'r_z'}, 'Location', 'best');

subplot(2,2,2); hold on; grid on;
plot(tv, v(1,:), 'LineWidth', 1.2);
plot(tv, v(2,:), 'LineWidth', 1.2);
plot(tv, v(3,:), 'LineWidth', 1.2);
xlabel('Time (s)'); ylabel('Velocity (m/s)');
title('Velocity');
legend({'v_x', 'v_y', 'v_z'}, 'Location', 'best');

subplot(2,2,3); hold on; grid on;
plot(tv, u(1,:), 'LineWidth', 1.2);
plot(tv, u(2,:), 'LineWidth', 1.2);
plot(tv, u(3,:), 'LineWidth', 1.2);
xlabel('Time (s)'); ylabel('Acceleration (m/s^2)');
title('Commanded acceleration');
legend({'u_x', 'u_y', 'u_z'}, 'Location', 'best');

subplot(2,2,4); grid on;
plot(tv, m, 'LineWidth', 1.2);
xlabel('Time (s)'); ylabel('Mass (kg)');
title('Mass');
legend({'m'}, 'Location', 'best');

% ---- Figure 3: Thrust magnitudes & direction (1x3 subplots) ----
figure('Name', 'Thrust profile');

subplot(1,3,1); grid on;
plot(tv, T_vals, 'LineWidth', 1.2);
xlabel('Time (s)'); ylabel('Thrust (N)');
title('Thrust magnitude');
legend({'|T|'}, 'Location', 'best');

subplot(1,3,2); grid on;
plot(tv, u_norms, 'LineWidth', 1.2);
xlabel('Time (s)'); ylabel('Acceleration (m/s^2)');
title('Acceleration magnitude');
legend({'|u|'}, 'Location', 'best');

subplot(1,3,3); grid on;
plot(tv, u_dirs, 'LineWidth', 1.2);
xlabel('Time (s)'); ylabel('Angle (deg)');
title('Thrust direction (from vertical)');
legend({'\theta_u'}, 'Location', 'best');

end
