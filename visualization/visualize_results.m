%% visualize_results.m - 可视化结果展示主程序
% =========================================================================
% 描述: 集成增强版可视化系统，展示仿真结果
% =========================================================================
function visualize_results(results, config, save_dir)
    %% 增强可视化功能：添加RADI、Nash均衡收敛度、攻击覆盖率变化曲线
    % 输入:
    %   results - 仿真结果结构体
    %   config - 配置参数
    %   save_dir - 保存目录
    
    if nargin < 3
        save_dir = './visualization_results';
    end
    
    % 确保保存目录存在
    if ~exist(save_dir, 'dir')
        mkdir(save_dir);
    end
    
    fprintf('开始生成增强可视化图表...\n');
    
    %% 1. RADI变化曲线图
    generateRADITrendPlot(results, save_dir);
    
    %% 2. Nash均衡收敛度变化曲线
    generateNashConvergencePlot(results, save_dir);
    
    %% 3. 攻击覆盖率变化曲线
    generateAttackCoveragePlot(results, save_dir);
    
    %% 4. 综合指标对比图
    generateComprehensiveMetricsPlot(results, save_dir);
    
    %% 5. 三维演化图
    generate3DEvolutionPlot(results, save_dir);
    
    %% 6. 性能热力图
    generatePerformanceHeatmap(results, save_dir);
    
    fprintf('✓ 所有增强可视化图表已生成完成！\n');
    fprintf('图表保存位置: %s\n', save_dir);
end

function generateRADITrendPlot(results, save_dir)
    %% 生成RADI变化曲线图
    
    figure('Position', [100, 100, 1200, 600]);
    
    % 获取RADI数据
    if isfield(results, 'radi_history')
        radi_data = results.radi_history;
        episodes = 1:length(radi_data);
    elseif isfield(results, 'radi')
        radi_data = mean(results.radi, 1); % 如果是多智能体，取平均
        episodes = 1:length(radi_data);
    else
        error('未找到RADI历史数据');
    end
    
    % 主图：RADI变化趋势
    subplot(2, 2, 1);
    plot(episodes, radi_data, 'b-', 'LineWidth', 2);
    hold on;
    
    % 添加趋势线
    if length(episodes) > 10
        p = polyfit(episodes, radi_data, 1);
        trend_line = polyval(p, episodes);
        plot(episodes, trend_line, 'r--', 'LineWidth', 1.5, 'DisplayName', '趋势线');
    end
    
    % 添加性能阈值线
    if exist('config', 'var') && isfield(config, 'radi')
        if isfield(config.radi, 'threshold_excellent')
            yline(config.radi.threshold_excellent, 'g--', '优秀阈值', 'LineWidth', 1);
        end
        if isfield(config.radi, 'threshold_acceptable')
            yline(config.radi.threshold_acceptable, 'y--', '可接受阈值', 'LineWidth', 1);
        end
    end
    
    title('RADI指标变化趋势', 'FontSize', 14, 'FontWeight', 'bold');
    xlabel('训练轮次', 'FontSize', 12);
    ylabel('RADI值', 'FontSize', 12);
    grid on;
    legend('RADI', '趋势线', 'Location', 'best');
    
    % 子图：RADI改善率
    subplot(2, 2, 2);
    if length(radi_data) > 1
        improvement_rate = (radi_data(1) - radi_data) ./ radi_data(1) * 100;
        plot(episodes, improvement_rate, 'g-', 'LineWidth', 2);
        title('RADI改善率', 'FontSize', 14, 'FontWeight', 'bold');
        xlabel('训练轮次', 'FontSize', 12);
        ylabel('改善率 (%)', 'FontSize', 12);
        grid on;
    end
    
    % 子图：RADI稳定性分析
    subplot(2, 2, 3);
    window_size = min(50, floor(length(radi_data)/4));
    if window_size > 1
        moving_std = movingstd(radi_data, window_size);
        plot(episodes, moving_std, 'm-', 'LineWidth', 2);
        title('RADI稳定性（移动标准差）', 'FontSize', 14, 'FontWeight', 'bold');
        xlabel('训练轮次', 'FontSize', 12);
        ylabel('移动标准差', 'FontSize', 12);
        grid on;
    end
    
    % 子图：RADI分布直方图
    subplot(2, 2, 4);
    histogram(radi_data, 30, 'FaceColor', 'skyblue', 'EdgeColor', 'black');
    title('RADI值分布', 'FontSize', 14, 'FontWeight', 'bold');
    xlabel('RADI值', 'FontSize', 12);
    ylabel('频次', 'FontSize', 12);
    grid on;
    
    % 添加统计信息
    mean_radi = mean(radi_data);
    std_radi = std(radi_data);
    final_radi = radi_data(end);
    
    annotation('textbox', [0.02, 0.02, 0.3, 0.15], ...
               'String', sprintf('统计信息:\n平均RADI: %.4f\n标准差: %.4f\n最终值: %.4f', ...
                                mean_radi, std_radi, final_radi), ...
               'BackgroundColor', 'white', ...
               'EdgeColor', 'black', ...
               'FontSize', 10);
    
    sgtitle('RADI指标综合分析', 'FontSize', 16, 'FontWeight', 'bold');
    
    % 保存图形
    saveas(gcf, fullfile(save_dir, 'radi_analysis.png'));
    saveas(gcf, fullfile(save_dir, 'radi_analysis.fig'));
    close(gcf);
    
    fprintf('✓ RADI变化曲线图已生成\n');
end

function generateNashConvergencePlot(results, save_dir)
    %% 生成Nash均衡收敛度变化曲线
    
    figure('Position', [150, 150, 1200, 600]);
    
    % 计算Nash收敛指标
    nash_conv = calculateNashConvergence(results);
    episodes = 1:length(nash_conv);
    
    % 主图：Nash收敛度变化
    subplot(2, 2, 1);
    plot(episodes, nash_conv, 'r-', 'LineWidth', 2);
    hold on;
    
    % 添加收敛阈值
    convergence_threshold = 0.01;
    yline(convergence_threshold, 'g--', '收敛阈值', 'LineWidth', 1.5);
    
    title('Nash均衡收敛度', 'FontSize', 14, 'FontWeight', 'bold');
    xlabel('训练轮次', 'FontSize', 12);
    ylabel('收敛度指标', 'FontSize', 12);
    grid on;
    legend('Nash收敛度', '收敛阈值', 'Location', 'best');
    
    % 子图：对数尺度收敛图
    subplot(2, 2, 2);
    semilogy(episodes, nash_conv, 'b-', 'LineWidth', 2);
    title('Nash收敛度（对数尺度）', 'FontSize', 14, 'FontWeight', 'bold');
    xlabel('训练轮次', 'FontSize', 12);
    ylabel('收敛度指标 (log)', 'FontSize', 12);
    grid on;
    
    % 子图：收敛速度
    subplot(2, 2, 3);
    if length(nash_conv) > 1
        conv_speed = -diff(nash_conv);  % 负的差分表示收敛速度
        plot(episodes(2:end), conv_speed, 'g-', 'LineWidth', 2);
        title('收敛速度', 'FontSize', 14, 'FontWeight', 'bold');
        xlabel('训练轮次', 'FontSize', 12);
        ylabel('收敛速度', 'FontSize', 12);
        grid on;
    end
    
    % 子图：收敛状态分析
    subplot(2, 2, 4);
    converged_episodes = nash_conv < convergence_threshold;
    convergence_ratio = cumsum(converged_episodes) ./ episodes;
    plot(episodes, convergence_ratio * 100, 'purple', 'LineWidth', 2);
    title('累积收敛率', 'FontSize', 14, 'FontWeight', 'bold');
    xlabel('训练轮次', 'FontSize', 12);
    ylabel('收敛率 (%)', 'FontSize', 12);
    grid on;
    
    % 添加统计信息
    final_conv = nash_conv(end);
    converged_at = find(converged_episodes, 1);
    if isempty(converged_at)
        converged_at = NaN;
    end
    
    annotation('textbox', [0.02, 0.02, 0.3, 0.15], ...
               'String', sprintf('收敛分析:\n最终收敛度: %.6f\n首次收敛轮次: %d\n收敛阈值: %.6f', ...
                                final_conv, converged_at, convergence_threshold), ...
               'BackgroundColor', 'white', ...
               'EdgeColor', 'black', ...
               'FontSize', 10);
    
    sgtitle('Nash均衡收敛分析', 'FontSize', 16, 'FontWeight', 'bold');
    
    % 保存图形
    saveas(gcf, fullfile(save_dir, 'nash_convergence.png'));
    saveas(gcf, fullfile(save_dir, 'nash_convergence.fig'));
    close(gcf);
    
    fprintf('✓ Nash均衡收敛度图已生成\n');
end

function generateAttackCoveragePlot(results, save_dir)
    %% 生成攻击覆盖率变化曲线
    
    figure('Position', [200, 200, 1200, 600]);
    
    % 计算攻击覆盖率
    attack_coverage = calculateAttackCoverage(results);
    episodes = 1:length(attack_coverage);
    
    % 主图：攻击覆盖率变化
    subplot(2, 2, 1);
    plot(episodes, attack_coverage * 100, 'orange', 'LineWidth', 2);
    hold on;
    
    % 添加目标覆盖率线
    target_coverage = 80; % 目标覆盖率80%
    yline(target_coverage, 'g--', '目标覆盖率', 'LineWidth', 1.5);
    
    title('攻击覆盖率变化', 'FontSize', 14, 'FontWeight', 'bold');
    xlabel('训练轮次', 'FontSize', 12);
    ylabel('覆盖率 (%)', 'FontSize', 12);
    ylim([0, 100]);
    grid on;
    legend('攻击覆盖率', '目标覆盖率', 'Location', 'best');
    
    % 子图：覆盖率改善趋势
    subplot(2, 2, 2);
    if length(attack_coverage) > 1
        coverage_improvement = attack_coverage - attack_coverage(1);
        plot(episodes, coverage_improvement * 100, 'blue', 'LineWidth', 2);
        title('覆盖率改善', 'FontSize', 14, 'FontWeight', 'bold');
        xlabel('训练轮次', 'FontSize', 12);
        ylabel('改善百分点', 'FontSize', 12);
        grid on;
    end
    
    % 子图：覆盖率稳定性
    subplot(2, 2, 3);
    window_size = min(50, floor(length(attack_coverage)/4));
    if window_size > 1
        moving_variance = movingvar(attack_coverage, window_size);
        plot(episodes, moving_variance, 'red', 'LineWidth', 2);
        title('覆盖率稳定性', 'FontSize', 14, 'FontWeight', 'bold');
        xlabel('训练轮次', 'FontSize', 12);
        ylabel('移动方差', 'FontSize', 12);
        grid on;
    end
    
    % 子图：防御有效性分析
    subplot(2, 2, 4);
    if isfield(results, 'success_rate_history')
        defense_effectiveness = (1 - results.success_rate_history) * 100;
        scatter(attack_coverage * 100, defense_effectiveness, 50, episodes, 'filled');
        colorbar;
        title('覆盖率 vs 防御有效性', 'FontSize', 14, 'FontWeight', 'bold');
        xlabel('攻击覆盖率 (%)', 'FontSize', 12);
        ylabel('防御有效性 (%)', 'FontSize', 12);
        grid on;
    end
    
    % 添加统计信息
    mean_coverage = mean(attack_coverage) * 100;
    final_coverage = attack_coverage(end) * 100;
    max_coverage = max(attack_coverage) * 100;
    
    annotation('textbox', [0.02, 0.02, 0.3, 0.15], ...
               'String', sprintf('覆盖率统计:\n平均覆盖率: %.1f%%\n最终覆盖率: %.1f%%\n最大覆盖率: %.1f%%', ...
                                mean_coverage, final_coverage, max_coverage), ...
               'BackgroundColor', 'white', ...
               'EdgeColor', 'black', ...
               'FontSize', 10);
    
    sgtitle('攻击覆盖率分析', 'FontSize', 16, 'FontWeight', 'bold');
    
    % 保存图形
    saveas(gcf, fullfile(save_dir, 'attack_coverage.png'));
    saveas(gcf, fullfile(save_dir, 'attack_coverage.fig'));
    close(gcf);
    
    fprintf('✓ 攻击覆盖率图已生成\n');
end

function generateComprehensiveMetricsPlot(results, save_dir)
    %% 生成综合指标对比图
    
    figure('Position', [250, 250, 1400, 800]);
    
    % 获取数据
    if isfield(results, 'radi_history')
        radi_data = results.radi_history;
    else
        radi_data = mean(results.radi, 1);
    end
    
    nash_conv = calculateNashConvergence(results);
    attack_coverage = calculateAttackCoverage(results);
    episodes = 1:length(radi_data);
    
    % 标准化数据用于对比
    radi_norm = (radi_data - min(radi_data)) / (max(radi_data) - min(radi_data));
    nash_norm = (nash_conv - min(nash_conv)) / (max(nash_conv) - min(nash_conv));
    coverage_norm = attack_coverage;
    
    % 主对比图
    subplot(2, 3, [1, 2]);
    plot(episodes, radi_norm, 'b-', 'LineWidth', 2, 'DisplayName', 'RADI (标准化)');
    hold on;
    plot(episodes, 1 - nash_norm, 'r-', 'LineWidth', 2, 'DisplayName', 'Nash收敛 (标准化)');
    plot(episodes, coverage_norm, 'orange', 'LineWidth', 2, 'DisplayName', '攻击覆盖率');
    
    title('关键指标综合对比', 'FontSize', 16, 'FontWeight', 'bold');
    xlabel('训练轮次', 'FontSize', 12);
    ylabel('标准化值', 'FontSize', 12);
    legend('show', 'Location', 'best');
    grid on;
    
    % 性能雷达图
    subplot(2, 3, 3);
    final_metrics = [
        1 - radi_norm(end),        % RADI性能 (越低越好，所以用1-x)
        1 - nash_norm(end),        % Nash收敛性能
        coverage_norm(end),        % 攻击覆盖率
        calculateResourceEfficiency(results),  % 资源效率
        calculateSystemStability(results)      % 系统稳定性
    ];
    
    angles = linspace(0, 2*pi, length(final_metrics) + 1);
    final_metrics = [final_metrics, final_metrics(1)]; % 闭合雷达图
    
    polarplot(angles, final_metrics, 'ro-', 'LineWidth', 2, 'MarkerSize', 8);
    rlim([0, 1]);
    thetaticks(rad2deg(angles(1:end-1)));
    thetaticklabels({'RADI性能', 'Nash收敛', '攻击覆盖', '资源效率', '系统稳定'});
    title('最终性能雷达图', 'FontSize', 14, 'FontWeight', 'bold');
    
    % 趋势分析
    subplot(2, 3, 4);
    window_size = min(20, floor(length(episodes)/5));
    if window_size > 1
        radi_trend = movmean(radi_data, window_size);
        nash_trend = movmean(nash_conv, window_size);
        coverage_trend = movmean(attack_coverage, window_size);
        
        yyaxis left;
        plot(episodes, radi_trend, 'b-', 'LineWidth', 2);
        ylabel('RADI值', 'Color', 'b', 'FontSize', 12);
        
        yyaxis right;
        plot(episodes, nash_trend, 'r-', 'LineWidth', 2);
        plot(episodes, coverage_trend, 'orange', 'LineWidth', 2);
        ylabel('收敛度 / 覆盖率', 'Color', 'r', 'FontSize', 12);
        
        title('趋势分析（移动平均）', 'FontSize', 14, 'FontWeight', 'bold');
        xlabel('训练轮次', 'FontSize', 12);
    end
    
    % 相关性分析
    subplot(2, 3, 5);
    correlation_data = [radi_data', nash_conv', attack_coverage'];
    corr_matrix = corrcoef(correlation_data);
    imagesc(corr_matrix);
    colorbar;
    colormap('RdBu');
    caxis([-1, 1]);
    
    labels = {'RADI', 'Nash收敛', '攻击覆盖'};
    xticks(1:3);
    yticks(1:3);
    xticklabels(labels);
    yticklabels(labels);
    title('指标相关性矩阵', 'FontSize', 14, 'FontWeight', 'bold');
    
    % 添加相关系数文本
    for i = 1:3
        for j = 1:3
            text(j, i, sprintf('%.2f', corr_matrix(i,j)), ...
                 'HorizontalAlignment', 'center', ...
                 'FontSize', 12, 'FontWeight', 'bold');
        end
    end
    
    % 性能改善汇总
    subplot(2, 3, 6);
    improvements = [
        (radi_data(1) - radi_data(end)) / radi_data(1) * 100,  % RADI改善
        (nash_conv(1) - nash_conv(end)) / nash_conv(1) * 100,  % Nash收敛改善
        (attack_coverage(end) - attack_coverage(1)) * 100      % 覆盖率提升
    ];
    
    bar_colors = {'blue', 'red', 'orange'};
    b = bar(improvements);
    for i = 1:length(improvements)
        b.FaceColor = 'flat';
        b.CData(i,:) = hex2rgb(bar_colors{i});
    end
    
    title('性能改善汇总', 'FontSize', 14, 'FontWeight', 'bold');
    ylabel('改善百分比 (%)', 'FontSize', 12);
    xticklabels({'RADI改善', 'Nash收敛改善', '覆盖率提升'});
    grid on;
    
    % 添加数值标签
    for i = 1:length(improvements)
        text(i, improvements(i) + sign(improvements(i))*2, ...
             sprintf('%.1f%%', improvements(i)), ...
             'HorizontalAlignment', 'center', ...
             'FontWeight', 'bold');
    end
    
    sgtitle('系统性能综合分析仪表板', 'FontSize', 18, 'FontWeight', 'bold');
    
    % 保存图形
    saveas(gcf, fullfile(save_dir, 'comprehensive_metrics.png'));
    saveas(gcf, fullfile(save_dir, 'comprehensive_metrics.fig'));
    close(gcf);
    
    fprintf('✓ 综合指标对比图已生成\n');
end

function generate3DEvolutionPlot(results, save_dir)
    %% 生成三维演化图
    
    figure('Position', [300, 300, 1200, 800]);
    
    % 获取数据
    if isfield(results, 'radi_history')
        radi_data = results.radi_history;
    else
        radi_data = mean(results.radi, 1);
    end
    
    nash_conv = calculateNashConvergence(results);
    attack_coverage = calculateAttackCoverage(results);
    episodes = 1:length(radi_data);
    
    % 3D轨迹图
    subplot(2, 2, [1, 2]);
    plot3(radi_data, nash_conv, attack_coverage, 'b-', 'LineWidth', 2);
    hold on;
    
    % 标记起点和终点
    scatter3(radi_data(1), nash_conv(1), attack_coverage(1), 100, 'g', 'filled', 'DisplayName', '起点');
    scatter3(radi_data(end), nash_conv(end), attack_coverage(end), 100, 'r', 'filled', 'DisplayName', '终点');
    
    xlabel('RADI值', 'FontSize', 12);
    ylabel('Nash收敛度', 'FontSize', 12);
    zlabel('攻击覆盖率', 'FontSize', 12);
    title('三维性能空间演化轨迹', 'FontSize', 14, 'FontWeight', 'bold');
    legend('show');
    grid on;
    view(45, 30);
    
    % 时间色彩映射的3D图
    subplot(2, 2, 3);
    scatter3(radi_data, nash_conv, attack_coverage, 50, episodes, 'filled');
    colorbar;
    xlabel('RADI值', 'FontSize', 12);
    ylabel('Nash收敛度', 'FontSize', 12);
    zlabel('攻击覆盖率', 'FontSize', 12);
    title('时间演化三维散点图', 'FontSize', 14, 'FontWeight', 'bold');
    view(-45, 20);
    
    % 投影到2D平面
    subplot(2, 2, 4);
    scatter(radi_data, attack_coverage, 50, nash_conv, 'filled');
    colorbar;
    xlabel('RADI值', 'FontSize', 12);
    ylabel('攻击覆盖率', 'FontSize', 12);
    title('RADI vs 覆盖率 (颜色=Nash收敛度)', 'FontSize', 14, 'FontWeight', 'bold');
    grid on;
    
    sgtitle('三维性能演化分析', 'FontSize', 16, 'FontWeight', 'bold');
    
    % 保存图形
    saveas(gcf, fullfile(save_dir, '3d_evolution.png'));
    saveas(gcf, fullfile(save_dir, '3d_evolution.fig'));
    close(gcf);
    
    fprintf('✓ 三维演化图已生成\n');
end

function generatePerformanceHeatmap(results, save_dir)
    %% 生成性能热力图
    
    figure('Position', [350, 350, 1000, 600]);
    
    % 获取数据
    if isfield(results, 'radi_history')
        radi_data = results.radi_history;
    else
        radi_data = mean(results.radi, 1);
    end
    
    nash_conv = calculateNashConvergence(results);
    attack_coverage = calculateAttackCoverage(results);
    
    % 创建时间窗口分析
    window_size = 50;
    n_windows = floor(length(radi_data) / window_size);
    
    if n_windows > 1
        heatmap_data = zeros(n_windows, 3);
        
        for i = 1:n_windows
            start_idx = (i-1) * window_size + 1;
            end_idx = min(i * window_size, length(radi_data));
            
            heatmap_data(i, 1) = mean(radi_data(start_idx:end_idx));
            heatmap_data(i, 2) = mean(nash_conv(start_idx:end_idx));
            heatmap_data(i, 3) = mean(attack_coverage(start_idx:end_idx));
        end
        
        % 标准化数据
        heatmap_data_norm = (heatmap_data - min(heatmap_data)) ./ (max(heatmap_data) - min(heatmap_data));
        
        subplot(1, 2, 1);
        imagesc(heatmap_data_norm');
        colorbar;
        colormap('hot');
        
        yticks(1:3);
        yticklabels({'RADI', 'Nash收敛', '攻击覆盖'});
        xlabel('时间窗口', 'FontSize', 12);
        title('性能指标时间热力图', 'FontSize', 14, 'FontWeight', 'bold');
        
        % 性能评分热力图
        subplot(1, 2, 2);
        
        % 计算综合性能评分
        performance_scores = zeros(n_windows, 1);
        for i = 1:n_windows
            score = (1 - heatmap_data_norm(i, 1)) * 0.4 + ...  % RADI (越低越好)
                    (1 - heatmap_data_norm(i, 2)) * 0.3 + ...  % Nash收敛
                    heatmap_data_norm(i, 3) * 0.3;             % 攻击覆盖率
            performance_scores(i) = score;
        end
        
        imagesc(performance_scores');
        colorbar;
        colormap('RdYlGn');
        
        xlabel('时间窗口', 'FontSize', 12);
        title('综合性能评分', 'FontSize', 14, 'FontWeight', 'bold');
        yticks([]);
        
        sgtitle('性能热力图分析', 'FontSize', 16, 'FontWeight', 'bold');
    else
        % 如果数据不足，显示简单的性能对比
        final_metrics = [radi_data(end), nash_conv(end), attack_coverage(end)];
        bar(final_metrics);
        title('最终性能指标', 'FontSize', 14, 'FontWeight', 'bold');
        xticklabels({'RADI', 'Nash收敛', '攻击覆盖'});
        ylabel('指标值', 'FontSize', 12);
        grid on;
    end
    
    % 保存图形
    saveas(gcf, fullfile(save_dir, 'performance_heatmap.png'));
    saveas(gcf, fullfile(save_dir, 'performance_heatmap.fig'));
    close(gcf);
    
    fprintf('✓ 性能热力图已生成\n');
end

%% 辅助计算函数

function nash_conv = calculateNashConvergence(results)
    %% 计算Nash均衡收敛度指标
    
    % 如果结果中已有Nash收敛数据，直接使用
    if isfield(results, 'nash_conv')
        nash_conv = results.nash_conv;
        return;
    end
    
    % 否则基于策略变化计算收敛度
    if isfield(results, 'attack_strategy_history') && isfield(results, 'defense_strategy_history')
        attack_strategies = results.attack_strategy_history;
        defense_strategies = results.defense_strategy_history;
        
        n_episodes = size(attack_strategies, 1);
        nash_conv = zeros(n_episodes, 1);
        
        % 计算策略变化的收敛度
        for i = 2:n_episodes
            % 攻击策略变化
            attack_change = norm(attack_strategies(i,:) - attack_strategies(i-1,:));
            
            % 防御策略变化
            defense_change = norm(defense_strategies(i,:) - defense_strategies(i-1,:));
            
            % 综合收敛度 (策略变化越小，收敛度越小)
            nash_conv(i) = (attack_change + defense_change) / 2;
        end
        
        % 第一个点设为较大值
        nash_conv(1) = max(nash_conv(2:end)) * 1.5;
        
    elseif isfield(results, 'exploitability')
        % 使用可利用性作为Nash收敛度代理
        nash_conv = results.exploitability;
        
    else
        % 基于RADI变化估算收敛度
        if isfield(results, 'radi_history')
            radi_data = results.radi_history;
        else
            radi_data = mean(results.radi, 1);
        end
        
        nash_conv = zeros(size(radi_data));
        window_size = min(10, floor(length(radi_data)/5));
        
        for i = window_size+1:length(radi_data)
            radi_window = radi_data(i-window_size:i);
            nash_conv(i) = std(radi_window); % 用标准差作为收敛度指标
        end
        
        % 填充前面的值
        nash_conv(1:window_size) = nash_conv(window_size+1);
    end
    
    % 确保收敛度非负
    nash_conv = max(nash_conv, 0);
end

function attack_coverage = calculateAttackCoverage(results)
    %% 计算攻击覆盖率（防御系统能有效防御的攻击类型比例）
    
    % 如果结果中已有攻击覆盖率数据，直接使用
    if isfield(results, 'attack_coverage')
        attack_coverage = results.attack_coverage;
        return;
    end
    
    % 基于成功率计算覆盖率
    if isfield(results, 'success_rate_history')
        success_rates = results.success_rate_history;
        % 覆盖率 = 1 - 成功率 (防御成功的比例)
        attack_coverage = 1 - success_rates;
        
    elseif isfield(results, 'detection_rates')
        % 使用检测率作为覆盖率代理
        detection_rates = results.detection_rates;
        if size(detection_rates, 1) > 1
            attack_coverage = mean(detection_rates, 1);
        else
            attack_coverage = detection_rates;
        end
        
    elseif isfield(results, 'episode_detection_rates')
        % 从episode检测率计算
        detection_data = results.episode_detection_rates;
        if ndims(detection_data) == 3
            % [n_agents x n_episodes x n_iterations]
            attack_coverage = squeeze(mean(mean(detection_data, 1), 2));
        else
            attack_coverage = mean(detection_data, 1);
        end
        
    else
        % 基于RADI改善估算覆盖率
        if isfield(results, 'radi_history')
            radi_data = results.radi_history;
        else
            radi_data = mean(results.radi, 1);
        end
        
        % RADI改善越大，覆盖率越高
        initial_radi = radi_data(1);
        radi_improvement = (initial_radi - radi_data) / initial_radi;
        
        % 映射到[0.3, 0.9]范围的覆盖率
        attack_coverage = 0.3 + 0.6 * max(0, radi_improvement);
        attack_coverage = min(attack_coverage, 0.9); % 限制最大值
    end
    
    % 确保覆盖率在合理范围内[0, 1]
    attack_coverage = max(0, min(attack_coverage, 1));
end

function efficiency = calculateResourceEfficiency(results)
    %% 计算资源效率
    
    if isfield(results, 'resource_efficiency')
        if size(results.resource_efficiency, 1) > 1
            efficiency = mean(results.resource_efficiency(:, end));
        else
            efficiency = results.resource_efficiency(end);
        end
    else
        % 基于RADI和成功率估算效率
        if isfield(results, 'radi_history')
            radi_data = results.radi_history;
        else
            radi_data = mean(results.radi, 1);
        end
        
        if isfield(results, 'success_rate_history')
            success_rate = results.success_rate_history(end);
        else
            success_rate = 0.3; % 默认值
        end
        
        % 效率 = (1 - RADI) * (1 - 成功率)
        efficiency = (1 - radi_data(end)) * (1 - success_rate);
        efficiency = max(0, min(efficiency, 1));
    end
end

function stability = calculateSystemStability(results)
    %% 计算系统稳定性
    
    if isfield(results, 'radi_history')
        radi_data = results.radi_history;
    else
        radi_data = mean(results.radi, 1);
    end
    
    % 计算最后20%数据的稳定性
    last_portion = max(1, floor(length(radi_data) * 0.2));
    recent_data = radi_data(end-last_portion+1:end);
    
    % 稳定性 = 1 - 相对标准差
    if mean(recent_data) > 0
        cv = std(recent_data) / mean(recent_data); % 变异系数
        stability = max(0, 1 - cv);
    else
        stability = 0.5; % 默认中等稳定性
    end
    
    stability = min(stability, 1); % 限制最大值
end

function moving_stat = movingstd(data, window_size)
    %% 计算移动标准差
    
    n = length(data);
    moving_stat = zeros(size(data));
    
    for i = 1:n
        start_idx = max(1, i - window_size + 1);
        end_idx = i;
        moving_stat(i) = std(data(start_idx:end_idx));
    end
end

function moving_stat = movingvar(data, window_size)
    %% 计算移动方差
    
    n = length(data);
    moving_stat = zeros(size(data));
    
    for i = 1:n
        start_idx = max(1, i - window_size + 1);
        end_idx = i;
        moving_stat(i) = var(data(start_idx:end_idx));
    end
end

function rgb = hex2rgb(hex_color)
    %% 将十六进制颜色转换为RGB
    
    switch hex_color
        case 'blue'
            rgb = [0, 0.4470, 0.7410];
        case 'red'
            rgb = [0.8500, 0.3250, 0.0980];
        case 'orange'
            rgb = [0.9290, 0.6940, 0.1250];
        case 'purple'
            rgb = [0.4940, 0.1840, 0.5560];
        case 'green'
            rgb = [0.4660, 0.6740, 0.1880];
        otherwise
            rgb = [0.3010, 0.7450, 0.9330]; % 默认浅蓝色
    end
end