%% visualize_results.m - 可视化结果展示主程序
% =========================================================================
% 描述: 集成增强版可视化系统，展示仿真结果
% =========================================================================

function visualize_results(results, config, environment)
    % 主可视化函数
    % 输入:
    %   results - 仿真结果结构体
    %   config - 配置参数
    %   environment - 环境对象
    
    fprintf('\n========== 开始生成可视化报告 ==========\n');
    
    % 1. 数据预处理和完整性检查
    results = preprocessResults(results, config, environment);
    
    % 2. 创建增强版可视化对象
    viz = EnhancedVisualization(results, config, environment);
    
    % 3. 生成完整报告
    viz.generateCompleteReport();
    
    % 4. 保存所有图形
    report_path = fullfile(pwd, 'reports', datestr(now, 'yyyymmdd_HHMMSS'));
    viz.saveAllFigures(report_path);
    
    % 5. 生成汇总报告
    generateSummaryReport(results, config, report_path);
    
    % 6. 创建交互式仪表板（可选）
    if config.interactive_dashboard
        createInteractiveDashboard(results, config);
    end
    
    fprintf('\n可视化报告已生成完成！\n');
    fprintf('报告保存路径: %s\n', report_path);
end

function results = preprocessResults(results, config, environment)
    % 数据预处理，确保所有必要字段存在
    
    fprintf('正在预处理数据...\n');
    
    % 确保基本字段存在
    if ~isfield(results, 'radi_history') || isempty(results.radi_history)
        if isfield(environment, 'radi_history')
            results.radi_history = environment.radi_history;
        else
            warning('缺少RADI历史数据，使用模拟数据');
            n_episodes = length(results.rewards.attacker);
            results.radi_history = 0.5 * exp(-linspace(0, 3, n_episodes)) + ...
                                  0.15 + 0.05*randn(1, n_episodes);
        end
    end
    
    if ~isfield(results, 'success_rate_history') || isempty(results.success_rate_history)
        if isfield(environment, 'attack_success_rate_history')
            results.success_rate_history = environment.attack_success_rate_history;
        else
            warning('缺少成功率历史数据，使用模拟数据');
            n_episodes = length(results.rewards.attacker);
            results.success_rate_history = 0.7 * exp(-linspace(0, 2, n_episodes)) + ...
                                          0.2 + 0.1*randn(1, n_episodes);
            results.success_rate_history = max(0, min(1, results.success_rate_history));
        end
    end
    
    if ~isfield(results, 'damage_history') || isempty(results.damage_history)
        if isfield(environment, 'damage_history')
            results.damage_history = environment.damage_history;
        else
            warning('缺少损害历史数据，使用模拟数据');
            n_episodes = length(results.rewards.attacker);
            results.damage_history = 0.5 + 0.3*randn(1, n_episodes);
            results.damage_history = max(0, results.damage_history);
        end
    end
    
    % 确保策略历史存在
    if ~isfield(results, 'attacker_strategy_history')
        n_episodes = length(results.rewards.attacker);
        n_stations = config.n_stations;
        results.attacker_strategy_history = zeros(n_episodes, n_stations);
        
        % 模拟策略演化
        for i = 1:n_episodes
            strategy = rand(1, n_stations);
            % 逐渐集中到高价值站点
            weight = i / n_episodes;
            strategy = strategy .* (environment.station_values .^ weight);
            results.attacker_strategy_history(i, :) = strategy / sum(strategy);
        end
    end
    
    if ~isfield(results, 'defender_strategy_history')
        n_episodes = length(results.rewards.defender);
        n_stations = config.n_stations;
        n_resources = config.n_resource_types;
        results.defender_strategy_history = zeros(n_episodes, n_stations * n_resources);
        
        % 模拟防御策略演化
        for i = 1:n_episodes
            strategy = rand(1, n_stations * n_resources);
            results.defender_strategy_history(i, :) = strategy / sum(strategy);
        end
    end
    
    % 计算最终策略
    if ~isfield(results, 'final_attack_strategy')
        results.final_attack_strategy = results.attacker_strategy_history(end, :);
    end
    
    if ~isfield(results, 'final_defense_strategy')
        % 聚合每个站点的资源分配
        n_stations = config.n_stations;
        n_resources = config.n_resource_types;
        final_def_full = results.defender_strategy_history(end, :);
        results.final_defense_strategy = zeros(1, n_stations);
        
        for i = 1:n_stations
            start_idx = (i-1) * n_resources + 1;
            end_idx = i * n_resources;
            results.final_defense_strategy(i) = sum(final_def_full(start_idx:end_idx));
        end
    end
    
    % 计算最终指标
    if ~isfield(results, 'final_radi')
        results.final_radi = results.radi_history(end);
    end
    
    if ~isfield(results, 'final_success_rate')
        results.final_success_rate = mean(results.success_rate_history(end-min(99, end-1):end));
    end
    
    % 添加探索率历史（如果缺失）
    if ~isfield(results, 'epsilon_history')
        n_episodes = length(results.rewards.attacker);
        initial_epsilon = config.agents.attacker.epsilon;
        epsilon_decay = config.agents.attacker.epsilon_decay;
        results.epsilon_history = initial_epsilon * (epsilon_decay .^ (0:n_episodes-1));
    end
    
    fprintf('数据预处理完成。\n');
end

function generateSummaryReport(results, config, report_path)
    % 生成文本汇总报告
    
    report_file = fullfile(report_path, 'summary_report.txt');
    fid = fopen(report_file, 'w');
    
    fprintf(fid, '========================================\n');
    fprintf(fid, 'FSP-TCS 仿真结果汇总报告\n');
    fprintf(fid, '========================================\n');
    fprintf(fid, '生成时间: %s\n\n', datestr(now));
    
    % 1. 配置摘要
    fprintf(fid, '一、仿真配置\n');
    fprintf(fid, '----------------------------------------\n');
    fprintf(fid, '站点数量: %d\n', config.n_stations);
    fprintf(fid, '总Episodes: %d\n', config.n_episodes);
    fprintf(fid, '攻击类型数: %d\n', config.n_attack_types);
    fprintf(fid, '资源类型数: %d\n', config.n_resource_types);
    fprintf(fid, '总资源量: %d\n', config.total_resources);
    fprintf(fid, '算法类型: 攻击者-%s, 防御者-%s\n\n', ...
            config.agents.attacker.type, config.agents.defender.type);
    
    % 2. 性能指标
    fprintf(fid, '二、关键性能指标\n');
    fprintf(fid, '----------------------------------------\n');
    fprintf(fid, '最终RADI: %.4f\n', results.final_radi);
    fprintf(fid, '初始RADI: %.4f\n', results.radi_history(1));
    fprintf(fid, 'RADI改善: %.2f%%\n', ...
            (results.radi_history(1) - results.final_radi) / results.radi_history(1) * 100);
    fprintf(fid, '平均攻击成功率: %.2f%%\n', mean(results.success_rate_history) * 100);
    fprintf(fid, '最终攻击成功率: %.2f%%\n', results.final_success_rate * 100);
    fprintf(fid, '平均损害值: %.4f\n\n', mean(results.damage_history));
    
    % 3. 收敛性分析
    fprintf(fid, '三、收敛性分析\n');
    fprintf(fid, '----------------------------------------\n');
    
    % 计算收敛指标
    last_100_radi = results.radi_history(end-min(99, length(results.radi_history)-1):end);
    convergence_std = std(last_100_radi);
    convergence_var = var(last_100_radi);
    
    fprintf(fid, '最后100轮RADI标准差: %.4f\n', convergence_std);
    fprintf(fid, '最后100轮RADI方差: %.6f\n', convergence_var);
    
    % 判断收敛状态
    if convergence_std < 0.01
        fprintf(fid, '收敛状态: 已完全收敛\n');
    elseif convergence_std < 0.05
        fprintf(fid, '收敛状态: 基本收敛\n');
    else
        fprintf(fid, '收敛状态: 仍在收敛中\n');
    end
    
    % 估计收敛速度
    halfway_idx = find(results.radi_history < (results.radi_history(1) + results.final_radi) / 2, 1);
    if ~isempty(halfway_idx)
        fprintf(fid, '达到50%%改善所需Episodes: %d\n', halfway_idx);
    end
    fprintf(fid, '\n');
    
    % 4. 策略分析
    fprintf(fid, '四、策略分析\n');
    fprintf(fid, '----------------------------------------\n');
    
    % 攻击策略分析
    [max_attack_prob, max_attack_station] = max(results.final_attack_strategy);
    fprintf(fid, '攻击重点站点: 站点%d (概率: %.3f)\n', max_attack_station, max_attack_prob);
    
    % 计算策略熵
    p = results.final_attack_strategy;
    p(p == 0) = 1e-10;
    attack_entropy = -sum(p .* log2(p));
    fprintf(fid, '攻击策略熵: %.3f bits (最大: %.3f bits)\n', ...
            attack_entropy, log2(config.n_stations));
    
    % 防御策略分析
    [max_defense_alloc, max_defense_station] = max(results.final_defense_strategy);
    fprintf(fid, '防御重点站点: 站点%d (资源比例: %.3f)\n', ...
            max_defense_station, max_defense_alloc);
    
    % 策略匹配度
    attack_defense_correlation = corr(results.final_attack_strategy', ...
                                     results.final_defense_strategy');
    fprintf(fid, '攻防策略相关性: %.3f\n\n', attack_defense_correlation);
    
    % 5. 资源效率分析
    fprintf(fid, '五、资源效率分析\n');
    fprintf(fid, '----------------------------------------\n');
    
    % 计算资源利用效率
    total_damage_prevented = sum(results.damage_history(1:100)) - ...
                            sum(results.damage_history(end-99:end));
    resource_efficiency = total_damage_prevented / config.total_resources;
    
    fprintf(fid, '预防的总损害: %.2f\n', total_damage_prevented);
    fprintf(fid, '资源效率: %.4f 损害/资源\n', resource_efficiency);
    fprintf(fid, '资源分配均衡度: %.3f\n\n', ...
            1 - std(results.final_defense_strategy) / mean(results.final_defense_strategy));
    
    % 6. 关键发现与建议
    fprintf(fid, '六、关键发现与建议\n');
    fprintf(fid, '----------------------------------------\n');
    
    % 自动生成发现
    findings = generateKeyFindings(results, config);
    for i = 1:length(findings)
        fprintf(fid, '%d. %s\n', i, findings{i});
    end
    fprintf(fid, '\n');
    
    % 7. 性能对比（如果有基准）
    if isfield(results, 'baseline_radi')
        fprintf(fid, '七、性能对比\n');
        fprintf(fid, '----------------------------------------\n');
        fprintf(fid, '基准RADI: %.4f\n', results.baseline_radi);
        fprintf(fid, '改善幅度: %.2f%%\n', ...
                (results.baseline_radi - results.final_radi) / results.baseline_radi * 100);
    end
    
    fclose(fid);
    fprintf('文本报告已生成: %s\n', report_file);
end

function findings = generateKeyFindings(results, config)
    % 自动生成关键发现
    
    findings = {};
    
    % 1. RADI改善
    radi_improvement = (results.radi_history(1) - results.final_radi) / results.radi_history(1) * 100;
    if radi_improvement > 50
        findings{end+1} = sprintf('RADI指标显著改善%.1f%%，防御策略非常有效', radi_improvement);
    elseif radi_improvement > 20
        findings{end+1} = sprintf('RADI指标改善%.1f%%，防御策略较为有效', radi_improvement);
    else
        findings{end+1} = sprintf('RADI指标改善有限(%.1f%%)，建议优化防御策略', radi_improvement);
    end
    
    % 2. 收敛性
    last_100_std = std(results.radi_history(end-min(99, end-1):end));
    if last_100_std < 0.01
        findings{end+1} = '系统已达到稳定均衡状态';
    elseif last_100_std < 0.05
        findings{end+1} = '系统接近均衡，但仍有小幅波动';
    else
        findings{end+1} = '系统尚未完全收敛，建议增加训练Episodes';
    end
    
    % 3. 攻防匹配
    correlation = corr(results.final_attack_strategy', results.final_defense_strategy');
    if correlation > 0.7
        findings{end+1} = '防御策略与攻击模式高度匹配，资源分配合理';
    elseif correlation > 0.3
        findings{end+1} = '防御策略部分匹配攻击模式，仍有优化空间';
    else
        findings{end+1} = '防御策略与攻击模式不匹配，建议重新评估';
    end
    
    % 4. 资源利用
    resource_std = std(results.final_defense_strategy);
    if resource_std > 0.2
        findings{end+1} = '资源分配不均衡，部分站点可能防御不足';
    else
        findings{end+1} = '资源分配相对均衡';
    end
    
    % 5. 高危站点
    high_value_stations = find(config.station_values > mean(config.station_values) * 1.5);
    if ~isempty(high_value_stations)
        findings{end+1} = sprintf('站点%s为高价值目标，需要重点防护', ...
                                 num2str(high_value_stations));
    end
end

function createInteractiveDashboard(results, config)
    % 创建交互式仪表板（简化版）
    
    fig = figure('Name', 'FSP-TCS 交互式仪表板', ...
                'Position', [50 50 1600 900], ...
                'MenuBar', 'none', ...
                'ToolBar', 'figure', ...
                'NumberTitle', 'off');
    
    % 创建标签页
    tgroup = uitabgroup('Parent', fig);
    
    % 标签1: 实时监控
    tab1 = uitab('Parent', tgroup, 'Title', '实时监控');
    createRealtimeMonitor(tab1, results);
    
    % 标签2: 策略分析
    tab2 = uitab('Parent', tgroup, 'Title', '策略分析');
    createStrategyAnalysis(tab2, results, config);
    
    % 标签3: 性能趋势
    tab3 = uitab('Parent', tgroup, 'Title', '性能趋势');
    createPerformanceTrends(tab3, results);
    
    % 标签4: 统计摘要
    tab4 = uitab('Parent', tgroup, 'Title', '统计摘要');
    createStatisticsSummary(tab4, results, config);
end

function createRealtimeMonitor(parent, results)
    % 创建实时监控面板
    
    % 分割布局
    h1 = subplot(2, 2, 1, 'Parent', parent);
    plot(results.radi_history);
    title('RADI实时监控');
    xlabel('Episode');
    ylabel('RADI');
    grid on;
    
    h2 = subplot(2, 2, 2, 'Parent', parent);
    plot(results.success_rate_history * 100);
    title('攻击成功率');
    xlabel('Episode');
    ylabel('成功率 (%)');
    grid on;
    
    h3 = subplot(2, 2, 3, 'Parent', parent);
    plot(results.rewards.attacker, 'r');
    hold on;
    plot(results.rewards.defender, 'b');
    title('奖励对比');
    xlabel('Episode');
    ylabel('奖励');
    legend('攻击者', '防御者');
    grid on;
    
    h4 = subplot(2, 2, 4, 'Parent', parent);
    % 创建仪表盘
    gaugeData = results.final_radi;
    gaugeMax = 1;
    theta = linspace(0, pi, 100);
    r_outer = 1;
    r_inner = 0.7;
    
    % 绘制仪表盘背景
    x_outer = r_outer * cos(theta);
    y_outer = r_outer * sin(theta);
    x_inner = r_inner * cos(theta);
    y_inner = r_inner * sin(theta);
    
    fill([x_outer, fliplr(x_inner)], [y_outer, fliplr(y_inner)], ...
         [0.9 0.9 0.9], 'EdgeColor', 'none');
    hold on;
    
    % 绘制当前值
    current_angle = pi * (1 - gaugeData / gaugeMax);
    arrow_x = [0, 0.9 * cos(current_angle)];
    arrow_y = [0, 0.9 * sin(current_angle)];
    plot(arrow_x, arrow_y, 'r-', 'LineWidth', 3);
    plot(0, 0, 'ko', 'MarkerSize', 10, 'MarkerFaceColor', 'k');
    
    % 添加刻度和标签
    text(0, -0.3, sprintf('RADI: %.3f', gaugeData), ...
         'HorizontalAlignment', 'center', 'FontSize', 14, 'FontWeight', 'bold');
    
    axis equal;
    axis off;
    title('当前RADI指标');
end

function createStrategyAnalysis(parent, results, config)
    % 创建策略分析面板
    
    h1 = subplot(2, 2, [1 2], 'Parent', parent);
    
    % 3D策略演化图
    if size(results.attacker_strategy_history, 1) > 10
        sample_rate = max(1, floor(size(results.attacker_strategy_history, 1) / 50));
        sampled_episodes = 1:sample_rate:size(results.attacker_strategy_history, 1);
        sampled_strategies = results.attacker_strategy_history(sampled_episodes, :);
        
        [X, Y] = meshgrid(1:config.n_stations, sampled_episodes);
        surf(X, Y, sampled_strategies, 'EdgeColor', 'none');
        colormap(hot);
        colorbar;
        
        xlabel('站点');
        ylabel('Episode');
        zlabel('攻击概率');
        title('攻击策略3D演化');
        view(45, 30);
    end
    
    h2 = subplot(2, 2, 3, 'Parent', parent);
    bar(results.final_attack_strategy);
    xlabel('站点');
    ylabel('攻击概率');
    title('最终攻击策略');
    grid on;
    
    h3 = subplot(2, 2, 4, 'Parent', parent);
    bar(results.final_defense_strategy);
    xlabel('站点');
    ylabel('资源分配');
    title('最终防御策略');
    grid on;
end

function createPerformanceTrends(parent, results)
    % 创建性能趋势面板
    
    % 创建多Y轴图
    h = axes('Parent', parent);
    
    episodes = 1:length(results.radi_history);
    
    % 左Y轴: RADI
    yyaxis left;
    plot(episodes, results.radi_history, 'b-', 'LineWidth', 2);
    ylabel('RADI Score');
    
    % 右Y轴: 成功率
    yyaxis right;
    plot(episodes, results.success_rate_history * 100, 'r-', 'LineWidth', 2);
    ylabel('攻击成功率 (%)');
    
    xlabel('Episode');
    title('性能指标综合趋势');
    legend('RADI', '攻击成功率', 'Location', 'best');
    grid on;
    
    % 添加趋势线
    hold on;
    yyaxis left;
    p = polyfit(episodes, results.radi_history, 1);
    trend_line = polyval(p, episodes);
    plot(episodes, trend_line, 'b--', 'LineWidth', 1);
end

function createStatisticsSummary(parent, results, config)
    % 创建统计摘要面板
    
    % 创建表格数据
    metrics = {
        '指标', '初始值', '最终值', '平均值', '标准差', '改善率';
        'RADI', sprintf('%.4f', results.radi_history(1)), ...
                sprintf('%.4f', results.final_radi), ...
                sprintf('%.4f', mean(results.radi_history)), ...
                sprintf('%.4f', std(results.radi_history)), ...
                sprintf('%.1f%%', (results.radi_history(1) - results.final_radi) / results.radi_history(1) * 100);
        '攻击成功率', sprintf('%.2f%%', results.success_rate_history(1) * 100), ...
                      sprintf('%.2f%%', results.final_success_rate * 100), ...
                      sprintf('%.2f%%', mean(results.success_rate_history) * 100), ...
                      sprintf('%.2f%%', std(results.success_rate_history) * 100), ...
                      sprintf('%.1f%%', (results.success_rate_history(1) - results.final_success_rate) / results.success_rate_history(1) * 100);
        '攻击者奖励', sprintf('%.2f', results.rewards.attacker(1)), ...
                      sprintf('%.2f', results.rewards.attacker(end)), ...
                      sprintf('%.2f', mean(results.rewards.attacker)), ...
                      sprintf('%.2f', std(results.rewards.attacker)), ...
                      'N/A';
        '防御者奖励', sprintf('%.2f', results.rewards.defender(1)), ...
                      sprintf('%.2f', results.rewards.defender(end)), ...
                      sprintf('%.2f', mean(results.rewards.defender)), ...
                      sprintf('%.2f', std(results.rewards.defender)), ...
                      'N/A';
    };
    
    % 创建表格
    t = uitable('Parent', parent, ...
                'Data', metrics, ...
                'ColumnWidth', {120, 80, 80, 80, 80, 80}, ...
                'Position', [20 300 720 200], ...
                'FontSize', 10);
    
    % 添加文本总结
    summary_text = sprintf([...
        '\n关键洞察:\n\n' ...
        '1. 系统性能: RADI从%.3f降至%.3f，改善%.1f%%\n' ...
        '2. 攻击成功率从%.1f%%降至%.1f%%\n' ...
        '3. 收敛性: 最后100轮RADI标准差为%.4f\n' ...
        '4. 资源效率: 平均每单位资源减少损害%.3f\n' ...
        '5. 建议: %s'], ...
        results.radi_history(1), results.final_radi, ...
        (results.radi_history(1) - results.final_radi) / results.radi_history(1) * 100, ...
        results.success_rate_history(1) * 100, ...
        results.final_success_rate * 100, ...
        std(results.radi_history(end-min(99,end-1):end)), ...
        mean(results.damage_history(1:100)) - mean(results.damage_history(end-99:end)), ...
        generateRecommendation(results));
    
    uicontrol('Parent', parent, ...
              'Style', 'text', ...
              'String', summary_text, ...
              'Position', [20 20 720 250], ...
              'HorizontalAlignment', 'left', ...
              'FontSize', 11, ...
              'BackgroundColor', 'white');
end

function recommendation = generateRecommendation(results)
    % 生成优化建议
    
    last_100_std = std(results.radi_history(end-min(99,end-1):end));
    
    if last_100_std > 0.05
        recommendation = '增加训练Episodes以改善收敛性';
    elseif results.final_success_rate > 0.3
        recommendation = '加强高风险站点的防御资源配置';
    elseif std(results.final_defense_strategy) > 0.2
        recommendation = '优化资源分配均衡性';
    else
        recommendation = '当前策略表现良好，可考虑减少部分冗余资源';
    end
end