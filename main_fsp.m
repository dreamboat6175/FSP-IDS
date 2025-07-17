function enhanced_main_fsp()
    % FSP-TCS框架主函数 - 多智能体版本
    
    clc; clear; close all;
    
    % 添加路径
    addpath(genpath(pwd));
    
    % 初始化日志系统
    log_file = fullfile(pwd, 'logs', ['simulation_' datestr(now, 'yyyymmdd_HHMMSS') '.log']);
    logger = FSPLogger(log_file, 'INFO');
    
    try
        logger.info('FSP-TCS多智能体仿真开始');
        
        % 创建配置
        config = createMultiAgentConfiguration();
        
        % 创建环境
        environment = TCSEnvironment(config);
        
        % 创建多个防御智能体
        agents = createMultipleAgents(environment, config);
        
        % 创建性能监控器（每个防御者一个）
        monitors = createMultipleMonitors(config);
        
        % 运行仿真
        results = runMultiAgentSimulation(environment, agents, monitors, config, logger);
        
        % 生成增强版报告
        generateEnhancedReport(results, config, logger);
        
        logger.info('FSP-TCS多智能体仿真成功完成');
        
    catch ME
        logger.error(sprintf('仿真出错: %s', ME.message));
        logger.error(sprintf('错误位置: %s', ME.stack(1).file));
        rethrow(ME);
    end
end

function config = createMultiAgentConfiguration()
    % 创建多智能体配置
    config = struct();
    
    % === 基础环境参数 ===
    config.n_stations = 10;
    config.n_components_per_station = repmat(5, 1, 10);
    config.total_resources = 100;
    config.random_seed = 42;
    
    % === 站点价值设置 ===
    config.station_values = [0.8, 0.6, 0.9, 0.5, 0.7, 1.0, 0.6, 0.8, 0.7, 0.95];
    [~, config.highest_value_station] = max(config.station_values);  % 找出最高价值站点
    
    % === 初始策略设置 ===
    config.initial_defense_strategy = 'random';  % 防守资源随机分配
    config.initial_attack_strategy = 'focused';   % 攻击集中在最高价值站点
    config.attack_focus_ratio = 0.8;              % 80%攻击资源集中在最高价值站点
    
    % === 仿真参数 ===
    config.n_episodes = 1000;  % 增加到1000轮
    config.max_steps_per_episode = 100;
    config.convergence_threshold = 0.01;
    config.performance_check_interval = 50;
    
    % === 多防御智能体配置 ===
    config.defender_types = {'QLearning', 'SARSA', 'DoubleQLearning'};
    config.n_defenders = length(config.defender_types);
    
    % === 智能体配置 ===
    config.agents = struct();
    
    % 攻击者智能体（共享）
    config.agents.attacker = struct();
    config.agents.attacker.type = 'QLearning';
    config.agents.attacker.learning_rate = 0.1;
    config.agents.attacker.discount_factor = 0.95;
    config.agents.attacker.epsilon = 0.3;
    config.agents.attacker.epsilon_decay = 0.995;
    config.agents.attacker.epsilon_min = 0.01;
    
    % 防御者智能体（多个）
    config.agents.defenders = {};
    for i = 1:config.n_defenders
        config.agents.defenders{i} = struct();
        config.agents.defenders{i}.type = config.defender_types{i};
        config.agents.defenders{i}.learning_rate = 0.05;
        config.agents.defenders{i}.discount_factor = 0.95;
        config.agents.defenders{i}.epsilon = 0.3;
        config.agents.defenders{i}.epsilon_decay = 0.995;
        config.agents.defenders{i}.epsilon_min = 0.01;
    end
    
    % === 其他配置 ===
    config.resource_types = {'cpu', 'memory', 'network', 'storage', 'security'};
    config.attack_types = {'dos', 'intrusion', 'malware', 'social', 'physical', 'cyber'};
    config.n_resource_types = length(config.resource_types);
    config.n_attack_types = length(config.attack_types);
    
    % === 输出配置 ===
    config.output = struct();
    config.output.save_models = true;
    config.output.generate_plots = true;
    config.output.save_data = true;
    config.output.results_dir = fullfile(pwd, 'results');
    config.output.verbose = true; % 详细输出
    
    if ~exist(config.output.results_dir, 'dir')
        mkdir(config.output.results_dir);
    end
end

function agents = createMultipleAgents(environment, config)
    % 创建多个智能体
    
    % 创建攻击者智能体（共享）
    agents.attacker = AgentFactory.createAttackerAgent(config, environment);
    
    % 创建多个防御者智能体
    agents.defenders = AgentFactory.createDefenderAgents(config, environment);
end

function monitors = createMultipleMonitors(config)
    % 为每个防御者创建监控器
    
    monitors = {};
    for i = 1:config.n_defenders
        monitors{i} = struct();
        monitors{i}.name = config.defender_types{i};
        monitors{i}.radi_history = [];
        monitors{i}.damage_history = [];
        monitors{i}.success_rate_history = [];
        monitors{i}.detection_rate_history = [];
        monitors{i}.episode_rewards = [];
    end
end

function results = runMultiAgentSimulation(environment, agents, monitors, config, logger)
    % 运行多智能体仿真
    
    results = initializeMultiAgentResults(config);
    
    for episode = 1:config.n_episodes
        % 为每个防御者运行episode
        episode_results = {};
        
        for def_idx = 1:config.n_defenders
            % 重置环境
            state = environment.reset();
            
            % 运行单个episode
            episode_data = runSingleEpisode(environment, agents.attacker, ...
                                           agents.defenders{def_idx}, state, config);
            
            % 更新监控器
            monitors{def_idx} = updateMonitor(monitors{def_idx}, episode_data, environment);
            
            % 存储episode结果
            episode_results{def_idx} = episode_data;
        end
        
        % 更新结果
        results = updateMultiAgentResults(results, episode_results, monitors, episode);
        
        % 定期输出详细信息
        if mod(episode, config.performance_check_interval) == 0
            displayDetailedProgress(episode, agents, monitors, config, logger);
        end
    end
    
    % 最终处理
    results = finalizeResults(results, monitors, agents, config);
end

function episode_data = runSingleEpisode(environment, attacker, defender, initial_state, config)
    % 运行单个episode
    
    state = initial_state;
    episode_data = struct();
    episode_data.states = [];
    episode_data.actions = struct('attacker', [], 'defender', []);
    episode_data.rewards = struct('attacker', [], 'defender', []);
    episode_data.episode_reward = struct('attacker', 0, 'defender', 0);
    episode_data.attack_success = [];
    episode_data.detection_success = [];
    episode_data.damages = [];
    
    for step = 1:config.max_steps_per_episode
        % 智能体选择动作
        attacker_action = attacker.selectAction(state);
        defender_action_raw = defender.selectAction(state);
        
        % 保证传给环境的是向量
        defender_action = defender_action_raw;
        if isscalar(defender_action_raw)
            defender_action = environment.parseDefenderAction(defender_action_raw);
        end
        
        % 环境执行动作
        [next_state, reward_def, reward_att, info] = environment.step(defender_action, attacker_action);
        
        % 记录数据
        episode_data.states = [episode_data.states; state];
        episode_data.actions.attacker = [episode_data.actions.attacker; attacker_action];
        episode_data.actions.defender = [episode_data.actions.defender; defender_action];
        episode_data.rewards.attacker = [episode_data.rewards.attacker; reward_att];
        episode_data.rewards.defender = [episode_data.rewards.defender; reward_def];
        
        % 记录攻击和检测结果
        episode_data.attack_success = [episode_data.attack_success; info.attack_success];
        % episode_data.detection_success = [episode_data.detection_success; info.detection_success];
        episode_data.damages = [episode_data.damages; info.damage];
        
        % 累计奖励
        episode_data.episode_reward.attacker = episode_data.episode_reward.attacker + reward_att;
        episode_data.episode_reward.defender = episode_data.episode_reward.defender + reward_def;
        
        % 智能体学习
        attacker.update(state, attacker_action, reward_att, next_state, []);
        defender.update(state, defender_action_raw, reward_def, next_state, []);
        
        % 状态转移
        state = next_state;
    end
end

function monitor = updateMonitor(monitor, episode_data, environment)
    % 更新监控器数据
    
    monitor.radi_history(end+1) = environment.radi_score;
    monitor.damage_history(end+1) = mean(episode_data.damages);
    monitor.success_rate_history(end+1) = mean(episode_data.attack_success);
    monitor.detection_rate_history(end+1) = mean(episode_data.detection_success);
    monitor.episode_rewards(end+1, :) = [episode_data.episode_reward.attacker, ...
                                         episode_data.episode_reward.defender];
end

function displayDetailedProgress(episode, agents, monitors, config, logger)
    % 显示详细的进度信息
    
    logger.info(sprintf('\n========== Episode %d ==========', episode));
    
    % 显示攻击者策略
    if ismethod(agents.attacker, 'getStrategy')
        att_strategy = agents.attacker.getStrategy();
        logger.info(sprintf('攻击者策略: [%s]', num2str(att_strategy, '%.3f ')));
    end
    
    % 显示每个防御者的信息
    for i = 1:config.n_defenders
        logger.info(sprintf('\n--- %s 防御者 ---', config.defender_types{i}));
        
        % 防御策略
        if ismethod(agents.defenders{i}, 'getStrategy')
            def_strategy = agents.defenders{i}.getStrategy();
            logger.info(sprintf('防御策略: [%s]', num2str(def_strategy, '%.3f ')));
        end
        
        % 性能指标
        logger.info(sprintf('RADI: %.3f', monitors{i}.radi_history(end)));
        logger.info(sprintf('Damage: %.3f', monitors{i}.damage_history(end)));
        logger.info(sprintf('Success Rate: %.3f', monitors{i}.success_rate_history(end)));
        logger.info(sprintf('Detection Rate: %.3f', monitors{i}.detection_rate_history(end)));
    end
    
    logger.info('================================\n');
end

function results = initializeMultiAgentResults(config)
    % 初始化多智能体结果结构
    
    results = struct();
    results.config = config;
    results.episodes = [];
    
    % 为每个防御者初始化结果
    for i = 1:config.n_defenders
        name = config.defender_types{i};
        results.(name) = struct();
        results.(name).radi_history = [];
        results.(name).damage_history = [];
        results.(name).success_rate_history = [];
        results.(name).detection_rate_history = [];
        results.(name).rewards = struct('attacker', [], 'defender', []);
        results.(name).strategy_history = [];
    end
    
    % 攻击者结果
    results.attacker = struct();
    results.attacker.strategy_history = [];
end

function results = updateMultiAgentResults(results, episode_results, monitors, episode)
    % 更新多智能体结果
    
    results.episodes(end+1) = episode;
    
    % 更新每个防御者的结果
    for i = 1:length(monitors)
        name = monitors{i}.name;
        results.(name).radi_history = monitors{i}.radi_history;
        results.(name).damage_history = monitors{i}.damage_history;
        results.(name).success_rate_history = monitors{i}.success_rate_history;
        results.(name).detection_rate_history = monitors{i}.detection_rate_history;
        results.(name).rewards.attacker = [results.(name).rewards.attacker; ...
                                           monitors{i}.episode_rewards(end, 1)];
        results.(name).rewards.defender = [results.(name).rewards.defender; ...
                                          monitors{i}.episode_rewards(end, 2)];
    end
end

function results = finalizeResults(results, monitors, agents, config)
    % 最终处理结果
    
    % 计算最终指标
    for i = 1:config.n_defenders
        name = config.defender_types{i};
        results.(name).final_radi = monitors{i}.radi_history(end);
        results.(name).final_success_rate = mean(monitors{i}.success_rate_history(end-min(49,end-1):end));
        results.(name).final_detection_rate = mean(monitors{i}.detection_rate_history(end-min(49,end-1):end));
        results.(name).avg_damage = mean(monitors{i}.damage_history);
    end
end

%% 增强版可视化报告生成
function generateEnhancedReport(results, config, logger)
    % 生成增强版报告
    
    logger.info('生成多智能体对比报告...');
    
    % 创建报告目录
    report_dir = fullfile(pwd, 'reports', datestr(now, 'yyyymmdd_HHMMSS'));
    if ~exist(report_dir, 'dir')
        mkdir(report_dir);
    end
    
    % 1. 生成对比可视化
    generateComparativeVisualization(results, config, report_dir);
    
    % 2. 生成性能对比表
    generatePerformanceTable(results, config, report_dir);
    
    % 3. 生成详细文本报告
    generateDetailedTextReport(results, config, report_dir);
    
    % 4. 保存完整数据
    save(fullfile(report_dir, 'multi_agent_results.mat'), 'results');
    
    logger.info(sprintf('报告已保存到: %s', report_dir));
end

function generateComparativeVisualization(results, config, report_dir)
    % 生成对比可视化
    
    figure('Position', [50 50 1800 1200], 'Name', '多智能体性能对比');
    
    episodes = results.episodes;
    colors = {'b', 'r', 'g'};
    markers = {'o', 's', '^'};
    
    % 1. RADI对比
    subplot(3, 3, 1);
    hold on;
    for i = 1:config.n_defenders
        name = config.defender_types{i};
        plot(episodes, results.(name).radi_history, ...
             'Color', colors{i}, 'LineWidth', 2, ...
             'DisplayName', name);
    end
    xlabel('Episode');
    ylabel('RADI Score');
    title('RADI演化对比');
    legend('Location', 'best');
    grid on;
    
    % 2. 损害对比
    subplot(3, 3, 2);
    hold on;
    for i = 1:config.n_defenders
        name = config.defender_types{i};
        plot(episodes, results.(name).damage_history, ...
             'Color', colors{i}, 'LineWidth', 2, ...
             'DisplayName', name);
    end
    xlabel('Episode');
    ylabel('Average Damage');
    title('平均损害对比');
    legend('Location', 'best');
    grid on;
    
    % 3. 攻击成功率对比
    subplot(3, 3, 3);
    hold on;
    for i = 1:config.n_defenders
        name = config.defender_types{i};
        plot(episodes, results.(name).success_rate_history * 100, ...
             'Color', colors{i}, 'LineWidth', 2, ...
             'DisplayName', name);
    end
    xlabel('Episode');
    ylabel('Success Rate (%)');
    title('攻击成功率对比');
    legend('Location', 'best');
    grid on;
    ylim([0 100]);
    
    % 4. 检测率对比
    subplot(3, 3, 4);
    hold on;
    for i = 1:config.n_defenders
        name = config.defender_types{i};
        plot(episodes, results.(name).detection_rate_history * 100, ...
             'Color', colors{i}, 'LineWidth', 2, ...
             'DisplayName', name);
    end
    xlabel('Episode');
    ylabel('Detection Rate (%)');
    title('检测率对比');
    legend('Location', 'best');
    grid on;
    ylim([0 100]);
    
    % 5. 收敛性分析（RADI标准差）
    subplot(3, 3, 5);
    window = 50;
    hold on;
    for i = 1:config.n_defenders
        name = config.defender_types{i};
        if length(results.(name).radi_history) >= window
            radi_std = movstd(results.(name).radi_history, window);
            plot(window:length(radi_std), radi_std(window:end), ...
                 'Color', colors{i}, 'LineWidth', 2, ...
                 'DisplayName', name);
        end
    end
    xlabel('Episode');
    ylabel('RADI Std Dev');
    title('收敛性分析（滑动标准差）');
    legend('Location', 'best');
    grid on;
    
    % 6. 累积奖励对比
    subplot(3, 3, 6);
    hold on;
    for i = 1:config.n_defenders
        name = config.defender_types{i};
        cum_reward = cumsum(results.(name).rewards.defender);
        plot(episodes, cum_reward, ...
             'Color', colors{i}, 'LineWidth', 2, ...
             'DisplayName', name);
    end
    xlabel('Episode');
    ylabel('Cumulative Reward');
    title('累积奖励对比');
    legend('Location', 'best');
    grid on;
    
    % 7. 最终性能雷达图
    subplot(3, 3, 7);
    metrics = {'RADI', 'Detection Rate', '1-Success Rate', '1-Damage'};
    values = zeros(config.n_defenders, length(metrics));
    
    for i = 1:config.n_defenders
        name = config.defender_types{i};
        values(i, 1) = 1 - results.(name).final_radi;  % 越低越好，所以用1减
        values(i, 2) = results.(name).final_detection_rate;
        values(i, 3) = 1 - results.(name).final_success_rate;
        values(i, 4) = 1 - mean(results.(name).damage_history) / max([results.(config.defender_types{1}).damage_history, ...
                                                                       results.(config.defender_types{2}).damage_history, ...
                                                                       results.(config.defender_types{3}).damage_history]);
    end
    
    % 绘制雷达图
    angles = linspace(0, 2*pi, length(metrics)+1);
    
    for i = 1:config.n_defenders
        vals = [values(i, :), values(i, 1)];
        polarplot(angles, vals, [colors{i} '-o'], 'LineWidth', 2, ...
                 'MarkerFaceColor', colors{i}, 'DisplayName', config.defender_types{i});
        hold on;
    end
    
    % 设置雷达图属性
    ax = gca;
    ax.ThetaTick = angles(1:end-1) * 180/pi;
    ax.ThetaTickLabel = metrics;
    title('综合性能雷达图');
    legend('Location', 'best');
    
    % 8. 学习曲线对比（移动平均）
    subplot(3, 3, 8);
    ma_window = 20;
    hold on;
    for i = 1:config.n_defenders
        name = config.defender_types{i};
        if length(results.(name).radi_history) >= ma_window
            ma_radi = movmean(results.(name).radi_history, ma_window);
            plot(ma_window:length(ma_radi), ma_radi(ma_window:end), ...
                 'Color', colors{i}, 'LineWidth', 2, ...
                 'DisplayName', [name ' (MA)']);
        end
    end
    xlabel('Episode');
    ylabel('RADI (Moving Average)');
    title(sprintf('学习曲线对比（%d-Episode MA）', ma_window));
    legend('Location', 'best');
    grid on;
    
    % 9. 性能改善率
    subplot(3, 3, 9);
    improvement_data = zeros(config.n_defenders, 3);
    categories = {'RADI改善率', '成功率降低率', '检测率提升率'};
    
    for i = 1:config.n_defenders
        name = config.defender_types{i};
        % RADI改善率
        initial_radi = mean(results.(name).radi_history(1:min(10, end)));
        final_radi = results.(name).final_radi;
        improvement_data(i, 1) = (initial_radi - final_radi) / initial_radi * 100;
        
        % 成功率降低率
        initial_sr = mean(results.(name).success_rate_history(1:min(10, end)));
        final_sr = results.(name).final_success_rate;
        improvement_data(i, 2) = (initial_sr - final_sr) / initial_sr * 100;
        
        % 检测率提升率
        initial_dr = mean(results.(name).detection_rate_history(1:min(10, end)));
        final_dr = results.(name).final_detection_rate;
        improvement_data(i, 3) = (final_dr - initial_dr) / initial_dr * 100;
    end
    
    bar(improvement_data');
    set(gca, 'XTickLabel', categories);
    ylabel('改善率 (%)');
    title('性能改善率对比');
    legend(config.defender_types, 'Location', 'best');
    grid on;
    
    % 保存图形
    saveas(gcf, fullfile(report_dir, 'multi_agent_comparison.png'));
end

function generatePerformanceTable(results, config, report_dir)
    % 生成性能对比表
    
    % 创建表格数据
    metrics = {'最终RADI', '平均RADI', 'RADI标准差', ...
               '最终成功率(%)', '平均成功率(%)', ...
               '最终检测率(%)', '平均检测率(%)', ...
               '平均损害', '累积防御奖励', ...
               '收敛速度(Episodes)'};
    
    data = zeros(config.n_defenders, length(metrics));
    
    for i = 1:config.n_defenders
        name = config.defender_types{i};
        
        data(i, 1) = results.(name).final_radi;
        data(i, 2) = mean(results.(name).radi_history);
        data(i, 3) = std(results.(name).radi_history);
        data(i, 4) = results.(name).final_success_rate * 100;
        data(i, 5) = mean(results.(name).success_rate_history) * 100;
        data(i, 6) = results.(name).final_detection_rate * 100;
        data(i, 7) = mean(results.(name).detection_rate_history) * 100;
        data(i, 8) = results.(name).avg_damage;
        data(i, 9) = sum(results.(name).rewards.defender);
        
        % 计算收敛速度（RADI变化小于阈值的首个episode）
        radi_diff = abs(diff(results.(name).radi_history));
        conv_idx = find(radi_diff < 0.001, 1);
        data(i, 10) = ifelse(isempty(conv_idx), config.n_episodes, conv_idx);
    end
    
    % 创建并保存表格
    T = table(config.defender_types', data(:,1), data(:,2), data(:,3), ...
              data(:,4), data(:,5), data(:,6), data(:,7), ...
              data(:,8), data(:,9), data(:,10), ...
              'VariableNames', ['Algorithm', metrics]);
    
    writetable(T, fullfile(report_dir, 'performance_comparison.csv'));
    
    % 在命令窗口显示
    disp('性能对比表:');
    disp(T);
end

function generateDetailedTextReport(results, config, report_dir)
    % 生成详细文本报告
    
    report_file = fullfile(report_dir, 'detailed_report.txt');
    fid = fopen(report_file, 'w');
    
    fprintf(fid, '========================================\n');
    fprintf(fid, 'FSP-TCS 多智能体对比分析报告\n');
    fprintf(fid, '========================================\n');
    fprintf(fid, '生成时间: %s\n\n', datestr(now));
    
    % 1. 配置摘要
    fprintf(fid, '一、仿真配置\n');
    fprintf(fid, '----------------------------------------\n');
    fprintf(fid, '站点数量: %d\n', config.n_stations);
    fprintf(fid, '总Episodes: %d\n', config.n_episodes);
    fprintf(fid, '防御智能体: %s\n', strjoin(config.defender_types, ', '));
    fprintf(fid, '攻击智能体: %s\n\n', config.agents.attacker.type);
    
    % 2. 各智能体详细分析
    fprintf(fid, '二、各智能体性能分析\n');
    fprintf(fid, '----------------------------------------\n');
    
    for i = 1:config.n_defenders
        name = config.defender_types{i};
        fprintf(fid, '\n【%s】\n', name);
        fprintf(fid, '  最终RADI: %.4f\n', results.(name).final_radi);
        fprintf(fid, '  平均RADI: %.4f ± %.4f\n', ...
                mean(results.(name).radi_history), ...
                std(results.(name).radi_history));
        fprintf(fid, '  攻击成功率: %.2f%% (最终) / %.2f%% (平均)\n', ...
                results.(name).final_success_rate * 100, ...
                mean(results.(name).success_rate_history) * 100);
        fprintf(fid, '  检测率: %.2f%% (最终) / %.2f%% (平均)\n', ...
                results.(name).final_detection_rate * 100, ...
                mean(results.(name).detection_rate_history) * 100);
        fprintf(fid, '  平均损害: %.4f\n', results.(name).avg_damage);
        
        % 计算改善率
        initial_radi = mean(results.(name).radi_history(1:min(10, end)));
        improvement = (initial_radi - results.(name).final_radi) / initial_radi * 100;
        fprintf(fid, '  RADI改善率: %.2f%%\n', improvement);
    end
    
    % 3. 对比分析
    fprintf(fid, '\n三、对比分析\n');
    fprintf(fid, '----------------------------------------\n');
    
    % 找出最佳算法
    final_radis = zeros(config.n_defenders, 1);
    for i = 1:config.n_defenders
        name = config.defender_types{i};
        final_radis(i) = results.(name).final_radi;
    end
    
    [best_radi, best_idx] = min(final_radis);
    fprintf(fid, '最佳RADI性能: %s (%.4f)\n', ...
            config.defender_types{best_idx}, best_radi);
    
    % 检测率对比
    detection_rates = zeros(config.n_defenders, 1);
    for i = 1:config.n_defenders
        name = config.defender_types{i};
        detection_rates(i) = results.(name).final_detection_rate;
    end
    
    [best_dr, best_dr_idx] = max(detection_rates);
    fprintf(fid, '最佳检测率: %s (%.2f%%)\n', ...
            config.defender_types{best_dr_idx}, best_dr * 100);
    
    % 收敛速度分析
    fprintf(fid, '\n收敛性分析:\n');
    for i = 1:config.n_defenders
        name = config.defender_types{i};
        last_50_std = std(results.(name).radi_history(end-min(49,end-1):end));
        fprintf(fid, '  %s - 最后50轮标准差: %.4f\n', name, last_50_std);
    end
    
    % 4. 建议
    fprintf(fid, '\n四、优化建议\n');
    fprintf(fid, '----------------------------------------\n');
    
    % 基于性能自动生成建议
    if best_radi > 0.2
        fprintf(fid, '1. RADI指标仍有改善空间，建议增加训练Episodes\n');
    else
        fprintf(fid, '1. RADI指标表现良好，系统防御有效\n');
    end
    
    if min(detection_rates) < 0.8
        fprintf(fid, '2. 部分算法检测率偏低，建议优化检测策略参数\n');
    else
        fprintf(fid, '2. 检测率整体表现优秀\n');
    end
    
    % 算法推荐
    fprintf(fid, '\n基于综合性能，推荐使用: %s\n', config.defender_types{best_idx});
    
    fclose(fid);
end

%% 辅助函数
function value = ifelse(condition, true_value, false_value)
    if condition
        value = true_value;
    else
        value = false_value;
    end
end