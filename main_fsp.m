%% main_fsp.m - 优化的FSP仿真主程序
% =========================================================================
% 系统: 改进的FSP列控系统仿真平台
% 版本: v2.0 (优化版)
% 优化重点: 可维护性、可测试性、可扩展性、性能与安全
% =========================================================================

function main_improved_fsp_simulation()
    % 主仿真函数 - 统一入口点
    
    try
        % 初始化系统
        [config, logger] = initializeSystem();
        
        % 运行仿真
        results = runSimulation(config, logger);
        
        % 生成报告
        generateResults(results, config, logger);
        
        logger.info('改进版仿真成功完成');
        
    catch ME
        if exist('logger', 'var') && ~isempty(logger)
            logger.error(['仿真出错: ' ME.message]);
            logger.error(['错误位置: ' ME.stack(1).file ', 行号: ' num2str(ME.stack(1).line)]);
        else
            fprintf('严重错误: %s\n', ME.message);
        end
        rethrow(ME);
    end
end

%% =========================== 系统初始化 ===========================

function [config, logger] = initializeSystem()
    % 系统初始化 - 配置参数和日志系统
    
    % 初始化日志系统
    logger = setupLogger();
    logger.info('改进版FSP仿真开始');
    
    % 创建配置对象
    config = createConfiguration();
    
    % === 自动补全 RADI 字段，防止 PerformanceMonitor 报错 ===
    if ~isfield(config, 'radi')
        if isfield(config, 'resource_types')
            n = length(config.resource_types);
        else
            n = 5;
        end
        config.radi.optimal_allocation = ones(1, n) / n;
        config.radi.weight_computation = 0.2;
        config.radi.weight_bandwidth = 0.2;
        config.radi.weight_sensors = 0.2;
        config.radi.weight_scanning = 0.2;
        config.radi.weight_inspection = 0.2;
        config.radi.threshold_excellent = 0.05;
        config.radi.threshold_good = 0.1;
        config.radi.threshold_acceptable = 0.2;
    end
    
    % === 自动补全 training 字段，防止 PerformanceMonitor 报错 ===
    if ~isfield(config, 'training')
        config.training = struct();
    end
    if ~isfield(config.training, 'performance_target_radi')
        config.training.performance_target_radi = 0.1; % 你期望的目标阈值，可根据实际调整
    end
    
    % 验证配置
    validateConfiguration(config);
    
    % 设置随机种子以确保可重现性
    rng(config.random_seed);
    
    logger.info('系统初始化完成');
end

function logger = setupLogger()
    % 设置日志系统
    logger = struct();
    
    % 创建日志文件名
    timestamp = datestr(now, 'yyyymmdd_HHMMSS');
    log_dir = fullfile(pwd, 'logs');
    if ~exist(log_dir, 'dir')
        mkdir(log_dir);
    end
    log_file = fullfile(log_dir, ['simulation_' timestamp '.log']);
    
    % 打开日志文件
    logger.file_id = fopen(log_file, 'w');
    if logger.file_id == -1
        error('无法创建日志文件: %s', log_file);
    end
    
    % 日志函数
    logger.info = @(msg) writeLog(logger.file_id, 'INFO', msg);
    logger.warning = @(msg) writeLog(logger.file_id, 'WARNING', msg);
    logger.error = @(msg) writeLog(logger.file_id, 'ERROR', msg);
    logger.debug = @(msg) writeLog(logger.file_id, 'DEBUG', msg);
    
    % 初始化日志
    logger.info('日志系统初始化');
end

function writeLog(file_id, level, msg)
    % 写入日志条目
    timestamp = datestr(now, 'yyyy-mm-dd HH:MM:SS');
    log_entry = sprintf('[%s] [%s] %s\n', timestamp, level, msg); % 只保留合法转义字符
    fprintf(file_id, log_entry);
    fprintf(log_entry); % 同时输出到控制台
end

function config = createConfiguration()
    % 创建标准化配置对象
    
    config = struct();
    
    % === 基础环境参数 ===
    config.n_stations = 10;                % 站点数量
    config.n_components_per_station = 5;   % 每站点组件数
    config.total_resources = 100;          % 总防御资源
    config.random_seed = 42;               % 随机种子
    
    % === 仿真参数 ===
    config.n_episodes = 500;               % 仿真轮次
    config.max_steps_per_episode = 100;    % 每轮最大步数
    config.convergence_threshold = 0.01;   % 收敛阈值
    config.performance_check_interval = 50; % 性能检查间隔
    
    % === 智能体配置 ===
    config.agents = struct();
    
    % 攻击者智能体
    config.agents.attacker = struct();
    config.agents.attacker.type = 'QLearning';
    config.agents.attacker.learning_rate = 0.1;
    config.agents.attacker.discount_factor = 0.95;
    config.agents.attacker.epsilon = 0.3;
    config.agents.attacker.epsilon_decay = 0.995;
    config.agents.attacker.epsilon_min = 0.01;
    
    % 防御者智能体
    config.agents.defender = struct();
    config.agents.defender.type = 'FSP';
    config.agents.defender.learning_rate = 0.05;
    config.agents.defender.discount_factor = 0.95;
    config.agents.defender.fsp_alpha = 0.1;
    
    % === 环境特定参数 ===
    config.resource_types = {'cpu', 'memory', 'network', 'storage', 'security'};
    config.attack_types = {'dos', 'intrusion', 'malware', 'social', 'physical', 'cyber'};
    config.n_resource_types = length(config.resource_types);
    config.n_attack_types = length(config.attack_types);
    
    % === 奖励权重 ===
    config.reward_weights = struct();
    config.reward_weights.radi = 0.5;
    config.reward_weights.damage = 0.5;
    config.reward_weights.efficiency = 0.3;
    
    % === 输出配置 ===
    config.output = struct();
    config.output.save_models = true;
    config.output.generate_plots = true;
    config.output.save_data = true;
    config.output.results_dir = fullfile(pwd, 'results');
    
    % 确保输出目录存在
    if ~exist(config.output.results_dir, 'dir')
        mkdir(config.output.results_dir);
    end
end

function station_values = generateStationValues(n_stations)
    % 生成归一化的站点价值向量，确保长度匹配站点数量
    
    % 生成随机基础价值，避免零值
    base_values = 0.5 + 0.5 * rand(1, n_stations);
    
    % 设置部分站点为关键站点（更高价值）
    num_critical = max(1, round(n_stations * 0.3));
    critical_indices = randperm(n_stations, num_critical);
    base_values(critical_indices) = base_values(critical_indices) * 1.8;
    
    % 归一化到总和为1
    station_values = base_values / sum(base_values);
    
    % 验证输出维度
    assert(length(station_values) == n_stations, ...
        '站点价值向量长度(%d)与站点数量(%d)不匹配', length(station_values), n_stations);
    assert(abs(sum(station_values) - 1) < 1e-10, ...
        '站点价值向量未正确归一化，总和为%.6f', sum(station_values));
end

function validateConfiguration(config)
    % 验证配置参数的有效性 - 增强维度检查
    
    assert(config.n_stations > 0, 'n_stations 必须大于 0');
    assert(config.n_components_per_station > 0, 'n_components_per_station 必须大于 0');
    assert(config.total_resources > 0, 'total_resources 必须大于 0');
    assert(config.n_episodes > 0, 'n_episodes 必须大于 0');
    assert(config.max_steps_per_episode > 0, 'max_steps_per_episode 必须大于 0');
    
    % 验证学习率范围
    assert(config.agents.attacker.learning_rate > 0 && config.agents.attacker.learning_rate <= 1, ...
           '攻击者学习率必须在 (0,1] 范围内');
    assert(config.agents.defender.learning_rate > 0 && config.agents.defender.learning_rate <= 1, ...
           '防御者学习率必须在 (0,1] 范围内');
    
    % 验证epsilon参数
    assert(config.agents.attacker.epsilon >= 0 && config.agents.attacker.epsilon <= 1, ...
           '攻击者epsilon必须在 [0,1] 范围内');
    assert(config.agents.attacker.epsilon_min >= 0 && config.agents.attacker.epsilon_min <= 1, ...
           '攻击者epsilon_min必须在 [0,1] 范围内');
    assert(config.agents.attacker.epsilon_decay > 0 && config.agents.attacker.epsilon_decay <= 1, ...
           '攻击者epsilon_decay必须在 (0,1] 范围内');
    
    % === 关键修复：验证维度一致性 ===
    if isfield(config, 'station_values')
        actual_length = length(config.station_values);
        expected_length = config.n_stations;
        assert(actual_length == expected_length, ...
            '站点价值向量长度(%d)与站点数量(%d)不匹配。请检查配置。', ...
            actual_length, expected_length);
        
        % 验证归一化
        sum_values = sum(config.station_values);
        assert(abs(sum_values - 1) < 1e-6, ...
            '站点价值向量未正确归一化，总和为%.6f，应为1.0', sum_values);
    end
    
    % 验证资源和攻击类型数量一致性
    if isfield(config, 'resource_types') && isfield(config, 'n_resource_types')
        assert(length(config.resource_types) == config.n_resource_types, ...
            'resource_types长度与n_resource_types不匹配');
    end
    
    if isfield(config, 'attack_types') && isfield(config, 'n_attack_types')
        assert(length(config.attack_types) == config.n_attack_types, ...
            'attack_types长度与n_attack_types不匹配');
    end
end

%% =========================== 仿真执行 ===========================

function results = runSimulation(config, logger)
    % 运行主仿真循环
    
    logger.info('开始创建环境和智能体');
    
    % 创建环境
    environment = createEnvironment(config);
    
    % 创建智能体
    agents = createAgents(config, environment);
    
    % 创建性能监控器
    monitor = createPerformanceMonitor(config);
    
    % 初始化结果存储
    results = initializeResults(config);
    
    logger.info('开始仿真循环');
    
    % 主仿真循环
    for episode = 1:config.n_episodes
        episode_start_time = tic;
        
        % 运行单个episode
        episode_data = runEpisode(environment, agents, monitor, config, episode);
        
        % 存储episode结果
        results = updateResults(results, episode_data, episode);
        
        % 更新智能体
        updateAgents(agents, episode_data);
        
        % 性能检查
        if mod(episode, config.performance_check_interval) == 0
            checkPerformance(results, monitor, episode, logger);
        end
        
        episode_time = toc(episode_start_time);
        logger.info(sprintf('迭代 %d 完成，用时 %.2f秒', episode, episode_time));
        
        % 检查收敛
        if episode > 100 && checkConvergence(results, config.convergence_threshold)
            logger.info(sprintf('仿真在第 %d 轮收敛', episode));
            break;
        end
    end
    
    % 最终化结果
    results = finalizeResults(results, agents, monitor);
    
    logger.info('仿真循环完成');
end

function environment = createEnvironment(config)
    % 自动修正n_components_per_station长度
    if isfield(config, 'n_stations') && isfield(config, 'n_components_per_station')
        n = config.n_stations;
        c = config.n_components_per_station;
        if length(c) < n
            config.n_components_per_station = [c, ones(1, n-length(c))*3];
        elseif length(c) > n
            config.n_components_per_station = c(1:n);
        end
    end
    environment = TCSEnvironment(config);
end

function agents = createAgents(config, environment)
    % 创建智能体 - 使用工厂模式
    
    agents = struct();
    
    % 获取状态和动作空间维度
    state_dim = environment.state_dim;
    action_dim = environment.action_dim;
    
    % 创建攻击者智能体
    attacker_config = config.agents.attacker;
    agents.attacker = createAgent('attacker', attacker_config.type, attacker_config, ...
                                  state_dim, action_dim);
    
    % 创建防御者智能体  
    defender_config = config.agents.defender;
    agents.defender = createAgent('defender', defender_config.type, defender_config, ...
                                  state_dim, action_dim);
end

function agent = createAgent(name, type, agent_config, state_dim, action_dim)
    % 智能体工厂函数
    
    switch type
        case 'QLearning'
            agent = QLearningAgent(name, type, agent_config, state_dim, action_dim);
        case 'SARSA'
            agent = SARSAAgent(name, type, agent_config, state_dim, action_dim);
        case 'DoubleQLearning'
            agent = DoubleQLearningAgent(name, type, agent_config, state_dim, action_dim);
        case 'FSP'
            agent = FSPAgent(name, type, agent_config, state_dim, action_dim);
        otherwise
            error('未知的智能体类型: %s', type);
    end
end

function monitor = createPerformanceMonitor(config)
    monitor = PerformanceMonitor(config.n_episodes, config.n_stations, config);
end

function results = initializeResults(config)
    % 初始化结果存储结构
    
    results = struct();
    results.episodes = [];
    results.rewards = struct('attacker', [], 'defender', []);
    results.metrics = struct();
    results.convergence_data = [];
    results.performance_data = [];
    results.config = config;
end

function episode_data = runEpisode(environment, agents, monitor, config, episode)
    % 运行单个episode
    
    % 重置环境
    state = environment.reset();
    
    % 初始化episode数据
    episode_data = struct();
    episode_data.states = [];
    episode_data.actions = struct('attacker', [], 'defender', []);
    episode_data.rewards = struct('attacker', [], 'defender', []);
    episode_data.episode_reward = struct('attacker', 0, 'defender', 0);
    
    % episode循环
    for step = 1:config.max_steps_per_episode
        % 智能体选择动作
        attacker_action = agents.attacker.selectAction(state);
        defender_action = agents.defender.selectAction(state);
        
        % 环境执行动作
        [next_state, reward_def, reward_att, info] = environment.step(attacker_action, defender_action);
        
        % 存储转换数据
        episode_data.states = [episode_data.states; state];
        episode_data.actions.attacker = [episode_data.actions.attacker; attacker_action];
        episode_data.actions.defender = [episode_data.actions.defender; defender_action];
        episode_data.rewards.attacker = [episode_data.rewards.attacker; reward_att];
        episode_data.rewards.defender = [episode_data.rewards.defender; reward_def];
        
        % 累计奖励
        episode_data.episode_reward.attacker = episode_data.episode_reward.attacker + reward_att;
        episode_data.episode_reward.defender = episode_data.episode_reward.defender + reward_def;
        
        % 更新智能体
        agents.attacker.update(state, attacker_action, reward_att, next_state, []);
        agents.defender.update(state, defender_action, reward_def, next_state, []);
        
        % 更新监控器
        monitor.update(episode, info);
        
        % 准备下一步
        state = next_state;
        
        if isfield(info, 'done') && info.done
            break;
        end
    end
    
    % 完成episode
    episode_data.final_state = state;
    episode_data.total_steps = step;
end

function results = updateResults(results, episode_data, episode)
    % 更新结果数据
    
    results.episodes = [results.episodes; episode];
    results.rewards.attacker = [results.rewards.attacker; episode_data.episode_reward.attacker];
    results.rewards.defender = [results.rewards.defender; episode_data.episode_reward.defender];
end

function updateAgents(agents, episode_data)
    % 更新智能体参数
    
    % 更新探索率等参数
    if ismethod(agents.attacker, 'updateEpsilon')
        agents.attacker.updateEpsilon();
    end
    
    if ismethod(agents.defender, 'updateParameters')
        agents.defender.updateParameters();
    end
end

function checkPerformance(results, monitor, episode, logger)
    % 检查性能指标
    
    if episode > 50
        recent_rewards_att = results.rewards.attacker(end-49:end);
        recent_rewards_def = results.rewards.defender(end-49:end);
        
        avg_reward_att = mean(recent_rewards_att);
        avg_reward_def = mean(recent_rewards_def);
        
        logger.info(sprintf('第 %d 轮 - 攻击者平均奖励: %.3f, 防御者平均奖励: %.3f', ...
                          episode, avg_reward_att, avg_reward_def));
    end
end

function converged = checkConvergence(results, threshold)
    % 检查收敛性
    
    converged = false;
    
    if length(results.rewards.attacker) > 100
        recent_window = 50;
        current_rewards = results.rewards.attacker(end-recent_window+1:end);
        previous_rewards = results.rewards.attacker(end-2*recent_window+1:end-recent_window);
        
        current_mean = mean(current_rewards);
        previous_mean = mean(previous_rewards);
        
        if abs(current_mean - previous_mean) < threshold
            converged = true;
        end
    end
end

function results = finalizeResults(results, agents, monitor)
    % 最终化结果数据
    
    results.final_performance = monitor.generateSummary();
    results.agent_statistics = struct();
    results.agent_statistics.attacker = agents.attacker.getStatistics();
    results.agent_statistics.defender = agents.defender.getStatistics();
end

%% =========================== 结果生成 ===========================

function generateResults(results, config, logger)
    % 生成仿真结果和报告
    
    logger.info('开始生成结果报告');
    
    % 保存数据
    if config.output.save_data
        saveResults(results, config);
    end
    
    % 生成图表
    if config.output.generate_plots
        generatePlots(results, config);
    end
    
    % 保存模型
    if config.output.save_models
        saveModels(results, config);
    end
    
    % 生成文本报告
    generateTextReport(results, config);
    
    logger.info('结果报告生成完成');
end

function saveResults(results, config)
    % 保存仿真结果数据
    
    timestamp = datestr(now, 'yyyymmdd_HHMMSS');
    filename = fullfile(config.output.results_dir, ['simulation_results_' timestamp '.mat']);
    
    save(filename, 'results', 'config');
end

function generatePlots(results, config)
    % 生成可视化图表
    
    % 奖励曲线
    figure('Name', '仿真结果', 'Position', [100, 100, 1200, 800]);
    
    subplot(2, 2, 1);
    plot(results.episodes, results.rewards.attacker, 'r-', 'LineWidth', 1.5);
    hold on;
    plot(results.episodes, results.rewards.defender, 'b-', 'LineWidth', 1.5);
    xlabel('Episode');
    ylabel('Reward');
    title('智能体奖励曲线');
    legend('攻击者', '防御者');
    grid on;
    
    % 移动平均奖励
    subplot(2, 2, 2);
    window_size = 50;
    if length(results.rewards.attacker) >= window_size
        moving_avg_att = movmean(results.rewards.attacker, window_size);
        moving_avg_def = movmean(results.rewards.defender, window_size);
        plot(results.episodes, moving_avg_att, 'r-', 'LineWidth', 2);
        hold on;
        plot(results.episodes, moving_avg_def, 'b-', 'LineWidth', 2);
    end
    xlabel('Episode');
    ylabel('Moving Average Reward');
    title('移动平均奖励');
    legend('攻击者', '防御者');
    grid on;
    
    % 奖励分布
    subplot(2, 2, 3);
    histogram(results.rewards.attacker, 30, 'FaceAlpha', 0.7);
    hold on;
    histogram(results.rewards.defender, 30, 'FaceAlpha', 0.7);
    xlabel('Reward');
    ylabel('Frequency');
    title('奖励分布');
    legend('攻击者', '防御者');
    
    % 性能指标
    subplot(2, 2, 4);
    if isfield(results, 'final_performance') && isfield(results.final_performance, 'radi_history')
        plot(results.final_performance.radi_history);
        xlabel('Episode');
        ylabel('RADI Score');
        title('RADI性能指标');
        grid on;
    end
    
    % 保存图像
    timestamp = datestr(now, 'yyyymmdd_HHMMSS');
    saveas(gcf, fullfile(config.output.results_dir, ['simulation_plots_' timestamp '.png']));
end

function saveModels(results, config)
    % 保存训练好的模型
    
    timestamp = datestr(now, 'yyyymmdd_HHMMSS');
    
    % 保存智能体统计信息
    agent_stats = results.agent_statistics;
    filename = fullfile(config.output.results_dir, ['agent_models_' timestamp '.mat']);
    save(filename, 'agent_stats');
end

function generateTextReport(results, config)
    % 生成文本格式的详细报告
    
    timestamp = datestr(now, 'yyyymmdd_HHMMSS');
    report_file = fullfile(config.output.results_dir, ['simulation_report_' timestamp '.txt']);
    
    fid = fopen(report_file, 'w');
    if fid == -1
        warning('无法创建报告文件');
        return;
    end
    
    fprintf(fid, '======================================\n');
    fprintf(fid, '       FSP 仿真系统结果报告\n');
    fprintf(fid, '======================================\n\n');
    
    fprintf(fid, '仿真时间: %s\n', timestamp);
    fprintf(fid, '总轮次: %d\n', length(results.episodes));
    fprintf(fid, '配置参数:\n');
    fprintf(fid, '  - 站点数量: %d\n', config.n_stations);
    fprintf(fid, '  - 总资源: %d\n', config.total_resources);
    fprintf(fid, '  - 随机种子: %d\n\n', config.random_seed);
    
    % 性能统计
    fprintf(fid, '性能统计:\n');
    fprintf(fid, '  攻击者:\n');
    fprintf(fid, '    - 平均奖励: %.4f\n', mean(results.rewards.attacker));
    fprintf(fid, '    - 标准差: %.4f\n', std(results.rewards.attacker));
    fprintf(fid, '    - 最大奖励: %.4f\n', max(results.rewards.attacker));
    fprintf(fid, '    - 最小奖励: %.4f\n', min(results.rewards.attacker));
    
    fprintf(fid, '  防御者:\n');
    fprintf(fid, '    - 平均奖励: %.4f\n', mean(results.rewards.defender));
    fprintf(fid, '    - 标准差: %.4f\n', std(results.rewards.defender));
    fprintf(fid, '    - 最大奖励: %.4f\n', max(results.rewards.defender));
    fprintf(fid, '    - 最小奖励: %.4f\n', min(results.rewards.defender));
    
    fprintf(fid, '\n仿真完成。\n');
    fclose(fid);
end

%% =========================== 辅助函数 ===========================

function success = ismethod(obj, method_name)
    % 检查对象是否有指定方法
    try
        methods_list = methods(obj);
        success = any(strcmp(methods_list, method_name));
    catch
        success = false;
    end
end

%% =========================== 程序入口 ===========================

% 如果直接运行此文件，则执行主函数
if ~isdeployed
    main_improved_fsp_simulation();
end

% =========================== FSP博弈结果输出与可视化 ===========================
function displayFSPResults(env, episode_radi, episode_damage, episode_success_rate, config, n_episodes)
    % FSP博弈结果输出与可视化函数
    % 输入参数：
    %   env - TCSEnvironment对象
    %   episode_radi - RADI历史记录
    %   episode_damage - 损害历史记录
    %   episode_success_rate - 攻击成功率历史记录
    %   config - 配置参数
    %   n_episodes - 总回合数
    %% 1. 计算最终策略
    final_defense_strategy = mean(env.deployment_history, 1);
    final_defense_strategy = final_defense_strategy / sum(final_defense_strategy);
    attack_frequency = zeros(1, config.n_stations);
    for i = 1:length(env.attack_history)
        attack_frequency(env.attack_history(i)) = attack_frequency(env.attack_history(i)) + 1;
    end
    final_attack_strategy = attack_frequency / sum(attack_frequency);
    perceived_attack_strategy = env.attacker_avg_strategy;
    %% 2. 计算平均指标
    last_n = min(100, n_episodes);
    avg_radi = mean(episode_radi(end-last_n+1:end));
    avg_damage = mean(episode_damage(end-last_n+1:end));
    avg_success_rate = mean(episode_success_rate(end-last_n+1:end));
    %% 3. 输出结果
    fprintf('\n\n========================================\n');
    fprintf('        FSP博弈仿真结果汇总\n');
    fprintf('========================================\n\n');
    fprintf('【防守策略】\n');
    fprintf('站点编号:     ');
    for i = 1:config.n_stations
        fprintf('%8d ', i);
    end
    fprintf('\n资源分配(%%):  ');
    for i = 1:config.n_stations
        fprintf('%8.2f ', final_defense_strategy(i) * 100);
    end
    fprintf('\n\n');
    fprintf('【攻击策略】\n');
    fprintf('实际攻击概率: ');
    for i = 1:config.n_stations
        fprintf('%8.2f ', final_attack_strategy(i) * 100);
    end
    fprintf('\n感知攻击概率: ');
    for i = 1:config.n_stations
        fprintf('%8.2f ', perceived_attack_strategy(i) * 100);
    end
    fprintf('\n\n');
    fprintf('【关键性能指标】(最后%d轮平均)\n', last_n);
    fprintf('RADI:           %.4f\n', avg_radi);
    fprintf('Damage:         %.4f\n', avg_damage);
    fprintf('Success Rate:   %.2f%%\n', avg_success_rate * 100);
    fprintf('\n');
    fprintf('【站点价值分析】\n');
    fprintf('站点价值:     ');
    for i = 1:config.n_stations
        fprintf('%8.3f ', env.station_values(i));
    end
    fprintf('\n\n');
    strategy_similarity = 1 - norm(final_attack_strategy - perceived_attack_strategy);
    fprintf('【策略分析】\n');
    fprintf('攻击策略相似度: %.2f%%\n', strategy_similarity * 100);
    convergence_score = 1 - std(episode_radi(end-last_n+1:end)) / mean(episode_radi(end-last_n+1:end));
    fprintf('收敛性得分:     %.2f%%\n', convergence_score * 100);
    recent_radi = episode_radi(end-last_n+1:end);
    radi_trend = analyzeTrend(recent_radi);
    fprintf('RADI趋势:       %s\n', radi_trend);
    fprintf('\n========================================\n\n');
    %% 4. 绘制可视化图（基于EnhancedReportGenerator风格）
    figure('Name', 'FSP博弈综合分析报告', 'Position', [50, 50, 1400, 900]);
    set(gcf, 'Color', 'white');
    subplot(2,3,1);
    plotRADIEvolution(episode_radi, config);
    subplot(2,3,2);
    plotDamageEvolution(episode_damage);
    subplot(2,3,3);
    plotSuccessRateEvolution(episode_success_rate);
    subplot(2,3,4);
    plotStrategyComparison(final_attack_strategy, perceived_attack_strategy, final_defense_strategy, config);
    subplot(2,3,5);
    plotStationAnalysis(env.station_values, final_attack_strategy, final_defense_strategy, config);
    subplot(2,3,6);
    plotPerformanceRadar(avg_radi, avg_damage, avg_success_rate, strategy_similarity, convergence_score);
    sgtitle('基于FSP的网络攻防博弈分析报告', 'FontSize', 16, 'FontWeight', 'bold');
    filename = sprintf('fsp_report_%s.png', datestr(now, 'yyyymmdd_HHMMSS'));
    saveas(gcf, filename);
    fprintf('\n可视化报告已保存至: %s\n', filename);
end

function plotRADIEvolution(radi, config)
    window_size = min(50, max(5, floor(length(radi)/10)));
    radi_smooth = movmean(radi, window_size);
    plot(1:length(radi), radi, 'Color', [0.2 0.4 0.8 0.3], 'LineWidth', 0.5);
    hold on;
    plot(1:length(radi_smooth), radi_smooth, 'b-', 'LineWidth', 2.5);
    x = 1:length(radi);
    p = polyfit(x, radi, 1);
    trend = polyval(p, x);
    plot(x, trend, '--', 'Color', [0.1 0.2 0.4 0.8], 'LineWidth', 2);
    yline(config.radi.threshold_excellent, ':', '优秀', 'Color', [0 0.6 0], 'LineWidth', 1.5);
    yline(config.radi.threshold_good, ':', '良好', 'Color', [0.7 0.7 0], 'LineWidth', 1.5);
    yline(config.radi.threshold_acceptable, ':', '可接受', 'Color', [1 0.5 0], 'LineWidth', 1.5);
    final_radi = mean(radi(max(1, end-99):end));
    text(length(radi)*0.7, min(radi)+0.05, sprintf('最终RADI: %.4f', final_radi), ...
        'FontSize', 12, 'FontWeight', 'bold', 'Color', 'b');
    xlabel('迭代次数', 'FontSize', 12);
    ylabel('RADI', 'FontSize', 12);
    title('RADI演化趋势', 'FontSize', 14, 'FontWeight', 'bold');
    legend({'原始数据', '平滑曲线', '趋势线'}, 'Location', 'best');
    grid on;
    ylim([0, max(0.5, max(radi)*1.1)]);
end

function plotDamageEvolution(damage)
    window_size = min(50, max(5, floor(length(damage)/10)));
    damage_smooth = movmean(damage, window_size);
    x = 1:length(damage);
    fill([x, fliplr(x)], [damage_smooth, zeros(size(damage_smooth))], ...
        [0.8 0.2 0.2], 'FaceAlpha', 0.3, 'EdgeColor', 'none');
    hold on;
    plot(x, damage_smooth, 'r-', 'LineWidth', 2);
    avg_damage = mean(damage);
    plot([1, length(damage)], [avg_damage, avg_damage], 'r--', 'LineWidth', 1.5);
    text(length(damage)*0.5, avg_damage+0.02, sprintf('平均: %.3f', avg_damage), ...
        'FontSize', 11, 'HorizontalAlignment', 'center');
    xlabel('迭代次数', 'FontSize', 12);
    ylabel('损害值', 'FontSize', 12);
    title('攻击损害(Damage)趋势', 'FontSize', 14, 'FontWeight', 'bold');
    grid on;
    ylim([0, max(damage)*1.2]);
end

function plotSuccessRateEvolution(success_rate)
    window_size = min(50, max(5, floor(length(success_rate)/10)));
    success_smooth = movmean(success_rate, window_size);
    plot(1:length(success_rate), success_rate * 100, 'Color', [0.8 0.2 0.2 0.3], 'LineWidth', 0.5);
    hold on;
    plot(1:length(success_smooth), success_smooth * 100, 'r-', 'LineWidth', 2.5);
    avg_success = mean(success_rate) * 100;
    plot([1, length(success_rate)], [avg_success, avg_success], 'r--', 'LineWidth', 1.5);
    text(length(success_rate)*0.5, avg_success+2, sprintf('平均: %.1f%%', avg_success), ...
        'FontSize', 11, 'HorizontalAlignment', 'center');
    xlabel('迭代次数', 'FontSize', 12);
    ylabel('成功率 (%)', 'FontSize', 12);
    title('攻击成功率(Success Rate)演化', 'FontSize', 14, 'FontWeight', 'bold');
    grid on;
    ylim([0, 100]);
end

function plotStrategyComparison(attack_actual, attack_perceived, defense, config)
    strategies = [attack_actual; attack_perceived; defense];
    b = bar(strategies', 'grouped');
    b(1).FaceColor = [0.8 0.2 0.2];
    b(2).FaceColor = [0.8 0.5 0.2];
    b(3).FaceColor = [0.2 0.4 0.8];
    for i = 1:3
        for j = 1:config.n_stations
            text(j + (i-2)*0.28, strategies(i,j) + 0.01, ...
                sprintf('%.1f%%', strategies(i,j)*100), ...
                'HorizontalAlignment', 'center', 'FontSize', 9);
        end
    end
    xlabel('站点', 'FontSize', 12);
    ylabel('概率/资源比例', 'FontSize', 12);
    title('攻防策略对比', 'FontSize', 14, 'FontWeight', 'bold');
    legend({'实际攻击', '感知攻击', '防御部署'}, 'Location', 'best');
    set(gca, 'XTick', 1:config.n_stations);
    grid on;
    ylim([0, max(strategies(:))*1.3]);
end

function plotStationAnalysis(values, attack_freq, defense_alloc, config)
    yyaxis left;
    bar(1:config.n_stations, values, 'FaceColor', [0.7, 0.7, 0.7], 'EdgeColor', 'k');
    ylabel('站点价值', 'FontSize', 12);
    ylim([0, max(values) * 1.2]);
    yyaxis right;
    plot(1:config.n_stations, attack_freq, 'ro-', 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', 'r');
    hold on;
    plot(1:config.n_stations, defense_alloc, 'b^-', 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', 'b');
    ylabel('攻击频率/防御分配', 'FontSize', 12);
    ylim([0, max([attack_freq, defense_alloc]) * 1.2]);
    xlabel('站点', 'FontSize', 12);
    title('站点价值与攻防策略分析', 'FontSize', 14, 'FontWeight', 'bold');
    legend({'站点价值', '攻击频率', '防御分配'}, 'Location', 'best');
    set(gca, 'XTick', 1:config.n_stations);
    grid on;
end

function plotPerformanceRadar(radi, damage, success_rate, similarity, convergence)
    categories = {'RADI\n(低)', 'Damage\n(低)', 'Success Rate\n(低)', '策略相似度\n(高)', '收敛性\n(高)'};
    data = [1-radi, 1-damage, 1-success_rate, similarity, convergence];
    data = max(0, min(1, data));
    angles = linspace(0, 2*pi, length(categories)+1);
    data_plot = [data, data(1)];
    fill(data_plot .* cos(angles), data_plot .* sin(angles), [0.2 0.4 0.8], 'FaceAlpha', 0.3);
    hold on;
    plot(data_plot .* cos(angles), data_plot .* sin(angles), 'b-', 'LineWidth', 2);
    plot(data_plot .* cos(angles), data_plot .* sin(angles), 'bo', 'MarkerSize', 8, 'MarkerFaceColor', 'b');
    for r = 0.2:0.2:1
        plot(r*cos(angles), r*sin(angles), 'k:', 'LineWidth', 0.5);
    end
    for i = 1:length(categories)
        plot([0, cos(angles(i))], [0, sin(angles(i))], 'k-', 'LineWidth', 0.5);
    end
    for i = 1:length(categories)
        text(1.2*cos(angles(i)), 1.2*sin(angles(i)), categories{i}, ...
            'HorizontalAlignment', 'center', 'FontSize', 11);
    end
    for i = 1:length(data)
        text(data(i)*cos(angles(i))*1.1, data(i)*sin(angles(i))*1.1, ...
            sprintf('%.2f', data(i)), 'FontSize', 10, 'FontWeight', 'bold');
    end
    title('综合性能指标', 'FontSize', 14, 'FontWeight', 'bold');
    axis equal;
    axis([-1.5 1.5 -1.5 1.5]);
    axis off;
end

function trend = analyzeTrend(data)
    n = length(data);
    x = (1:n)';
    p = polyfit(x, data(:), 1);
    if p(1) < -0.0001
        trend = '下降↓';
    elseif p(1) > 0.0001
        trend = '上升↑';
    else
        trend = '稳定→';
    end
end