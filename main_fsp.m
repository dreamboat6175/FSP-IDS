%% main_fsp_simulation_optimized.m - 优化后的主仿真程序
% =========================================================================
% 描述: 使用集中化配置管理的FSP-TCS主仿真程序
% 版本: v2.0 - 优化版，展示参数集中管理
% =========================================================================

function main_fsp_simulation_optimized()
    % FSP-TCS主仿真程序
    
    try
        %% === 1. 系统初始化 ===
        fprintf('\n=== FSP-TCS 智能防御系统仿真 v2.0 ===\n');
        fprintf('正在初始化系统...\n\n');
        
        % 加载配置 - 所有参数都在ConfigManager中管理
        config = ConfigManager.loadConfig();  % 使用默认配置
        % config = ConfigManager.getOptimizedConfig();  % 或使用优化配置
        % config = ConfigManager.getTestConfig();       % 或使用测试配置
        
        % 显示配置摘要
        ConfigManager.displayConfigSummary(config);
        
        % 初始化日志系统
        Logger.initialize(config.output.log_file);
        Logger.info('FSP-TCS仿真开始');
        
        % 设置随机种子以确保可重现性
        if isfield(config, 'random_seed')
            rng(config.random_seed);
            Logger.info(sprintf('随机种子设置为: %d', config.random_seed));
        end
        
        %% === 2. 环境和智能体初始化 ===
        fprintf('正在创建环境和智能体...\n');
        
        % 创建环境 - 使用配置中的参数
        env = TCSEnvironment(config);
        Logger.info('TCS环境创建完成');
        
        % 创建智能体 - AgentFactory使用配置参数
        defender_agents = AgentFactory.createDefenderAgents(config, env);
        attacker_agent = AgentFactory.createAttackerAgent(config, env);
        Logger.info(sprintf('智能体创建完成: %d个防御者, 1个攻击者', length(defender_agents)));
        
        % 创建性能监控器 - 使用配置中的监控参数
        monitor = PerformanceMonitor(config.n_iterations, length(defender_agents), config);
        Logger.info('性能监控器初始化完成');
        
        %% === 3. FSP仿真主循环 ===
        fprintf('开始FSP仿真训练...\n');
        
        % 初始化结果存储
        results = initializeResults(config, length(defender_agents));
        
        % 仿真主循环
        for iteration = 1:config.n_iterations
            tic;
            
            % 运行一轮episodes
            episode_results = FSPSimulator.runEpisodes(env, defender_agents, attacker_agent, config);
            
            % 记录结果
            results = recordIterationResults(results, episode_results, iteration);
            
            % 更新性能监控
            updatePerformanceMonitor(monitor, iteration, episode_results, config);
            
            % 动态更新学习参数
            if mod(iteration, config.performance.param_update_interval) == 0
                config = ConfigManager.updateLearningParameters(config, iteration);
                updateAgentParameters(defender_agents, attacker_agent, config);
            end
            
            % 显示进度和保存检查点
            iteration_time = toc;
            handleIterationOutput(iteration, config, iteration_time, episode_results);
            
            if mod(iteration, config.output.checkpoint_interval) == 0 && config.output.save_checkpoints
                saveCheckpoint(defender_agents, attacker_agent, results, iteration, config);
            end
        end
        
        %% === 4. 结果分析和可视化 ===
        fprintf('\n仿真完成，正在生成分析报告...\n');
        
        % 保存最终结果
        DataManager.saveResults(results, config);
        Logger.info('仿真结果已保存');
        
        % 生成可视化报告 - 使用EnhancedVisualization的集中配置
        if config.output.visualization
            visualization = EnhancedVisualization(results, config, env);
            visualization.generateCompleteReport();
            Logger.info('可视化报告生成完成');
        end
        
        % 生成文本报告
        if config.output.generate_report
            ReportGenerator.generateTextReport(results, config, monitor);
            Logger.info('文本报告生成完成');
        end
        
        %% === 5. 系统清理 ===
        fprintf('\n=== 仿真完成 ===\n');
        printFinalSummary(results, config);
        
        Logger.info('FSP-TCS仿真成功完成');
        Logger.close();
        
    catch ME
        % 错误处理
        fprintf('\n❌ 仿真过程中发生错误:\n');
        fprintf('错误信息: %s\n', ME.message);
        fprintf('错误位置: %s, 行号: %d\n', ME.stack(1).file, ME.stack(1).line);
        
        if exist('Logger', 'class')
            Logger.error(sprintf('仿真出错: %s', ME.message));
            Logger.error(sprintf('错误位置: %s, 行号: %d', ME.stack(1).file, ME.stack(1).line));
            Logger.close();
        end
        
        rethrow(ME);
    end
end

%% === 辅助函数 ===

function results = initializeResults(config, n_agents)
    % 初始化结果存储结构
    
    results = struct();
    results.config = config;
    results.n_iterations = config.n_iterations;
    results.n_agents = n_agents;
    results.timestamp = datestr(now);
    
    % 性能指标历史
    results.radi_history = zeros(config.n_iterations, n_agents);
    results.success_rate_history = zeros(config.n_iterations, 1);
    results.detection_rate_history = zeros(config.n_iterations, 1);
    results.resource_efficiency = zeros(config.n_iterations, n_agents);
    results.allocation_balance = zeros(config.n_iterations, n_agents);
    
    % 策略历史
    results.attacker_strategy_history = zeros(config.n_iterations, config.n_stations);
    results.defender_strategy_history = cell(n_agents, 1);
    for i = 1:n_agents
        results.defender_strategy_history{i} = zeros(config.n_iterations, config.n_stations);
    end
    
    % 奖励历史
    results.defender_rewards = zeros(config.n_iterations, n_agents);
    results.attacker_rewards = zeros(config.n_iterations, 1);
    
    % 学习参数历史
    results.epsilon_history = zeros(config.n_iterations, 1);
    results.learning_rate_history = zeros(config.n_iterations, 1);
    results.temperature_history = zeros(config.n_iterations, 1);
end

function results = recordIterationResults(results, episode_results, iteration)
    % 记录单次迭代的结果
    
    % 性能指标
    results.radi_history(iteration, :) = episode_results.avg_radi;
    results.success_rate_history(iteration) = mean([episode_results.attack_info{:}]);
    results.resource_efficiency(iteration, :) = episode_results.avg_efficiency;
    results.allocation_balance(iteration, :) = episode_results.avg_balance;
    
    % 奖励信息
    results.defender_rewards(iteration, :) = episode_results.avg_defender_reward;
    results.attacker_rewards(iteration) = episode_results.avg_attacker_reward;
    
    % 策略信息
    if isfield(episode_results, 'attacker_strategy')
        results.attacker_strategy_history(iteration, :) = episode_results.attacker_strategy;
    end
    
    if isfield(episode_results, 'defender_strategies')
        for i = 1:length(episode_results.defender_strategies)
            results.defender_strategy_history{i}(iteration, :) = episode_results.defender_strategies{i};
        end
    end
end

function updatePerformanceMonitor(monitor, iteration, episode_results, config)
    % 更新性能监控器
    
    % 构造监控指标
    metrics = struct();
    metrics.resource_allocation = mean(episode_results.avg_resource_allocation, 1);
    metrics.resource_efficiency = mean(episode_results.avg_efficiency);
    metrics.allocation_balance = mean(episode_results.avg_balance);
    metrics.detection_rate = mean([episode_results.attack_info{:}]);
    
    % 更新监控器
    monitor.updateMetrics(iteration, metrics);
    
    % 实时状态显示
    if mod(iteration, config.performance.display_interval) == 0
        monitor.displayRealTimeStatus(iteration);
    end
end

function updateAgentParameters(defender_agents, attacker_agent, config)
    % 更新智能体学习参数
    
    % 更新防御者智能体参数
    for i = 1:length(defender_agents)
        if isprop(defender_agents{i}, 'learning_rate')
            defender_agents{i}.learning_rate = config.learning_rate;
        end
        if isprop(defender_agents{i}, 'epsilon')
            defender_agents{i}.epsilon = config.epsilon;
        end
        if isprop(defender_agents{i}, 'temperature')
            defender_agents{i}.temperature = config.temperature;
        end
    end
    
    % 更新攻击者智能体参数
    if isprop(attacker_agent, 'learning_rate')
        attacker_agent.learning_rate = config.learning_rate;
    end
    if isprop(attacker_agent, 'epsilon')
        attacker_agent.epsilon = config.epsilon;
    end
    if isprop(attacker_agent, 'temperature')
        attacker_agent.temperature = config.temperature;
    end
end

function handleIterationOutput(iteration, config, iteration_time, episode_results)
    % 处理迭代输出和进度显示
    
    % 记录学习参数
    if exist('results', 'var')
        results.epsilon_history(iteration) = config.epsilon;
        results.learning_rate_history(iteration) = config.learning_rate;
        if isfield(config, 'temperature')
            results.temperature_history(iteration) = config.temperature;
        end
    end
    
    % 显示进度信息
    if mod(iteration, config.performance.display_interval) == 0
        avg_radi = mean(episode_results.avg_radi);
        avg_success_rate = mean([episode_results.attack_info{:}]);
        avg_efficiency = mean(episode_results.avg_efficiency);
        
        fprintf('Iteration %d/%d: RADI=%.3f, Success=%.3f, Efficiency=%.3f, Time=%.2fs\n', ...
                iteration, config.n_iterations, avg_radi, avg_success_rate, avg_efficiency, iteration_time);
        
        Logger.info(sprintf('迭代 %d 完成，用时 %.2f秒', iteration, iteration_time));
    end
    
    % 保存中间结果
    if mod(iteration, config.performance.save_interval) == 0
        fprintf('保存中间结果...\n');
        % 这里可以保存中间结果
    end
end

function saveCheckpoint(defender_agents, attacker_agent, results, iteration, config)
    % 保存训练检查点
    
    checkpoint_dir = config.output.checkpoints_dir;
    if ~exist(checkpoint_dir, 'dir')
        mkdir(checkpoint_dir);
    end
    
    checkpoint_file = fullfile(checkpoint_dir, sprintf('checkpoint_iter_%d.mat', iteration));
    
    try
        % 保存智能体状态
        agents_state = struct();
        agents_state.defender_agents = defender_agents;
        agents_state.attacker_agent = attacker_agent;
        agents_state.iteration = iteration;
        agents_state.config = config;
        agents_state.results = results;
        
        save(checkpoint_file, 'agents_state');
        Logger.info(sprintf('检查点已保存: %s', checkpoint_file));
        
    catch ME
        warning('保存检查点失败: %s', ME.message);
        Logger.warning(sprintf('检查点保存失败: %s', ME.message));
    end
end

function printFinalSummary(results, config)
    % 打印最终结果摘要
    
    fprintf('\n=== 仿真结果摘要 ===\n');
    
    % 性能指标
    final_radi = mean(results.radi_history(end-min(99,end-1):end, :), 'all');
    initial_radi = mean(results.radi_history(1:min(100,end), :), 'all');
    radi_improvement = initial_radi - final_radi;
    
    final_success_rate = mean(results.success_rate_history(end-min(99,end-1):end));
    initial_success_rate = mean(results.success_rate_history(1:min(100,end)));
    success_improvement = final_success_rate - initial_success_rate;
    
    final_efficiency = mean(results.resource_efficiency(end-min(99,end-1):end, :), 'all');
    
    fprintf('性能改善:\n');
    fprintf('  RADI: %.3f → %.3f (改善: %.3f)\n', initial_radi, final_radi, radi_improvement);
    fprintf('  攻击成功率: %.3f → %.3f (变化: %+.3f)\n', initial_success_rate, final_success_rate, success_improvement);
    fprintf('  资源效率: %.3f\n', final_efficiency);
    
    % 训练统计
    fprintf('\n训练统计:\n');
    fprintf('  总迭代数: %d\n', config.n_iterations);
    fprintf('  每轮Episodes: %d\n', config.n_episodes_per_iter);
    fprintf('  智能体数量: %d\n', size(results.radi_history, 2));
    fprintf('  最终探索率: %.3f\n', config.epsilon);
    fprintf('  最终学习率: %.3f\n', config.learning_rate);
    
    % 算法比较
    if size(results.radi_history, 2) > 1
        fprintf('\n算法性能对比 (最终RADI):\n');
        for i = 1:length(config.algorithms)
            final_radi_agent = mean(results.radi_history(end-min(99,end-1):end, i));
            fprintf('  %s: %.3f\n', config.algorithms{i}, final_radi_agent);
        end
    end
    
    % 收敛性分析
    if length(results.radi_history) > 100
        recent_var = var(results.radi_history(end-99:end, :), 0, 1);
        fprintf('\n收敛性分析:\n');
        fprintf('  近期RADI方差: %.6f\n', mean(recent_var));
        
        % 判断收敛状态
        if mean(recent_var) < config.performance.convergence_threshold
            fprintf('  收敛状态: ✓ 已收敛\n');
        else
            fprintf('  收敛状态: ⚠ 仍在学习\n');
        end
    end
    
    fprintf('==================\n');
end

%% === 使用示例和配置选择 ===

function demo_different_configs()
    % 演示不同配置的使用方法
    
    fprintf('=== 配置选择演示 ===\n');
    
    % 1. 使用默认配置
    fprintf('\n1. 默认配置仿真:\n');
    config1 = ConfigManager.getDefaultConfig();
    % main_fsp_simulation_with_config(config1);
    
    % 2. 使用优化配置
    fprintf('\n2. 优化配置仿真:\n');
    config2 = ConfigManager.getOptimizedConfig();
    % main_fsp_simulation_with_config(config2);
    
    % 3. 使用测试配置
    fprintf('\n3. 快速测试配置:\n');
    config3 = ConfigManager.getTestConfig();
    % main_fsp_simulation_with_config(config3);
    
    % 4. 自定义配置
    fprintf('\n4. 自定义配置示例:\n');
    config4 = ConfigManager.getDefaultConfig();
    
    % 修改特定参数
    config4.n_iterations = 2000;           % 增加迭代次数
    config4.learning_rate = 0.2;           % 提高学习率
    config4.epsilon = 0.6;                 % 增加探索
    config4.algorithms = {'Q-Learning'};   % 只使用Q-Learning
    
    % 修改可视化设置
    config4.output.visualization = true;
    config4.output.generate_report = true;
    
    % 保存自定义配置
    ConfigManager.saveConfig(config4, 'custom_config.json');
    
    fprintf('配置演示完成。\n');
end

function main_fsp_simulation_with_config(config)
    % 使用指定配置运行仿真的简化版本
    
    fprintf('使用配置运行仿真: %d次迭代, 学习率%.3f\n', ...
            config.n_iterations, config.learning_rate);
    
    % 这里可以调用完整的仿真流程
    % 为演示目的，只显示配置信息
    ConfigManager.displayConfigSummary(config);
end

%% === 配置验证和测试 ===

function validateAllConfigs()
    % 验证所有预定义配置的有效性
    
    fprintf('=== 配置验证测试 ===\n');
    
    configs = {
        ConfigManager.getDefaultConfig(), '默认配置';
        ConfigManager.getOptimizedConfig(), '优化配置';
        ConfigManager.getTestConfig(), '测试配置'
    };
    
    for i = 1:size(configs, 1)
        config = configs{i, 1};
        name = configs{i, 2};
        
        fprintf('\n验证 %s...\n', name);
        try
            ConfigManager.validateConfig(config);
            fprintf('✓ %s 验证通过\n', name);
        catch ME
            fprintf('❌ %s 验证失败: %s\n', name, ME.message);
        end
    end
    
    fprintf('\n配置验证完成。\n');
end

%% === 主程序入口点选择 ===

% 取消注释以下其中一行来运行不同的功能：

% 运行标准仿真
main_fsp_simulation_optimized();

% 演示不同配置
% demo_different_configs();

% 验证所有配置
% validateAllConfigs();