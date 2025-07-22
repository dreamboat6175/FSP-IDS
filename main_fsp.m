%% main_fsp.m - FSP-TCS 仿真系统主入口 (改进版)
% =========================================================================
% 描述: 该脚本是FSP-TCS仿真系统的主入口。
%       它负责加载配置、初始化环境和智能体、运行FSP迭代、
%       收集结果并生成报告。
% =========================================================================

% 清理工作区、命令行和关闭所有图形窗口
clear;
clc;
close all;

%% 1. 添加路径
% =========================================================================
% 确保所有子文件夹都在MATLAB路径中，以便系统能够找到所有类和函数文件。
% =========================================================================
addpath(genpath(pwd));
rehash toolboxcache; % 刷新MATLAB的路径缓存
fprintf('🚀 FSP-TCS仿真系统启动\n');

%% 2. 加载配置
% =========================================================================
% 使用ConfigManager加载和验证配置。
% 默认从 config/default_config.json 加载，也可以指定其他文件。
% =========================================================================
config = []; % 初始化 config 变量
try
    % 尝试加载用户自定义的仿真配置文件
    config_file = fullfile('config', 'simulation_config.json');
    if ~exist(config_file, 'file')
        % 如果用户配置文件不存在，则加载默认配置文件
        config_file = fullfile('config', 'default_config.json');
        fprintf('⚠️ 未找到用户配置文件 (%s)，将加载默认配置。\n', fullfile('config', 'simulation_config.json'));
    end
    
    % 调用 ConfigManager 加载并验证配置
    config = ConfigManager.loadConfig(config_file);
    
    % 显示加载和合并后的配置摘要
    ConfigManager.displayConfig(config); 
    
catch ME
    fprintf('❌ 配置加载或验证过程中发生严重错误: %s\n', ME.message);
    fprintf('💡 尝试使用硬编码的备用默认配置以继续...\n');
    % 如果配置加载失败，使用硬编码的默认配置作为备用方案
    config = ConfigManager.getDefaultConfig();
    ConfigManager.displayConfig(config); % 显示备用配置
end

% 再次检查 config 是否成功加载为结构体
if ~isstruct(config)
    fprintf('❌ 错误: 配置对象未成功初始化为结构体。仿真无法继续。\n');
    return; % 终止脚本执行
end

%% 3. 初始化日志系统
% =========================================================================
% 根据配置初始化日志系统
% =========================================================================
try
    log_file_path = ConfigManager.getConfigValue(config, 'output.log_file', 'simulation.log');
    log_level = ConfigManager.getConfigValue(config, 'debug.log_level', 'INFO');
    
    Logger.init(log_file_path, log_level);
    Logger.info('日志系统初始化完成');
    fprintf('✓ 日志系统初始化: %s\n', log_file_path);
catch ME
    fprintf('❌ 日志系统初始化失败: %s\n', ME.message);
    fprintf('✓ 将使用默认日志文件: simulation.log\n');
    % 如果日志系统初始化失败，使用备用日志配置
    Logger.init('simulation.log', 'INFO');
    Logger.warning('日志系统初始化失败，已切换到备用日志配置。');
end
Logger.info('FSP-TCS仿真系统启动');

%% 4. 创建环境和智能体
% =========================================================================
% 根据配置创建TCS环境和RL智能体（攻击者和防御者）。
% =========================================================================
fprintf('🔧 创建环境和智能体...\n');
env = []; % 初始化环境和智能体变量
attacker_agent = [];
defender_agents = {};

try
    % --- 修正: TCSEnvironment 构造函数现在只接受一个 config 参数 ---
    % TCSEnvironment 会在内部从 config 中提取所有必要的参数
    env = TCSEnvironment(config); % 传递整个配置结构体
    Logger.info('TCS 环境创建成功。');

    % --- 从已创建的环境对象中获取状态和动作空间大小 ---
    % 这些值现在由 TCSEnvironment 内部计算并提供
    state_space_size = env.state_dim;
    action_space_size = env.action_dim;

    % --- 关键检查：确保状态和动作空间大小有效 ---
    if ~isnumeric(state_space_size) || state_space_size <= 0
        error('环境创建后 state_space_size 无效或缺失 (%s)。请检查 TCSEnvironment 内部计算逻辑。', num2str(state_space_size));
    end
    if ~isnumeric(action_space_size) || action_space_size <= 0
        error('环境创建后 action_space_size 无效或缺失 (%s)。请检查 TCSEnvironment 内部计算逻辑。', num2str(action_space_size));
    end
    % --- 结束关键检查 ---

    % 创建攻击者智能体
    attacker_type = ConfigManager.getConfigValue(config, 'algorithms.attacker', 'QLearning');
    try
        % 尝试使用 AgentFactory 创建攻击者智能体
        attacker_agent = AgentFactory.createAgent(attacker_type, 'Attacker', 'attacker', config, state_space_size, action_space_size);
        Logger.info(sprintf('攻击者智能体 "%s" 创建成功。', attacker_agent.name));
    catch ME_Attacker
        Logger.error(sprintf('创建攻击者智能体 "%s" 失败: %s', attacker_type, ME_Attacker.message));
        fprintf('❌ 创建攻击者智能体失败: %s\n', ME_Attacker.message);
        fprintf('⚠️  将使用备用 QLearning 智能体作为攻击者。\n');
        % 备用方案：使用硬编码的 QLearningAgent
        try
            attacker_agent = QLearningAgent('Attacker_Fallback', 'attacker', config, state_space_size, action_space_size);
            Logger.info('备用 QLearning 攻击者智能体创建成功。');
        catch ME_FallbackAttacker
            Logger.error(sprintf('创建备用 QLearning 攻击者智能体也失败: %s', ME_FallbackAttacker.message));
            fprintf('❌ 创建备用攻击者智能体也失败: %s\n', ME_FallbackAttacker.message);
            rethrow(ME_FallbackAttacker); % 如果备用智能体也无法创建，则重新抛出错误
        end
    end

    % 创建防御者智能体
    defender_types = ConfigManager.getConfigValue(config, 'algorithms.defender', {'QLearning'});
    if ~iscell(defender_types) % 确保 defender_types 是 cell 数组
        defender_types = {defender_types};
    end

    for i = 1:length(defender_types)
        current_defender_type = defender_types{i};
        try
            % 尝试使用 AgentFactory 创建防御者智能体
            defender_agents{i} = AgentFactory.createAgent(current_defender_type, ...
                                                          sprintf('Defender_%s', current_defender_type), ...
                                                          'defender', config, state_space_size, action_space_size);
            Logger.info(sprintf('防御者智能体 "%s" 创建成功。', defender_agents{i}.name));
        catch ME_Defender
            Logger.error(sprintf('创建防御者智能体 "%s" 失败: %s', current_defender_type, ME_Defender.message));
            fprintf('❌ 创建防御者智能体 "%s" 失败: %s\n', current_defender_type, ME_Defender.message);
            fprintf('⚠️  将使用备用 QLearning 智能体作为防御者。\n');
            % 备用方案：使用硬编码的 QLearningAgent
            try
                defender_agents{i} = QLearningAgent(sprintf('Defender_%s_Fallback', current_defender_type), ...
                                                    'defender', config, state_space_size, action_space_size);
                Logger.info(sprintf('备用 QLearning 防御者智能体 "%s" 创建成功。', defender_agents{i}.name));
            catch ME_FallbackDefender
                Logger.error(sprintf('创建备用 QLearning 防御者智能体也失败: %s', ME_FallbackDefender.message));
                fprintf('❌ 创建备用防御者智能体也失败: %s\n', ME_FallbackDefender.message);
                rethrow(ME_FallbackDefender); % 如果备用智能体也无法创建，则重新抛出错误
            end
        end
    end
    fprintf('✓ 环境和智能体创建完成。\n');

catch ME_Create
    Logger.error(sprintf('环境或智能体创建过程中发生致命错误: %s', ME_Create.message));
    fprintf('❌ 仿真出错: 环境或智能体创建失败。错误信息: %s\n', ME_Create.message);
    fprintf('错误位置: %s (第%d行)\n', ME_Create.stack(1).file, ME_Create.stack(1).line);
    fprintf('🔧 解决建议:\n');
    fprintf('1. 检查 ConfigManager.m 中配置值的获取和验证逻辑。\n');
    fprintf('2. 验证 AgentFactory.m 和所有智能体类 (RLAgent, QLearningAgent 等) 文件是否存在且无语法错误。\n');
    fprintf('3. 确保配置中 "system.state_space_size" 和 "system.action_space_size" 为有效的正数值。\n');
    fprintf('4. 确保所有类文件都在 MATLAB 路径中 (已执行 addpath(genpath(pwd)))。\n');
    return; % 终止脚本执行
end

% 再次检查智能体是否成功创建
if isempty(attacker_agent) || isempty(defender_agents)
    fprintf('❌ 错误: 智能体未成功创建。仿真无法继续。\n');
    Logger.error('智能体未成功创建，仿真终止。');
    return; % 终止脚本执行
end

%% 5. 初始化结果收集器和性能监控器
% =========================================================================
% 设置用于存储仿真结果和监控性能的组件。
% ResultsCollector 负责数据的持久化，PerformanceMonitor 负责实时性能跟踪。
% =========================================================================
Logger.info('初始化结果收集器和性能监控器。');
% 修正: ResultsCollector 构造函数现在需要 agents_list 和 config
all_agents = [{attacker_agent}, defender_agents]; % 创建所有智能体的列表
results_collector = ResultsCollector(all_agents, config); 
performance_monitor = PerformanceMonitor(config.simulation.n_iterations, ...
                                         config.simulation.n_episodes_per_iter, ...
                                         length(defender_agents)); % 传入防御者数量
fprintf('✓ 结果收集器和性能监控器初始化完成。\n');

%% 6. 运行FSP仿真主循环
% =========================================================================
% FSP (Fictitious Play) 迭代过程是仿真的核心。
% 攻击者和防御者智能体在此过程中通过反复博弈学习和适应。
% =========================================================================
Logger.info('开始 FSP 仿真主循环。');
fprintf('🔄 开始 FSP 仿真主循环 (%d 迭代)...\n', config.simulation.n_iterations);

global_timer = tic; % 启动全局计时器，用于跟踪总运行时间

for iter = 1:config.simulation.n_iterations
    Logger.info(sprintf('--- FSP 迭代 %d/%d ---', iter, config.simulation.n_iterations));
    fprintf('--- FSP 迭代 %d/%d ---\n', iter, config.simulation.n_iterations);
    
    iter_timer = tic; % 启动当前迭代计时器

    % 6.1. 智能体学习阶段
    % =====================================================================
    % 在每个 FSP 迭代中，智能体进行多轮 episode 学习，与环境交互并更新策略。
    % =====================================================================
    Logger.info('智能体学习阶段开始。');
    
    % 获取当前策略（用于日志和可视化，表示智能体在当前迭代开始时的策略）
    attacker_policy = attacker_agent.getPolicy();
    defender_policies = cell(1, length(defender_agents));
    for i = 1:length(defender_agents)
        defender_policies{i} = defender_agents{i}.getPolicy();
    end

    % 运行多个 episode，收集本迭代的原始仿真数据
    % runSimpleEpisodes 函数负责智能体与环境的交互和奖励计算
    [iter_rewards, iter_detections, iter_resource_utilization, iter_allocation_balance] = ...
        runSimpleEpisodes(env, attacker_agent, defender_agents, config);
    
    % 记录当前迭代的总奖励（用于性能监控）
    current_attacker_reward = sum(iter_rewards.attacker_total);
    % 对于多个防御者，iter_rewards.defender_total 可能是矩阵，按列求和得到每个防御者的总和
    current_defender_rewards = sum(iter_rewards.defender_total, 1); 

    % 6.2. 更新性能监控器
    % =====================================================================
    % 将当前迭代的性能数据传递给 PerformanceMonitor 进行聚合和更新。
    % =====================================================================
    performance_monitor.update(iter, current_attacker_reward, current_defender_rewards, ...
                               iter_detections, iter_resource_utilization, iter_allocation_balance);
    
    % 6.3. 记录迭代结果
    % =====================================================================
    % 将当前迭代的详细结果保存到 ResultsCollector，以便后续分析和报告生成。
    % =====================================================================
    results_collector.recordIterationResults(iter, attacker_agent, defender_agents, ...
                                             iter_rewards, iter_detections, iter_resource_utilization, ...
                                             iter_allocation_balance, attacker_policy, defender_policies);

    % 6.4. 智能体参数衰减
    % =====================================================================
    % 衰减学习率和探索参数 (如 epsilon 或 temperature)，以促进智能体策略的收敛。
    % =====================================================================
    attacker_agent.decay();
    for i = 1:length(defender_agents)
        defender_agents{i}.decay();
    end

    % 6.5. 输出迭代进度
    % =====================================================================
    % 在控制台和日志中显示当前迭代的进度和关键性能指标。
    % =====================================================================
    outputIterationResults(iter, config.simulation.n_iterations, toc(iter_timer), ...
                           current_attacker_reward, current_defender_rewards, ...
                           performance_monitor.current_detection_rate, ...
                           performance_monitor.current_resource_efficiency);

    % 6.6. 检查点保存
    % =====================================================================
    % 定期保存智能体模型和仿真状态，以便在仿真中断时可以从检查点恢复。
    % =====================================================================
    if ConfigManager.getConfigValue(config, 'output.save_checkpoints', true) && ...
       mod(iter, ConfigManager.getConfigValue(config, 'simulation.save_interval', 10)) == 0
        Logger.info(sprintf('保存检查点在迭代 %d', iter));
        results_collector.saveCheckpoint(iter, attacker_agent, defender_agents);
    end

    % 6.7. 检查最大运行时间
    % =====================================================================
    % 如果仿真总运行时间超过预设的最大时间限制，则提前终止仿真。
    % =====================================================================
    max_time_hours = ConfigManager.getConfigValue(config, 'simulation.max_time_hours', 24);
    if toc(global_timer) / 3600 > max_time_hours
        Logger.warning(sprintf('仿真时间超过最大限制 (%.2f 小时)，提前终止。', max_time_hours));
        fprintf('⚠️ 仿真时间超过最大限制 (%.2f 小时)，提前终止。\n', max_time_hours);
        break; % 退出 FSP 迭代循环
    end
end

Logger.info('FSP仿真主循环结束。');
fprintf('✓ FSP仿真主循环结束。\n');

%% 7. 仿真结果后处理与报告生成
% =========================================================================
% 仿真结束后，保存最终结果，并生成详细的性能报告和可视化图表。
% =========================================================================
Logger.info('开始仿真结果后处理和报告生成。');
fprintf('📊 生成仿真报告...\n');

try
    % 保存最终训练好的智能体模型
    results_collector.saveAgentModels(attacker_agent, defender_agents);

    % 保存所有在仿真过程中收集到的结果数据
    results_collector.saveAllResults();

    % 生成可视化报告，包括图表和关键指标总结
    generateVisualizationReport(performance_monitor, results_collector, config);
    Logger.info('仿真报告生成完成。');
    fprintf('✓ 仿真报告生成完成。\n');

catch ME_Report
    Logger.error(sprintf('结果后处理或报告生成失败: %s', ME_Report.message));
    fprintf('❌ 结果后处理或报告生成失败: %s\n', ME_Report.message);
    fprintf('错误位置: %s (第%d行)\n', ME_Report.stack(1).file, ME_Report.stack(1).line);
end

Logger.info('FSP-TCS仿真系统运行结束。');
fprintf('✅ FSP-TCS仿真系统运行结束。\n');
