function main_fsp()
    %% main_fsp - 修复版FSP-TCS主函数
    % =============================================
    % 描述：主函数只负责调用，修复配置字段访问问题
    % 版本：v2.1 - 修复版本
    % =============================================
    
    clear;
    clc;
    close all;
    
    fprintf('🚀 FSP-TCS仿真系统启动\n\n');
    
    % 添加路径
    addpath(genpath(pwd));
    
    try
        % 1. 加载配置（所有参数由ConfigManager统一管理）
        config = ConfigManager.loadConfig();
        
        % 2. 初始化日志系统
        initializeLogging(config);
        
        % 3. 创建环境和智能体
        [env, agents] = createEnvironmentAndAgents(config);
        
        % 4. 创建仿真器
        simulator = FSPSimulator(config);
        
        % 5. 运行仿真
        runSimulation(env, agents, simulator, config);
        
        fprintf('✅ FSP仿真完成！\n');
        
    catch ME
        handleError(ME);
    end
end

%% 初始化日志系统
function initializeLogging(config)
    try
        % 确保日志目录存在
        log_file = '';
        if isfield(config, 'output') && isfield(config.output, 'log_file')
            log_file = config.output.log_file;
        elseif isfield(config, 'log_file')
            log_file = config.log_file;
        else
            log_file = 'logs/simulation.log';
        end
        
        % 创建日志目录
        log_dir = fileparts(log_file);
        if ~isempty(log_dir) && ~exist(log_dir, 'dir')
            mkdir(log_dir);
        end
        
        % 初始化Logger
        if exist('Logger', 'class') == 8
            Logger.initialize(log_file);
            Logger.info('FSP-TCS仿真系统启动');
        end
        
        fprintf('✓ 日志系统初始化: %s\n', log_file);
    catch ME
        fprintf('⚠️  日志初始化失败，使用标准输出: %s\n', ME.message);
    end
end

%% 创建环境和智能体
function [env, agents] = createEnvironmentAndAgents(config)
    fprintf('🔧 创建环境和智能体...\n');
    
    % 准备环境配置 - 安全地访问配置字段
    env_config = prepareEnvironmentConfig(config);
    
    % 创建TCS环境
    env = TCSEnvironment(env_config);
    
    % 创建智能体
    defender_algorithms = getConfigValue(config, 'algorithms.defender', {'QLearning'});
    attacker_algorithm = getConfigValue(config, 'algorithms.attacker', 'QLearning');
    
    agents = cell(length(defender_algorithms) + 1, 1);
    
    % 创建攻击者
    agents{1} = createAgent(attacker_algorithm, config, 'attacker');
    
    % 创建防御者们
    for i = 1:length(defender_algorithms)
        agents{i+1} = createAgent(defender_algorithms{i}, config, 'defender');
    end
    
    fprintf('✓ 环境和智能体创建完成\n');
end

%% 准备环境配置
function env_config = prepareEnvironmentConfig(config)
    % 安全地将ConfigManager的嵌套配置转换为TCSEnvironment期望的平坦结构
    
    env_config = struct();
    
    % 系统基础参数 - 使用安全访问函数
    env_config.n_stations = getConfigValue(config, 'system.n_stations', 10);
    
    % 处理 n_components_per_station 字段
    if isfield(config, 'n_components_per_station')
        env_config.n_components_per_station = config.n_components_per_station;
    elseif isfield(config, 'system') && isfield(config.system, 'n_components_per_station')
        env_config.n_components_per_station = config.system.n_components_per_station;
    else
        % 使用默认值：每个主站3个组件
        env_config.n_components_per_station = repmat(3, 1, env_config.n_stations);
    end
    
    env_config.total_components = getConfigValue(config, 'system.total_components', sum(env_config.n_components_per_station));
    env_config.total_resources = getConfigValue(config, 'system.total_resources', 100);
    env_config.n_resource_types = getConfigValue(config, 'system.n_resource_types', 5);
    env_config.n_attack_types = getConfigValue(config, 'system.n_attack_types', 6);
    env_config.state_space_size = getConfigValue(config, 'system.state_space_size', 77);
    env_config.action_space_size = getConfigValue(config, 'system.action_space_size', 20);
    
    % 仿真参数
    env_config.alpha_ewma = getConfigValue(config, 'simulation.alpha_ewma', 0.1);
    env_config.max_episode_steps = getConfigValue(config, 'simulation.max_episode_steps', 100);
    
    % 学习参数（用于内置智能体）
    env_config.attacker_lr = getConfigValue(config, 'learning.learning_rate', 0.1);
    env_config.attacker_gamma = getConfigValue(config, 'learning.discount_factor', 0.95);
    env_config.attacker_epsilon = getConfigValue(config, 'learning.epsilon', 0.3);
    env_config.attacker_epsilon_decay = getConfigValue(config, 'learning.epsilon_decay', 0.995);
    env_config.attacker_epsilon_min = getConfigValue(config, 'learning.epsilon_min', 0.01);
    
    % 环境参数
    env_config.reward_scaling = getConfigValue(config, 'environment.reward_scaling', 1);
    env_config.noise_level = getConfigValue(config, 'environment.noise_level', 0);
    env_config.attack_success_rate = getConfigValue(config, 'environment.attack_success_rate', 0.3);
    env_config.defense_success_rate = getConfigValue(config, 'environment.defense_success_rate', 0.7);
    
    % 调试参数
    env_config.debug_mode = getConfigValue(config, 'debug.debug_mode', false);
    
    % 为了向后兼容，添加顶层字段
    field_names = fieldnames(env_config);
    for i = 1:length(field_names)
        field = field_names{i};
        if ~isfield(config, field)
            config.(field) = env_config.(field);
        end
    end
end

%% 安全获取配置值
function value = getConfigValue(config, field_path, default_value)
    % 安全地获取嵌套配置值
    % 输入: config - 配置结构体
    %       field_path - 字段路径（如 'learning.learning_rate'）
    %       default_value - 默认值
    
    if nargin < 3
        default_value = [];
    end
    
    try
        path_parts = strsplit(field_path, '.');
        value = config;
        
        for i = 1:length(path_parts)
            if isstruct(value) && isfield(value, path_parts{i})
                value = value.(path_parts{i});
            else
                value = default_value;
                return;
            end
        end
        
    catch
        value = default_value;
    end
end

%% 创建智能体
function agent = createAgent(algorithm, config, role)
    try
        % 获取状态和动作空间维度
        state_dim = getConfigValue(config, 'system.state_space_size', 77);
        action_dim = getConfigValue(config, 'system.action_space_size', 20);
        
        % 生成智能体名称
        if strcmp(role, 'attacker')
            agent_name = sprintf('%s_attacker', algorithm);
        else
            agent_name = sprintf('%s_defender', algorithm);
        end
        
        if exist('AgentFactory', 'class') == 8
            agent = AgentFactory.createAgent(algorithm, config, role);
        else
            % 简化的智能体创建 - 使用正确的构造函数参数
            switch algorithm
                case 'QLearning'
                    if exist('QLearningAgent', 'class') == 8
                        agent = QLearningAgent(agent_name, role, config, state_dim, action_dim);
                    else
                        error('QLearningAgent类不存在');
                    end
                case 'SARSA'
                    if exist('SARSAAgent', 'class') == 8
                        agent = SARSAAgent(agent_name, role, config, state_dim, action_dim);
                    else
                        error('SARSAAgent类不存在');
                    end
                case 'DQN'
                    if exist('DQNAgent', 'class') == 8
                        agent = DQNAgent(agent_name, role, config, state_dim, action_dim);
                    else
                        fprintf('⚠️  DQN不可用，使用QLearning替代\n');
                        agent = QLearningAgent(agent_name, role, config, state_dim, action_dim);
                    end
                case 'DoubleQLearning'
                    if exist('DoubleQLearningAgent', 'class') == 8
                        agent = DoubleQLearningAgent(agent_name, role, config, state_dim, action_dim);
                    else
                        fprintf('⚠️  DoubleQLearning不可用，使用QLearning替代\n');
                        agent = QLearningAgent(agent_name, role, config, state_dim, action_dim);
                    end
                otherwise
                    fprintf('⚠️  未知算法%s，使用QLearning替代\n', algorithm);
                    agent = QLearningAgent(agent_name, role, config, state_dim, action_dim);
            end
        end
        
        fprintf('✓ 智能体创建成功: %s (%s)\n', agent_name, algorithm);
        
    catch ME
        fprintf('❌ 创建智能体失败: %s\n', ME.message);
        % 使用最基本的智能体 - 提供所有必需参数
        try
            fallback_name = sprintf('fallback_%s', role);
            fallback_config = struct('learning_rate', 0.1, 'discount_factor', 0.9, ...
                                   'epsilon', 0.1, 'epsilon_decay', 0.995, 'epsilon_min', 0.01);
            state_dim = getConfigValue(config, 'system.state_space_size', 77);
            action_dim = getConfigValue(config, 'system.action_space_size', 20);
            agent = QLearningAgent(fallback_name, role, fallback_config, state_dim, action_dim);
            fprintf('⚠️  使用备用QLearning智能体\n');
        catch ME2
            % 创建最基本的配置
            fprintf('❌ 创建基础智能体也失败: %s\n', ME2.message);
            fprintf('🔧 尝试使用硬编码配置创建智能体\n');
            
            % 硬编码最小配置
            basic_config = struct();
            basic_config.learning_rate = 0.1;
            basic_config.discount_factor = 0.9;
            basic_config.epsilon = 0.1;
            basic_config.epsilon_decay = 0.995;
            basic_config.epsilon_min = 0.01;
            
            state_dim = 77;  % 硬编码默认值
            action_dim = 20; % 硬编码默认值
            fallback_name = sprintf('emergency_%s', role);
            
            agent = QLearningAgent(fallback_name, role, basic_config, state_dim, action_dim);
            fprintf('✓ 紧急智能体创建成功: %s\n', fallback_name);
        end
    end
end

%% 运行仿真
function runSimulation(env, agents, simulator, config)
    try
        fprintf('🚀 开始FSP仿真...\n');
        
        n_iterations = getConfigValue(config, 'simulation.n_iterations', 50);
        
        % 分离攻击者和防御者
        attacker_agent = agents{1};
        defender_agents = agents(2:end);
        
        for iter = 1:n_iterations
            fprintf('📊 FSP迭代 %d/%d\n', iter, n_iterations);
            
            % 运行episodes
            results = simulator.runEpisodes(env, defender_agents, attacker_agent, config);
            
            % 显示结果
            if ~isempty(results) && isfield(results, 'mean_rewards')
                fprintf('  平均奖励: %.4f\n', mean(results.mean_rewards));
            end
            
            % 定期保存
            save_interval = getConfigValue(config, 'simulation.save_interval', 10);
            if mod(iter, save_interval) == 0
                saveCheckpoint(results, iter, config);
            end
        end
        
        fprintf('✅ FSP仿真完成!\n');
        
    catch ME
        fprintf('❌ 仿真运行失败: %s\n', ME.message);
        rethrow(ME);
    end
end

%% 保存检查点
function saveCheckpoint(results, iter, config)
    try
        results_dir = getConfigValue(config, 'output.results_dir', 'results');
        if ~exist(results_dir, 'dir')
            mkdir(results_dir);
        end
        
        checkpoint_file = fullfile(results_dir, sprintf('checkpoint_iter_%d.mat', iter));
        save(checkpoint_file, 'results', 'iter');
        fprintf('💾 检查点已保存: %s\n', checkpoint_file);
    catch ME
        fprintf('⚠️  保存检查点失败: %s\n', ME.message);
    end
end

%% 错误处理
function handleError(ME)
    fprintf('❌ 仿真出错: %s\n', ME.message);
    
    if ~isempty(ME.stack)
        fprintf('错误位置: %s (第%d行)\n', ME.stack(1).file, ME.stack(1).line);
        
        % 记录到日志
        try
            if exist('Logger', 'class') == 8
                Logger.error('仿真出错: %s', ME.message);
                Logger.error('错误位置: %s, 行号: %d', ME.stack(1).file, ME.stack(1).line);
            end
        catch
            % 忽略日志记录错误
        end
    end
    
    fprintf('🔧 解决建议:\n');
    fprintf('1. 检查ConfigManager.m配置是否正确\n');
    fprintf('2. 验证所有必要的类文件是否存在\n');
    fprintf('3. 运行: addpath(genpath(pwd))\n');
    
    % 可选：显示完整错误堆栈
    if length(ME.stack) > 1
        fprintf('\n完整错误堆栈:\n');
        for i = 1:length(ME.stack)
            fprintf('  %d. %s (第%d行)\n', i, ME.stack(i).file, ME.stack(i).line);
        end
    end
end