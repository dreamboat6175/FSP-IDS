function main_fsp()
    %% main_fsp - 简化版FSP-TCS主函数
    % =============================================
    % 描述：主函数只负责调用，所有配置由ConfigManager统一管理
    % 版本：v2.0 - 简化版本
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
        log_dir = fileparts(config.output.log_file);
        if ~exist(log_dir, 'dir')
            mkdir(log_dir);
        end
        
        % 初始化Logger
        if exist('Logger', 'class') == 8
            Logger.initialize(config.output.log_file);
            Logger.info('FSP-TCS仿真系统启动');
        end
        
        fprintf('📝 日志系统初始化完成\n');
    catch
        fprintf('⚠️  日志初始化失败，使用标准输出\n');
    end
end

%% 创建环境和智能体
function [env, agents] = createEnvironmentAndAgents(config)
    fprintf('🔧 创建环境和智能体...\n');
    
    % 准备环境配置 - 将嵌套结构展平为TCSEnvironment期望的格式
    env_config = prepareEnvironmentConfig(config);
    
    % 创建TCS环境
    env = TCSEnvironment(env_config);
    
    % 创建智能体
    agents = cell(length(config.algorithms.defender) + 1, 1);
    
    % 创建攻击者
    agents{1} = createAgent(config.algorithms.attacker, config, 'attacker');
    
    % 创建防御者们
    for i = 1:length(config.algorithms.defender)
        agents{i+1} = createAgent(config.algorithms.defender{i}, config, 'defender');
    end
    
    fprintf('✓ 环境和智能体创建完成\n');
end

%% 准备环境配置
function env_config = prepareEnvironmentConfig(config)
    % 将ConfigManager的嵌套配置转换为TCSEnvironment期望的平坦结构
    
    env_config = struct();
    
    % 系统基础参数
    env_config.n_stations = config.system.n_stations;
    env_config.n_components_per_station = config.system.n_components_per_station;
    env_config.total_components = config.system.total_components;
    env_config.total_resources = config.system.total_resources;
    env_config.n_resource_types = config.system.n_resource_types;
    env_config.n_attack_types = config.system.n_attack_types;
    env_config.state_space_size = config.system.state_space_size;
    env_config.action_space_size = config.system.action_space_size;
    
    % 仿真参数
    env_config.alpha_ewma = config.simulation.alpha_ewma;
    env_config.max_episode_steps = config.simulation.max_episode_steps;
    
    % 学习参数（用于内置智能体）
    env_config.attacker_lr = config.learning.learning_rate;
    env_config.attacker_gamma = config.learning.discount_factor;
    env_config.attacker_epsilon = config.learning.epsilon;
    env_config.attacker_epsilon_decay = config.learning.epsilon_decay;
    env_config.attacker_epsilon_min = config.learning.epsilon_min;
    
    % 环境参数
    env_config.reward_scaling = config.environment.reward_scaling;
    env_config.noise_level = config.environment.noise_level;
    env_config.dynamic_environment = config.environment.dynamic_environment;
    env_config.failure_probability = config.environment.failure_probability;
    env_config.repair_probability = config.environment.repair_probability;
    env_config.attack_success_rate = config.environment.attack_success_rate;
    env_config.defense_success_rate = config.environment.defense_success_rate;
    
    % 安全参数
    env_config.attack_detection_rate = config.security.attack_detection_rate;
    env_config.false_positive_rate = config.security.false_positive_rate;
    env_config.detection_enabled = true;
    env_config.base_detection_rate = config.security.attack_detection_rate;
    env_config.detection_sensitivity = 1 - config.security.false_positive_rate;
    
    % 调试参数
    env_config.debug_mode = config.debug.enabled;
    
    fprintf('✓ 环境配置准备完成\n');
end

%% 创建单个智能体
function agent = createAgent(algorithm, config, type)
    % 准备智能体配置
    agent_config = prepareAgentConfig(config);
    
    % 提取必要的参数
    state_dim = config.system.state_space_size;
    action_dim = config.system.action_space_size;
    agent_name = sprintf('%s_%s', algorithm, type);
    
    % 根据算法类型创建智能体
    try
        switch algorithm
            case 'QLearning'
                agent = QLearningAgent(agent_name, type, agent_config, state_dim, action_dim);
            case 'SARSA'
                if exist('SARSAAgent', 'class') == 8
                    agent = SARSAAgent(agent_name, type, agent_config, state_dim, action_dim);
                else
                    fprintf('⚠️  SARSA不可用，使用QLearning替代\n');
                    agent = QLearningAgent(agent_name, type, agent_config, state_dim, action_dim);
                end
            case 'DQN'
                if exist('DQNAgent', 'class') == 8
                    agent = DQNAgent(agent_name, type, agent_config, state_dim, action_dim);
                else
                    fprintf('⚠️  DQN不可用，使用QLearning替代\n');
                    agent = QLearningAgent(agent_name, type, agent_config, state_dim, action_dim);
                end
            case 'DoubleQLearning'
                if exist('DoubleQLearningAgent', 'class') == 8
                    agent = DoubleQLearningAgent(agent_name, type, agent_config, state_dim, action_dim);
                else
                    fprintf('⚠️  DoubleQLearning不可用，使用QLearning替代\n');
                    agent = QLearningAgent(agent_name, type, agent_config, state_dim, action_dim);
                end
            otherwise
                error('未知算法类型: %s', algorithm);
        end
        
        fprintf('✓ %s智能体创建成功 (%s)\n', algorithm, type);
        
    catch ME
        fprintf('⚠️  %s智能体创建失败: %s\n', algorithm, ME.message);
        fprintf('    使用QLearning作为备选方案\n');
        agent = QLearningAgent(agent_name, type, agent_config, state_dim, action_dim);
    end
end

%% 准备智能体配置
function agent_config = prepareAgentConfig(config)
    % 将ConfigManager的配置转换为智能体期望的格式
    
    agent_config = struct();
    
    % 基本学习参数
    agent_config.learning_rate = config.learning.learning_rate;
    agent_config.discount_factor = config.learning.discount_factor;
    agent_config.epsilon = config.learning.epsilon;
    agent_config.epsilon_decay = config.learning.epsilon_decay;
    agent_config.epsilon_min = config.learning.epsilon_min;
    
    % DQN特有参数
    agent_config.target_update_frequency = config.learning.target_update_frequency;
    agent_config.replay_buffer_size = config.learning.replay_buffer_size;
    agent_config.batch_size = config.learning.batch_size;
    agent_config.tau = config.learning.tau;
    
    % 系统参数
    agent_config.state_dim = config.system.state_space_size;
    agent_config.action_dim = config.system.action_space_size;
    agent_config.n_stations = config.system.n_stations;
    
    % 策略池参数
    agent_config.pool_size_limit = config.simulation.pool_size_limit;
    agent_config.pool_update_interval = config.simulation.pool_update_interval;
    
    % 调试参数
    agent_config.debug_mode = config.debug.enabled;
    agent_config.verbose = config.debug.verbose;
    
    fprintf('✓ 智能体配置准备完成\n');
end

%% 运行仿真
function runSimulation(env, agents, simulator, config)
    fprintf('🔄 开始仿真循环...\n');
    fprintf('总迭代次数：%d\n', config.simulation.n_iterations);
    fprintf('每次迭代episodes：%d\n\n', config.simulation.n_episodes_per_iter);
    
    % 运行主仿真循环
    for iteration = 1:config.simulation.n_iterations
        try
            % 运行当前迭代
            episode_results = simulator.runEpisodes(env, agents(2:end), agents{1}, config);
            
            % 输出迭代结果
            outputIterationResults(iteration, agents, episode_results, config);
            
        catch ME
            fprintf('⚠️  迭代 %d 出错: %s\n', iteration, ME.message);
            continue;
        end
    end
    
    fprintf('✅ 所有 %d 次迭代完成！\n', config.simulation.n_iterations);
end

%% 输出迭代结果
function outputIterationResults(iteration, agents, episode_results, config)
    % 根据配置决定是否输出详细结果
    if config.output.show_iteration_details
        fprintf('--- 迭代 %d/%d 完成 ---\n', iteration, config.simulation.n_iterations);
        
        % 显示智能体性能（如果有结果数据）
        if ~isempty(episode_results)
            for i = 1:length(agents)-1
                fprintf('防御者%d 平均奖励: %.3f\n', i, mean(episode_results.defender_rewards(:,i)));
            end
            fprintf('攻击者平均奖励: %.3f\n', mean(episode_results.attacker_rewards));
        end
        fprintf('\n');
    else
        % 简洁输出
        if mod(iteration, config.output.progress_interval) == 0
            fprintf('进度: %d/%d (%.1f%%)\n', iteration, config.simulation.n_iterations, ...
                   100 * iteration / config.simulation.n_iterations);
        end
    end
end

%% 错误处理
function handleError(ME)
    fprintf('❌ 仿真出错: %s\n', ME.message);
    
    if ~isempty(ME.stack)
        fprintf('错误位置: %s (第%d行)\n', ME.stack(1).file, ME.stack(1).line);
    end
    
    % 记录到日志
    try
        if exist('Logger', 'class') == 8
            Logger.error(sprintf('仿真出错: %s', ME.message));
            if ~isempty(ME.stack)
                Logger.error(sprintf('错误位置: %s, 行号: %d', ME.stack(1).file, ME.stack(1).line));
            end
        end
    catch
        % 日志记录失败也不影响错误报告
    end
    
    fprintf('\n🔧 解决建议:\n');
    fprintf('1. 检查ConfigManager.m配置是否正确\n');
    fprintf('2. 验证所有必要的类文件是否存在\n');
    fprintf('3. 运行: addpath(genpath(pwd))\n');
end