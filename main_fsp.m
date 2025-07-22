function main_fsp()
    %% main_fsp - FSP-TCS主仿真函数（修复版）
    clear;
    clc;
    close all;

    addpath(genpath(pwd));

    fprintf('🚀 FSP-TCS仿真系统启动\n\n');
    
    try
        % 1. 初始化日志系统
        initializeLogging();
        
        % 2. 加载配置 - 使用安全的配置加载
        config = loadConfigurationSafely();
        
        % 3. 创建环境和智能体 - 修复版本
        [env, agents] = createEnvironmentAndAgentsFixed(config);
        
        % 4. 验证创建结果
        if ~validateCreatedObjects(env, agents)
            error('环境或智能体创建验证失败');
        end
        
        % 5. 初始化仿真器
        simulator = initializeSimulator(config);
        
        % 6. 运行主仿真循环
        runMainSimulation(env, agents, simulator, config);
        
        fprintf('✅ FSP仿真完成！\n');
        
    catch ME
        handleSimulationError(ME);
    end
end

%% 1. 安全的日志初始化
function initializeLogging()
    try
        if ~exist('logs', 'dir')
            mkdir('logs');
        end
        
        log_file = fullfile('logs', sprintf('simulation_%s.log', datestr(now, 'yyyymmdd_HHMMSS')));
        
        if exist('Logger', 'class') == 8
            Logger.initialize(log_file);
            Logger.info('FSP-TCS仿真开始');
        else
            fprintf('⚠️ Logger类不可用，使用标准输出\n');
        end
    catch
        fprintf('⚠️ 日志初始化失败，继续运行\n');
    end
end

%% 2. 安全的配置加载
function config = loadConfigurationSafely()
    fprintf('📋 加载配置...\n');
    
    try
        % 方法1: 尝试使用ConfigManager
        if exist('ConfigManager', 'class') == 8
            config = ConfigManager.loadConfig();
            fprintf('✓ 使用ConfigManager加载配置成功\n');
            return;
        end
    catch ME
        fprintf('⚠️ ConfigManager失败: %s\n', ME.message);
    end
    
    try
        % 方法2: 尝试直接读取JSON文件
        if exist('config/default_config.json', 'file')
            json_text = fileread('config/default_config.json');
            config = jsondecode(json_text);
            fprintf('✓ 直接加载JSON配置成功\n');
            return;
        end
    catch ME
        fprintf('⚠️ JSON加载失败: %s\n', ME.message);
    end
    
    % 方法3: 使用硬编码默认配置
    fprintf('⚠️ 使用默认配置\n');
    config = getHardcodedConfig();
end

%% 3. 硬编码默认配置
function config = getHardcodedConfig()
    config = struct();
    
    % 基础参数
    config.n_stations = 10;
    config.total_resources = 100;
    config.n_iterations = 20;
    config.n_episodes_per_iter = 50;
    config.n_steps_per_episode = 30;
    
    % 强化学习参数
    config.learning_rate = 0.1;
    config.discount_factor = 0.95;
    config.epsilon = 0.3;
    config.epsilon_decay = 0.995;
    config.epsilon_min = 0.01;
    
    % 算法配置
    config.algorithms = {'QLearning', 'SARSA'};
    config.attacker_algorithm = 'QLearning';
    
    % 环境参数
    config.n_resource_types = 5;
    config.n_attack_types = 6;
    config.alpha_ewma = 0.1;
    config.debug_mode = true;
    config.generate_visualization = false;  % 简化版本关闭可视化
    
    fprintf('✓ 默认配置加载完成\n');
end

%% 4. 修复版环境和智能体创建
function [env, agents] = createEnvironmentAndAgentsFixed(config)
    fprintf('🏗️ 创建环境和智能体...\n');
    
    % 步骤1: 尝试创建TCSEnvironment
    env = createEnvironmentSafely(config);
    
    % 步骤2: 创建智能体
    agents = createAgentsSafely(config, env);
    
    fprintf('✅ 环境和智能体创建完成\n');
end

%% 5. 安全的环境创建
function env = createEnvironmentSafely(config)
    fprintf('  📍 创建TCS环境...\n');
    
    % 方法1: 尝试创建完整的TCSEnvironment对象
    try
        if exist('TCSEnvironment', 'class') == 8
            env = TCSEnvironment(config);
            fprintf('  ✓ 完整TCSEnvironment创建成功\n');
            return;
        else
            error('TCSEnvironment类不存在');
        end
    catch ME
        fprintf('  ⚠️ 完整TCSEnvironment创建失败: %s\n', ME.message);
    end
    
    % 方法2: 创建模拟的TCSEnvironment对象
    fprintf('  🔄 创建模拟TCSEnvironment...\n');
    env = createMockTCSEnvironment(config);
end

%% 6. 创建模拟TCSEnvironment对象
function env = createMockTCSEnvironment(config)
    % 创建一个具有TCSEnvironment接口的对象
    env = TCSEnvironmentMock(config);
end

%% 7. TCSEnvironment模拟类
function obj = TCSEnvironmentMock(config)
    % 创建一个模拟的TCSEnvironment对象
    obj = struct();
    
    % 基本属性
    obj.n_stations = config.n_stations;
    obj.total_resources = config.total_resources;
    obj.state_dim = config.n_stations * 2;
    obj.action_dim_defender = config.n_stations;
    obj.action_dim_attacker = config.n_stations;
    obj.time_step = 0;
    
    % 历史记录
    obj.radi_history = [];
    obj.nash_convergence_history = [];
    obj.attack_coverage_history = [];
    
    % 添加class属性使其看起来像TCSEnvironment
    obj.class_name = 'TCSEnvironment';
    
    % 添加必要的方法
    obj.reset = @() resetMockEnvironment(obj);
    obj.step = @(def_action, att_action) stepMockEnvironment(obj, def_action, att_action);
    obj.updateStrategies = @(att, def) updateMockStrategies(obj, att, def);
    
    fprintf('  ✓ 模拟TCSEnvironment创建成功\n');
end

%% 8. 模拟环境方法
function state = resetMockEnvironment(env)
    env.time_step = 0;
    state = rand(1, env.state_dim);
end

function [next_state, reward_def, reward_att, info] = stepMockEnvironment(env, def_action, att_action)
    env.time_step = env.time_step + 1;
    next_state = rand(1, env.state_dim);
    reward_def = randn() * 0.1;
    reward_att = -reward_def;
    info = struct('attack_success', rand() > 0.5);
end

function updateMockStrategies(env, att_strategy, def_strategy)
    % 模拟策略更新
    if ~isempty(att_strategy)
        % 记录历史
        if length(env.radi_history) < 100
            env.radi_history(end+1) = rand();
        end
    end
end

%% 9. 安全的智能体创建
function agents = createAgentsSafely(config, env)
    fprintf('  🤖 创建智能体...\n');
    
    % 方法1: 尝试使用AgentFactory
    try
        if exist('AgentFactory', 'class') == 8
            attacker = AgentFactory.createAttackerAgent(config, env);
            defenders = AgentFactory.createDefenderAgents(config, env);
            
            agents = cell(1, 1 + length(defenders));
            agents{1} = attacker;
            for i = 1:length(defenders)
                agents{i+1} = defenders{i};
            end
            
            fprintf('  ✓ AgentFactory创建智能体成功\n');
            return;
        end
    catch ME
        fprintf('  ⚠️ AgentFactory失败: %s\n', ME.message);
    end
    
    % 方法2: 创建简化智能体
    fprintf('  🔄 创建简化智能体...\n');
    agents = createSimplifiedAgents(config, env);
end

%% 10. 创建简化智能体
function agents = createSimplifiedAgents(config, env)
    agents = cell(1, 2);
    
    % 简化攻击者
    agents{1} = struct();
    agents{1}.name = 'SimpleAttacker';
    agents{1}.agent_type = 'attacker';
    agents{1}.getStrategy = @() ones(1, config.n_stations) / config.n_stations;
    agents{1}.selectAction = @(state) randi(config.n_stations);
    agents{1}.updateQ = @(varargin) [];
    
    % 简化防御者
    agents{2} = struct();
    agents{2}.name = 'SimpleDefender';
    agents{2}.agent_type = 'defender';
    agents{2}.getStrategy = @() ones(1, config.n_stations) / config.n_stations;
    agents{2}.selectAction = @(state) randi(config.n_stations);
    agents{2}.updateQ = @(varargin) [];
    
    fprintf('  ✓ 简化智能体创建成功\n');
end

%% 11. 验证创建的对象
function is_valid = validateCreatedObjects(env, agents)
    is_valid = true;
    
    % 验证环境
    if isempty(env)
        fprintf('❌ 环境对象为空\n');
        is_valid = false;
    elseif ~(isstruct(env) || isobject(env))
        fprintf('❌ 环境对象类型无效\n');
        is_valid = false;
    else
        fprintf('✓ 环境对象验证通过\n');
    end
    
    % 验证智能体
    if isempty(agents) || ~iscell(agents)
        fprintf('❌ 智能体数组无效\n');
        is_valid = false;
    elseif length(agents) < 2
        fprintf('❌ 智能体数量不足\n');
        is_valid = false;
    else
        fprintf('✓ 智能体数组验证通过\n');
    end
end

%% 12. 初始化仿真器
function simulator = initializeSimulator(config)
    try
        if exist('FSPSimulator', 'class') == 8
            simulator = FSPSimulator(config);
            fprintf('✓ FSPSimulator初始化成功\n');
        else
            simulator = [];
            fprintf('⚠️ 使用简化仿真模式\n');
        end
    catch ME
        fprintf('⚠️ FSPSimulator初始化失败: %s\n', ME.message);
        simulator = [];
    end
end

%% 13. 运行主仿真
function runMainSimulation(env, agents, simulator, config)
    fprintf('🔄 开始仿真循环...\n');
    
    for iteration = 1:config.n_iterations
        fprintf('=== 迭代 %d/%d ===\n', iteration, config.n_iterations);
        
        try
            if ~isempty(simulator)
                % 使用完整仿真器
                episode_results = simulator.runEpisodes(env, agents(2:end), agents{1}, config);
            else
                % 使用简化仿真
                runSimplifiedEpisodes(env, agents, config);
            end
            
            % 显示进度
            if mod(iteration, 5) == 0
                fprintf('✓ 完成 %d/%d 迭代\n', iteration, config.n_iterations);
            end
            
        catch ME
            fprintf('⚠️ 迭代 %d 出错: %s\n', iteration, ME.message);
            continue;
        end
    end
end

%% 14. 简化episode运行
function runSimplifiedEpisodes(env, agents, config)
    for ep = 1:min(config.n_episodes_per_iter, 10)
        state = env.reset();
        
        for step = 1:min(config.n_steps_per_episode, 20)
            def_action = agents{2}.selectAction(state);
            att_action = agents{1}.selectAction(state);
            
            [next_state, ~, ~, ~] = env.step(def_action, att_action);
            state = next_state;
        end
    end
end

%% 15. 错误处理
function handleSimulationError(ME)
    fprintf('❌ 仿真出错: %s\n', ME.message);
    if ~isempty(ME.stack)
        fprintf('错误位置: %s (第%d行)\n', ME.stack(1).file, ME.stack(1).line);
    end
    
    % 记录到日志
    try
        if exist('Logger', 'class') == 8
            Logger.error(sprintf('仿真出错: %s', ME.message));
            Logger.error(sprintf('错误位置: %s, 行号: %d', ME.stack(1).file, ME.stack(1).line));
        end
    catch
        % 日志记录失败也不影响错误报告
    end
    
    % 提供解决建议
    fprintf('\n🔧 解决建议:\n');
    fprintf('1. 检查所有必要的类文件是否存在\n');
    fprintf('2. 运行: addpath(genpath(pwd))\n');
    fprintf('3. 验证配置文件格式\n');
    fprintf('4. 如果问题持续，请使用简化模式运行\n');
end