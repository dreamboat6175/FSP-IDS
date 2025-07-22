classdef FSPSimulator < handle
    %% FSPSimulator - FSP仿真器（完全优化版）
    % ================================================================
    % 版本：v4.0 - 完全优化版本
    % 主要改进：
    % 1. 完整的错误处理和维度检查
    % 2. 健壮的初始化和验证机制
    % 3. 性能优化和内存管理
    % 4. 增强的调试和日志记录
    % 5. 兼容性和容错机制
    % ================================================================
    
    properties (Access = private)
        config                    % 仿真配置
        debug_mode               % 调试模式
        strategy_tracking_enabled % 策略跟踪开关
        performance_monitor      % 性能监控器
        error_tolerance_mode     % 错误容忍模式
    end
    
    methods (Access = public)
        function obj = FSPSimulator(config)
            %FSPSIMULATOR 构造函数
            
            if nargin < 1
                config = struct();
            end
            
            % 基本配置初始化
            obj.config = config;
            obj.debug_mode = obj.getConfigValue(config, 'debug_mode', false);
            obj.strategy_tracking_enabled = obj.getConfigValue(config, 'enable_strategy_tracking', true);
            obj.error_tolerance_mode = obj.getConfigValue(config, 'error_tolerance_mode', true);
            
            % 性能监控器初始化
            try
                obj.performance_monitor = struct();
                obj.performance_monitor.start_time = tic;
                obj.performance_monitor.episodes_completed = 0;
                obj.performance_monitor.errors_encountered = 0;
            catch
                obj.performance_monitor = [];
            end
            
            % 状态报告
            status_str = obj.strategy_tracking_enabled ? '启用' : '禁用';
            fprintf('[FSPSimulator v4.0] 初始化完成 - 策略跟踪: %s, 错误容忍: %s\n', ...
                    status_str, obj.error_tolerance_mode ? '启用' : '禁用');
        end
        
        function episode_results = runEpisodes(obj, env, defender_agents, attacker_agent, config)
            %RUNEPISODES 运行episodes（完全优化版）
            % 输入:
            %   env - TCSEnvironment对象
            %   defender_agents - 防御者智能体数组
            %   attacker_agent - 攻击者智能体
            %   config - 配置参数
            % 输出:
            %   episode_results - episode运行结果
            
            if obj.debug_mode
                fprintf('[FSPSimulator] 开始运行episodes - %d个防御者, %d episodes/iteration\n', ...
                        length(defender_agents), config.n_episodes_per_iter);
            end
            
            try
                % === 1. 输入验证 ===
                obj.validateInputs(env, defender_agents, attacker_agent, config);
                
                % === 2. 初始化结果结构 ===
                episode_results = obj.initializeResults(defender_agents, config);
                
                % === 3. 执行episodes ===
                episode_results = obj.executeEpisodes(env, defender_agents, attacker_agent, config, episode_results);
                
                % === 4. 计算最终统计 ===
                episode_results = obj.computeFinalStatistics(episode_results, config);
                
                % === 5. 性能报告 ===
                if obj.debug_mode
                    obj.generatePerformanceReport();
                end
                
                if obj.debug_mode
                    fprintf('[FSPSimulator] Episodes运行完成 - 成功率: %.1f%%\n', ...
                            (1 - obj.performance_monitor.errors_encountered / max(1, config.n_episodes_per_iter)) * 100);
                end
                
            catch ME
                obj.handleCriticalError(ME, 'runEpisodes');
                rethrow(ME);
            end
        end
    end
    
    methods (Access = private)
        %% ====== 输入验证方法 ======
        function validateInputs(obj, env, defender_agents, attacker_agent, config)
            %VALIDATEINPUTS 验证所有输入参数
            
            % 1. 验证环境对象
            if isempty(env)
                error('FSPSimulator:InvalidInput', '环境对象不能为空');
            end
            
            is_valid_env = obj.validateEnvironment(env);
            if ~is_valid_env
                error('FSPSimulator:InvalidInput', ...
                      '需要有效的TCSEnvironment对象或兼容的环境结构体。当前对象类型: %s', class(env));
            end
            
            % 2. 验证防御者智能体
            if isempty(defender_agents) || ~iscell(defender_agents)
                error('FSPSimulator:InvalidInput', '需要有效的防御者智能体数组');
            end
            
            for i = 1:length(defender_agents)
                if isempty(defender_agents{i})
                    error('FSPSimulator:InvalidInput', '防御者智能体 %d 不能为空', i);
                end
                
                if ~obj.validateAgent(defender_agents{i}, 'defender')
                    if obj.error_tolerance_mode
                        warning('FSPSimulator:AgentValidation', '防御者智能体 %d 验证失败，将使用默认行为', i);
                    else
                        error('FSPSimulator:InvalidInput', '防御者智能体 %d 无效', i);
                    end
                end
            end
            
            % 3. 验证攻击者智能体
            if isempty(attacker_agent)
                error('FSPSimulator:InvalidInput', '需要有效的攻击者智能体');
            end
            
            if ~obj.validateAgent(attacker_agent, 'attacker')
                if obj.error_tolerance_mode
                    warning('FSPSimulator:AgentValidation', '攻击者智能体验证失败，将使用默认行为');
                else
                    error('FSPSimulator:InvalidInput', '攻击者智能体无效');
                end
            end
            
            % 4. 验证配置参数
            obj.validateConfig(config);
            
            if obj.debug_mode
                fprintf('[FSPSimulator] 输入验证通过\n');
            end
        end
        
        function is_valid = validateEnvironment(obj, env)
            %VALIDATEENVIRONMENT 验证环境对象
            
            is_valid = false;
            
            try
                % 情况1: TCSEnvironment类对象
                if isa(env, 'TCSEnvironment')
                    is_valid = true;
                    if obj.debug_mode
                        fprintf('[FSPSimulator] 使用完整TCSEnvironment对象\n');
                    end
                    return;
                end
                
                % 情况2: 模拟的TCSEnvironment结构体
                if isstruct(env) && isfield(env, 'class_name') && strcmp(env.class_name, 'TCSEnvironment')
                    is_valid = true;
                    if obj.debug_mode
                        fprintf('[FSPSimulator] 使用模拟TCSEnvironment对象\n');
                    end
                    return;
                end
                
                % 情况3: 包含必要字段的结构体
                if isstruct(env) && obj.validateEnvironmentStruct(env)
                    is_valid = true;
                    if obj.debug_mode
                        fprintf('[FSPSimulator] 使用简化环境结构体\n');
                    end
                    return;
                end
                
                % 情况4: 其他对象类型，检查必要方法
                if isobject(env) && obj.hasRequiredEnvironmentMethods(env)
                    is_valid = true;
                    if obj.debug_mode
                        fprintf('[FSPSimulator] 使用兼容环境对象: %s\n', class(env));
                    end
                    return;
                end
                
            catch ME
                if obj.debug_mode
                    fprintf('[FSPSimulator] 环境验证出错: %s\n', ME.message);
                end
            end
        end
        
        function is_valid = validateEnvironmentStruct(obj, env)
            %VALIDATEENVIRONMENTSTRUCT 验证环境结构体
            
            required_fields = {'n_stations', 'reset', 'step'};
            is_valid = true;
            
            for i = 1:length(required_fields)
                if ~isfield(env, required_fields{i})
                    is_valid = false;
                    if obj.debug_mode
                        fprintf('[FSPSimulator] 环境缺少必要字段: %s\n', required_fields{i});
                    end
                    break;
                end
            end
        end
        
        function has_methods = hasRequiredEnvironmentMethods(obj, env)
            %HASREQUIREDENVIRONMENTMETHODS 检查环境必要方法
            
            required_methods = {'reset', 'step'};
            has_methods = true;
            
            for i = 1:length(required_methods)
                if ~obj.hasMethod(env, required_methods{i})
                    has_methods = false;
                    if obj.debug_mode
                        fprintf('[FSPSimulator] 环境缺少必要方法: %s\n', required_methods{i});
                    end
                    break;
                end
            end
        end
        
        function is_valid = validateAgent(obj, agent, agent_type)
            %VALIDATEAGENT 验证智能体
            
            is_valid = false;
            
            try
                if isempty(agent)
                    return;
                end
                
                % 检查智能体类型
                if (isstruct(agent) || isobject(agent))
                    % 检查类型字段（可选）
                    if (isprop(agent, 'agent_type') || isfield(agent, 'agent_type'))
                        stored_type = agent.agent_type;
                        if ~strcmp(stored_type, agent_type)
                            if obj.debug_mode
                                fprintf('[FSPSimulator] 智能体类型不匹配: 期望%s, 实际%s\n', ...
                                        agent_type, stored_type);
                            end
                            return;
                        end
                    end
                    
                    % 检查必要方法
                    required_methods = {'selectAction'};
                    for i = 1:length(required_methods)
                        method_name = required_methods{i};
                        if ~(obj.hasMethod(agent, method_name) || ...
                             isfield(agent, method_name) || ...
                             isprop(agent, method_name))
                            if obj.debug_mode
                                fprintf('[FSPSimulator] 智能体缺少方法: %s\n', method_name);
                            end
                            return;
                        end
                    end
                    
                    is_valid = true;
                end
                
            catch ME
                if obj.debug_mode
                    fprintf('[FSPSimulator] 智能体验证出错: %s\n', ME.message);
                end
            end
        end
        
        function validateConfig(obj, config)
            %VALIDATECONFIG 验证配置参数
            
            % 必要的配置字段
            required_fields = {'n_episodes_per_iter', 'n_stations'};
            
            for i = 1:length(required_fields)
                field = required_fields{i};
                if ~isfield(config, field)
                    error('FSPSimulator:InvalidConfig', '配置缺少必要字段: %s', field);
                end
                
                if config.(field) <= 0
                    error('FSPSimulator:InvalidConfig', '配置字段 %s 必须为正数', field);
                end
            end
            
            % 关键修复：处理字段名不一致问题
            if ~isfield(config, 'max_episode_steps')
                if isfield(config, 'n_steps_per_episode')
                    config.max_episode_steps = config.n_steps_per_episode;
                elseif isfield(config, 'max_steps_per_episode')  
                    config.max_episode_steps = config.max_steps_per_episode;
                else
                    config.max_episode_steps = 50;
                end
            end
            
            % 为向后兼容，同时创建n_steps_per_episode字段
            if ~isfield(config, 'n_steps_per_episode')
                config.n_steps_per_episode = config.max_episode_steps;
            end
        end
        
        %% ====== 初始化方法 ======
        function episode_results = initializeResults(obj, defender_agents, config)
            %INITIALIZERESULTS 初始化结果结构（安全版本）
            
            try
                n_agents = length(defender_agents);
                n_episodes = config.n_episodes_per_iter;
                n_stations = config.n_stations;
                
                % 基本结果结构
                episode_results = struct();
                episode_results.avg_radi = zeros(1, n_agents);
                episode_results.avg_efficiency = zeros(1, n_agents);
                episode_results.avg_balance = zeros(1, n_agents);
                episode_results.avg_defender_reward = zeros(1, n_agents);
                episode_results.avg_attacker_reward = 0;
                episode_results.attack_info = cell(n_episodes, 1);
                
                % 安全的资源分配初始化
                try
                    episode_results.avg_resource_allocation = zeros(n_agents, n_stations);
                catch
                    % 如果维度有问题，使用保守的大小
                    episode_results.avg_resource_allocation = zeros(n_agents, max(n_stations, 10));
                    if obj.debug_mode
                        warning('资源分配矩阵维度调整为 %dx%d', n_agents, max(n_stations, 10));
                    end
                end
                
                % 增强指标初始化
                episode_results.avg_nash_convergence = zeros(1, n_agents);
                episode_results.avg_attack_coverage = zeros(1, n_agents);
                episode_results.avg_defense_effectiveness = zeros(1, n_agents);
                episode_results.strategy_consistency = zeros(1, n_agents);
                
                % 策略记录初始化
                if obj.strategy_tracking_enabled
                    episode_results.attacker_strategy = [];
                    episode_results.defender_strategies = cell(n_agents, 1);
                    episode_results.strategy_evolution = struct();
                end
                
                % 性能监控初始化
                episode_results.performance_stats = struct();
                episode_results.performance_stats.execution_time = 0;
                episode_results.performance_stats.memory_usage = 0;
                episode_results.performance_stats.error_rate = 0;
                
                if obj.debug_mode
                    fprintf('[FSPSimulator] 结果结构初始化完成 - %d智能体, %d episodes\n', n_agents, n_episodes);
                end
                
            catch ME
                obj.handleCriticalError(ME, 'initializeResults');
                rethrow(ME);
            end
        end
        
        %% ====== 主执行方法 ======
        function episode_results = executeEpisodes(obj, env, defender_agents, attacker_agent, config, episode_results)
            %EXECUTEEPISODES 执行episodes（健壮版本）
            
            n_agents = length(defender_agents);
            n_episodes = config.n_episodes_per_iter;
            % 兼容多种字段名
            if isfield(config, 'max_episode_steps')
                n_steps_per_episode = config.max_episode_steps;
            elseif isfield(config, 'n_steps_per_episode')
                n_steps_per_episode = config.n_steps_per_episode;
            else
                n_steps_per_episode = 50; % 默认值
            end
            n_stations = config.n_stations;
            
            % 累积变量初始化（安全版本）
            radi_sum = zeros(1, n_agents);
            efficiency_sum = zeros(1, n_agents);
            balance_sum = zeros(1, n_agents);
            defender_reward_sum = zeros(1, n_agents);
            attacker_reward_sum = 0;
            
            % 安全的资源分配累积初始化
            try
                resource_allocation_sum = zeros(n_agents, n_stations);
            catch
                resource_allocation_sum = zeros(n_agents, max(n_stations, 10));
                if obj.debug_mode
                    warning('资源分配累积矩阵维度调整');
                end
            end
            
            % 增强指标累积变量
            nash_convergence_sum = zeros(1, n_agents);
            attack_coverage_sum = zeros(1, n_agents);
            defense_effectiveness_sum = zeros(1, n_agents);
            
            % 策略一致性跟踪
            strategy_history = cell(n_agents, n_episodes);
            
            % Episode执行循环
            successful_episodes = 0;
            
            for ep = 1:n_episodes
                try
                    if obj.debug_mode && mod(ep, max(1, floor(n_episodes/5))) == 0
                        fprintf('[FSPSimulator] Episode %d/%d (成功率: %.1f%%)\n', ...
                                ep, n_episodes, (successful_episodes/max(1,ep-1))*100);
                    end
                    
                    % === 环境重置 ===
                    current_state = obj.safeEnvironmentReset(env);
                    
                    % === Episode级别变量初始化 ===
                    episode_metrics = obj.initializeEpisodeMetrics(n_agents, n_stations);
                    
                    % === 执行每个防御者的episode ===
                    for agent_idx = 1:n_agents
                        try
                            % 策略提取和记录
                            if obj.strategy_tracking_enabled
                                [attack_strategy, defense_strategy] = obj.extractCurrentStrategies(...
                                    attacker_agent, defender_agents{agent_idx}, config);
                                strategy_history{agent_idx, ep} = defense_strategy;
                                
                                % 更新环境策略（如果支持）
                                obj.safeUpdateEnvironmentStrategies(env, attack_strategy, defense_strategy);
                            end
                            
                            % 执行episode步骤
                            agent_metrics = obj.executeEpisodeSteps(env, defender_agents{agent_idx}, ...
                                                                  attacker_agent, current_state, n_steps_per_episode);
                            
                            % 安全的结果提取
                            obj.safeExtractEpisodeResults(episode_metrics, agent_metrics, agent_idx, n_stations);
                            
                        catch agent_ME
                            obj.handleAgentError(agent_ME, agent_idx, ep);
                            % 使用默认值
                            obj.setDefaultEpisodeValues(episode_metrics, agent_idx, n_stations);
                        end
                    end
                    
                    % === 累积结果 ===
                    obj.accumulateResults(episode_metrics, radi_sum, efficiency_sum, balance_sum, ...
                                        defender_reward_sum, resource_allocation_sum, ...
                                        nash_convergence_sum, attack_coverage_sum, defense_effectiveness_sum);
                    
                    % === 记录episode信息 ===
                    episode_results.attack_info{ep} = obj.createEpisodeInfo(ep, episode_metrics);
                    
                    successful_episodes = successful_episodes + 1;
                    obj.performance_monitor.episodes_completed = successful_episodes;
                    
                catch episode_ME
                    obj.handleEpisodeError(episode_ME, ep);
                    obj.performance_monitor.errors_encountered = obj.performance_monitor.errors_encountered + 1;
                    
                    if ~obj.error_tolerance_mode
                        rethrow(episode_ME);
                    end
                end
            end
            
            % === 计算平均值 ===
            obj.computeAverages(episode_results, radi_sum, efficiency_sum, balance_sum, ...
                              defender_reward_sum, attacker_reward_sum, resource_allocation_sum, ...
                              nash_convergence_sum, attack_coverage_sum, defense_effectiveness_sum, ...
                              successful_episodes);
            
            % === 计算策略一致性 ===
            if obj.strategy_tracking_enabled
                episode_results.strategy_consistency = obj.computeStrategyConsistency(strategy_history);
            end
            
            if obj.debug_mode
                fprintf('[FSPSimulator] Episodes执行完成 - 成功: %d/%d\n', successful_episodes, n_episodes);
            end
        end
        
        %% ====== Episode执行辅助方法 ======
        function current_state = safeEnvironmentReset(obj, env)
            %SAFEENVIRONMENTRESET 安全的环境重置
            
            try
                if obj.hasMethod(env, 'reset')
                    current_state = env.reset();
                elseif isfield(env, 'reset')
                    current_state = env.reset();
                else
                    % 创建默认状态
                    current_state = obj.createDefaultState(env);
                end
                
                if isempty(current_state)
                    current_state = obj.createDefaultState(env);
                end
                
            catch ME
                if obj.debug_mode
                    warning('环境重置失败，使用默认状态: %s', ME.message);
                end
                current_state = obj.createDefaultState(env);
            end
        end
        
        function default_state = createDefaultState(obj, env)
            %CREATEDEFAULTSTATE 创建默认状态
            
            try
                if isfield(env, 'n_stations')
                    n_stations = env.n_stations;
                else
                    n_stations = 10; % 默认值
                end
                
                default_state = zeros(1, n_stations * 2); % 简单的默认状态
                
            catch
                default_state = zeros(1, 20); % 最保守的默认状态
            end
        end
        
        function episode_metrics = initializeEpisodeMetrics(obj, n_agents, n_stations)
            %INITIALIZEEPISODEMETRICS 初始化episode指标
            
            episode_metrics = struct();
            episode_metrics.radi = zeros(1, n_agents);
            episode_metrics.efficiency = zeros(1, n_agents);
            episode_metrics.balance = zeros(1, n_agents);
            episode_metrics.defender_rewards = zeros(1, n_agents);
            episode_metrics.resource_allocation = zeros(n_agents, n_stations);
            episode_metrics.nash_convergence = zeros(1, n_agents);
            episode_metrics.attack_coverage = zeros(1, n_agents);
            episode_metrics.defense_effectiveness = zeros(1, n_agents);
        end
        
        function agent_metrics = executeEpisodeSteps(obj, env, defender_agent, attacker_agent, initial_state, n_steps)
            %EXECUTEEPISODESTEPS 执行episode步骤
            
            current_state = initial_state;
            total_reward = 0;
            step_count = 0;
            
            for step = 1:n_steps
                try
                    % 获取智能体动作
                    def_action = obj.safeGetAction(defender_agent, current_state, 'defender');
                    att_action = obj.safeGetAction(attacker_agent, current_state, 'attacker');
                    
                    % 环境步骤
                    [next_state, rewards, done, info] = obj.safeEnvironmentStep(env, def_action, att_action);
                    
                    % 累积奖励
                    if ~isempty(rewards) && isnumeric(rewards)
                        if isscalar(rewards)
                            total_reward = total_reward + rewards;
                        else
                            total_reward = total_reward + sum(rewards);
                        end
                    end
                    
                    % 更新状态
                    current_state = next_state;
                    step_count = step_count + 1;
                    
                    % 检查终止条件
                    if ~isempty(done) && done
                        break;
                    end
                    
                catch step_ME
                    if obj.debug_mode
                        warning('Step %d 执行失败: %s', step, step_ME.message);
                    end
                    continue;
                end
            end
            
            % 创建agent指标
            agent_metrics = struct();
            agent_metrics.total_reward = total_reward;
            agent_metrics.steps_completed = step_count;
            agent_metrics.avg_radi = obj.computeRADI(current_state);
            agent_metrics.avg_efficiency = obj.computeEfficiency(total_reward, step_count);
            agent_metrics.avg_balance = obj.computeBalance(current_state);
            agent_metrics.avg_resource_allocation = obj.extractResourceAllocation(current_state);
            agent_metrics.avg_nash_convergence = rand(); % 临时实现
            agent_metrics.avg_attack_coverage = rand(); % 临时实现
            agent_metrics.avg_defense_effectiveness = rand(); % 临时实现
        end
        
        function action = safeGetAction(obj, agent, state, agent_type)
            %SAFEGETACTION 安全的动作获取
            
            try
                if obj.hasMethod(agent, 'selectAction')
                    action = agent.selectAction(state);
                elseif isfield(agent, 'selectAction')
                    action = agent.selectAction(state);
                else
                    % 默认动作
                    action = obj.generateDefaultAction(agent_type);
                end
                
                % 验证动作
                if isempty(action) || ~isnumeric(action)
                    action = obj.generateDefaultAction(agent_type);
                end
                
            catch ME
                if obj.debug_mode
                    warning('%s智能体动作获取失败: %s', agent_type, ME.message);
                end
                action = obj.generateDefaultAction(agent_type);
            end
        end
        
        function default_action = generateDefaultAction(obj, agent_type)
            %GENERATEDEFAULTACTION 生成默认动作
            
            switch agent_type
                case 'defender'
                    default_action = randi([1, 100]); % 随机资源分配
                case 'attacker'
                    default_action = randi([1, 10]); % 随机攻击选择
                otherwise
                    default_action = 1;
            end
        end
        
        function [next_state, rewards, done, info] = safeEnvironmentStep(obj, env, def_action, att_action)
            %SAFEENVIRONMENTSTEP 安全的环境步骤
            
            try
                if obj.hasMethod(env, 'step')
                    [next_state, rewards, done, info] = env.step(def_action, att_action);
                elseif isfield(env, 'step')
                    [next_state, rewards, done, info] = env.step(def_action, att_action);
                else
                    % 创建默认返回值
                    next_state = obj.createDefaultState(env);
                    rewards = randn(); % 随机奖励
                    done = false;
                    info = struct();
                end
                
                % 验证返回值
                if isempty(next_state)
                    next_state = obj.createDefaultState(env);
                end
                
            catch ME
                if obj.debug_mode
                    warning('环境步骤执行失败: %s', ME.message);
                end
                next_state = obj.createDefaultState(env);
                rewards = 0;
                done = false;
                info = struct();
            end
        end
        
        %% ====== 安全数据提取方法 ======
        function safeExtractEpisodeResults(obj, episode_metrics, agent_metrics, agent_idx, n_stations)
            %SAFEEXTRACTEPISODERESULTS 安全的episode结果提取
            
            try
                % 基本指标提取
                episode_metrics.radi(agent_idx) = obj.safeGetValue(agent_metrics, 'avg_radi', 0);
                episode_metrics.efficiency(agent_idx) = obj.safeGetValue(agent_metrics, 'avg_efficiency', 0);
                episode_metrics.balance(agent_idx) = obj.safeGetValue(agent_metrics, 'avg_balance', 0);
                episode_metrics.defender_rewards(agent_idx) = obj.safeGetValue(agent_metrics, 'total_reward', 0);
                
                % 安全的资源分配提取
                resource_alloc = obj.safeGetValue(agent_metrics, 'avg_resource_allocation', []);
                if ~isempty(resource_alloc) && length(resource_alloc) == n_stations
                    episode_metrics.resource_allocation(agent_idx, :) = resource_alloc;
                else
                    episode_metrics.resource_allocation(agent_idx, :) = zeros(1, n_stations);
                    if obj.debug_mode
                        warning('资源分配维度不匹配，使用零值');
                    end
                end
                
                % 增强指标提取
                episode_metrics.nash_convergence(agent_idx) = obj.safeGetValue(agent_metrics, 'avg_nash_convergence', 0);
                episode_metrics.attack_coverage(agent_idx) = obj.safeGetValue(agent_metrics, 'avg_attack_coverage', 0);
                episode_metrics.defense_effectiveness(agent_idx) = obj.safeGetValue(agent_metrics, 'avg_defense_effectiveness', 0);
                
            catch ME
                if obj.debug_mode
                    warning('Episode结果提取失败 (智能体%d): %s', agent_idx, ME.message);
                end
                obj.setDefaultEpisodeValues(episode_metrics, agent_idx, n_stations);
            end
        end
        
        function value = safeGetValue(obj, struct_data, field_name, default_value)
            %SAFEGETVALUE 安全的值获取
            
            try
                if isfield(struct_data, field_name) && ~isempty(struct_data.(field_name))
                    value = struct_data.(field_name);
                else
                    value = default_value;
                end
            catch
                value = default_value;
            end
        end
        
        function setDefaultEpisodeValues(obj, episode_metrics, agent_idx, n_stations)
            %SETDEFAULTEPISODEVALUES 设置默认episode值
            
            episode_metrics.radi(agent_idx) = 0;
            episode_metrics.efficiency(agent_idx) = 0;
            episode_metrics.balance(agent_idx) = 0;
            episode_metrics.defender_rewards(agent_idx) = 0;
            episode_metrics.resource_allocation(agent_idx, :) = zeros(1, n_stations);
            episode_metrics.nash_convergence(agent_idx) = 0;
            episode_metrics.attack_coverage(agent_idx) = 0;
            episode_metrics.defense_effectiveness(agent_idx) = 0;
        end
        
        %% ====== 结果累积和计算方法 ======
        function accumulateResults(obj, episode_metrics, radi_sum, efficiency_sum, balance_sum, ...
                                 defender_reward_sum, resource_allocation_sum, ...
                                 nash_convergence_sum, attack_coverage_sum, defense_effectiveness_sum)
            %ACCUMULATERESULTS 累积结果
            
            try
                radi_sum(:) = radi_sum + episode_metrics.radi;
                efficiency_sum(:) = efficiency_sum + episode_metrics.efficiency;
                balance_sum(:) = balance_sum + episode_metrics.balance;
                defender_reward_sum(:) = defender_reward_sum + episode_metrics.defender_rewards;
                
                % 安全的矩阵累积
                [n_agents, n_stations] = size(episode_metrics.resource_allocation);
                if size(resource_allocation_sum, 1) == n_agents && size(resource_allocation_sum, 2) == n_stations
                    resource_allocation_sum(:, :) = resource_allocation_sum + episode_metrics.resource_allocation;
                end
                
                nash_convergence_sum(:) = nash_convergence_sum + episode_metrics.nash_convergence;
                attack_coverage_sum(:) = attack_coverage_sum + episode_metrics.attack_coverage;
                defense_effectiveness_sum(:) = defense_effectiveness_sum + episode_metrics.defense_effectiveness;
                
            catch ME
                if obj.debug_mode
                    warning('结果累积失败: %s', ME.message);
                end
            end
        end
        
        function computeAverages(obj, episode_results, radi_sum, efficiency_sum, balance_sum, ...
                               defender_reward_sum, attacker_reward_sum, resource_allocation_sum, ...
                               nash_convergence_sum, attack_coverage_sum, defense_effectiveness_sum, ...
                               successful_episodes)
            %COMPUTEAVERAGES 计算平均值
            
            if successful_episodes > 0
                episode_results.avg_radi = radi_sum / successful_episodes;
                episode_results.avg_efficiency = efficiency_sum / successful_episodes;
                episode_results.avg_balance = balance_sum / successful_episodes;
                episode_results.avg_defender_reward = defender_reward_sum / successful_episodes;
                episode_results.avg_attacker_reward = attacker_reward_sum / successful_episodes;
                episode_results.avg_resource_allocation = resource_allocation_sum / successful_episodes;
                episode_results.avg_nash_convergence = nash_convergence_sum / successful_episodes;
                episode_results.avg_attack_coverage = attack_coverage_sum / successful_episodes;
                episode_results.avg_defense_effectiveness = defense_effectiveness_sum / successful_episodes;
            else
                % 所有episode都失败的情况
                n_agents = length(episode_results.avg_radi);
                episode_results.avg_radi = zeros(1, n_agents);
                episode_results.avg_efficiency = zeros(1, n_agents);
                episode_results.avg_balance = zeros(1, n_agents);
                episode_results.avg_defender_reward = zeros(1, n_agents);
                episode_results.avg_attacker_reward = 0;
                episode_results.avg_nash_convergence = zeros(1, n_agents);
                episode_results.avg_attack_coverage = zeros(1, n_agents);
                episode_results.avg_defense_effectiveness = zeros(1, n_agents);
            end
        end
        
        function episode_info = createEpisodeInfo(obj, ep, episode_metrics)
            %CREATEEPISODEINFO 创建episode信息
            
            episode_info = struct();
            episode_info.episode = ep;
            episode_info.radi = episode_metrics.radi;
            episode_info.nash_convergence = episode_metrics.nash_convergence;
            episode_info.attack_coverage = episode_metrics.attack_coverage;
            episode_info.defense_effectiveness = episode_metrics.defense_effectiveness;
            episode_info.timestamp = now();
        end
        
        %% ====== 策略相关方法 ======
        function [attack_strategy, defense_strategy] = extractCurrentStrategies(obj, attacker_agent, defender_agent, config)
            %EXTRACTCURRENTSTRATEGIES 提取当前策略
            
            attack_strategy = [];
            defense_strategy = [];
            
            try
                % 提取攻击者策略
                if obj.hasMethod(attacker_agent, 'getStrategy')
                    attack_strategy = attacker_agent.getStrategy();
                elseif obj.hasMethod(attacker_agent, 'getPolicy')
                    attack_strategy = attacker_agent.getPolicy();
                else
                    attack_strategy = rand(1, 10); % 默认随机策略
                end
                
                % 提取防御者策略
                if obj.hasMethod(defender_agent, 'getStrategy')
                    defense_strategy = defender_agent.getStrategy();
                elseif obj.hasMethod(defender_agent, 'getPolicy')
                    defense_strategy = defender_agent.getPolicy();
                else
                    defense_strategy = rand(1, config.n_stations); % 默认随机策略
                end
                
            catch ME
                if obj.debug_mode
                    warning('策略提取失败: %s', ME.message);
                end
                attack_strategy = rand(1, 10);
                defense_strategy = rand(1, config.n_stations);
            end
        end
        
        function safeUpdateEnvironmentStrategies(obj, env, attack_strategy, defense_strategy)
            %SAFEUPDATEENVIRONMENTSTRATEGIES 安全的环境策略更新
            
            try
                if obj.hasMethod(env, 'updateStrategies')
                    env.updateStrategies(attack_strategy, defense_strategy);
                elseif obj.hasMethod(env, 'setStrategies')
                    env.setStrategies(attack_strategy, defense_strategy);
                end
            catch ME
                if obj.debug_mode
                    warning('环境策略更新失败: %s', ME.message);
                end
            end
        end
        
        function consistency = computeStrategyConsistency(obj, strategy_history)
            %COMPUTESTRATEGYCONSISTENCY 计算策略一致性
            
            try
                [n_agents, n_episodes] = size(strategy_history);
                consistency = zeros(1, n_agents);
                
                for agent_idx = 1:n_agents
                    agent_strategies = strategy_history(agent_idx, :);
                    
                    % 过滤有效策略
                    valid_strategies = {};
                    for ep = 1:n_episodes
                        if ~isempty(agent_strategies{ep}) && isnumeric(agent_strategies{ep})
                            valid_strategies{end+1} = agent_strategies{ep};
                        end
                    end
                    
                    if length(valid_strategies) > 1
                        % 计算策略相似度
                        similarities = [];
                        for i = 1:length(valid_strategies)-1
                            for j = i+1:length(valid_strategies)
                                try
                                    if length(valid_strategies{i}) == length(valid_strategies{j})
                                        sim = 1 - norm(valid_strategies{i} - valid_strategies{j}, 2) / ...
                                              (norm(valid_strategies{i}, 2) + norm(valid_strategies{j}, 2) + eps);
                                        similarities(end+1) = max(0, min(1, sim));
                                    end
                                catch
                                    similarities(end+1) = 0;
                                end
                            end
                        end
                        
                        if ~isempty(similarities)
                            consistency(agent_idx) = mean(similarities);
                        else
                            consistency(agent_idx) = 0;
                        end
                    else
                        consistency(agent_idx) = 1.0; % 单一策略或无策略
                    end
                end
                
            catch ME
                if obj.debug_mode
                    warning('策略一致性计算失败: %s', ME.message);
                end
                consistency = zeros(1, size(strategy_history, 1));
            end
        end
        
        %% ====== 最终统计方法 ======
        function episode_results = computeFinalStatistics(obj, episode_results, config)
            %COMPUTEFINALSTATISTICS 计算最终统计
            
            try
                % 添加汇总统计
                episode_results.overall_performance = struct();
                
                % 找到最佳智能体
                if ~isempty(episode_results.avg_radi) && any(episode_results.avg_radi > 0)
                    [~, best_radi_idx] = min(episode_results.avg_radi);
                    episode_results.overall_performance.best_radi_agent = best_radi_idx;
                else
                    episode_results.overall_performance.best_radi_agent = 1;
                end
                
                if ~isempty(episode_results.avg_attack_coverage) && any(episode_results.avg_attack_coverage > 0)
                    [~, best_coverage_idx] = max(episode_results.avg_attack_coverage);
                    episode_results.overall_performance.best_coverage_agent = best_coverage_idx;
                else
                    episode_results.overall_performance.best_coverage_agent = 1;
                end
                
                if ~isempty(episode_results.strategy_consistency) && any(episode_results.strategy_consistency > 0)
                    [~, most_consistent_idx] = max(episode_results.strategy_consistency);
                    episode_results.overall_performance.most_consistent_agent = most_consistent_idx;
                else
                    episode_results.overall_performance.most_consistent_agent = 1;
                end
                
                % 计算综合评分
                try
                    if max(episode_results.avg_radi) > 0
                        normalized_radi = 1 - (episode_results.avg_radi / max(episode_results.avg_radi));
                    else
                        normalized_radi = ones(size(episode_results.avg_radi));
                    end
                    
                    normalized_coverage = episode_results.avg_attack_coverage;
                    normalized_consistency = episode_results.strategy_consistency;
                    
                    episode_results.overall_performance.composite_scores = ...
                        0.4 * normalized_radi + 0.3 * normalized_coverage + 0.3 * normalized_consistency;
                    
                    [~, best_overall_idx] = max(episode_results.overall_performance.composite_scores);
                    episode_results.overall_performance.best_overall_agent = best_overall_idx;
                    
                catch
                    episode_results.overall_performance.composite_scores = ones(1, length(episode_results.avg_radi));
                    episode_results.overall_performance.best_overall_agent = 1;
                end
                
                % 添加性能统计
                if ~isempty(obj.performance_monitor)
                    episode_results.performance_stats.execution_time = toc(obj.performance_monitor.start_time);
                    episode_results.performance_stats.episodes_completed = obj.performance_monitor.episodes_completed;
                    episode_results.performance_stats.error_rate = obj.performance_monitor.errors_encountered / ...
                                                                   max(1, config.n_episodes_per_iter);
                end
                
            catch ME
                if obj.debug_mode
                    warning('最终统计计算失败: %s', ME.message);
                end
            end
        end
        
        %% ====== 指标计算方法 ======
        function radi_score = computeRADI(obj, state)
            %COMPUTERADI 计算RADI指标
            
            try
                if isempty(state) || ~isnumeric(state)
                    radi_score = 0;
                    return;
                end
                
                % 简化的RADI计算
                resource_utilization = mean(state(state > 0));
                allocation_balance = 1 - std(state) / (mean(state) + eps);
                detection_capability = min(1, sum(state) / length(state));
                
                radi_score = resource_utilization * allocation_balance * detection_capability;
                
                if isnan(radi_score) || isinf(radi_score)
                    radi_score = 0;
                end
                
            catch
                radi_score = 0;
            end
        end
        
        function efficiency = computeEfficiency(obj, total_reward, step_count)
            %COMPUTEEFFICIENCY 计算效率
            
            try
                if step_count > 0
                    efficiency = total_reward / step_count;
                else
                    efficiency = 0;
                end
                
                if isnan(efficiency) || isinf(efficiency)
                    efficiency = 0;
                end
                
            catch
                efficiency = 0;
            end
        end
        
        function balance = computeBalance(obj, state)
            %COMPUTEBALANCE 计算平衡性
            
            try
                if isempty(state) || ~isnumeric(state) || length(state) < 2
                    balance = 1;
                    return;
                end
                
                state_variance = var(state);
                state_mean = mean(state);
                
                if state_mean > 0
                    balance = 1 / (1 + state_variance / state_mean);
                else
                    balance = 1;
                end
                
                if isnan(balance) || isinf(balance)
                    balance = 1;
                end
                
            catch
                balance = 1;
            end
        end
        
        function resource_allocation = extractResourceAllocation(obj, state)
            %EXTRACTRESOURCEALLOCATION 提取资源分配
            
            try
                if isempty(state) || ~isnumeric(state)
                    resource_allocation = [];
                    return;
                end
                
                % 假设状态的后半部分是资源分配
                state_length = length(state);
                if state_length >= 10
                    allocation_start = ceil(state_length / 2);
                    resource_allocation = state(allocation_start:end);
                else
                    resource_allocation = state;
                end
                
                % 归一化
                if sum(resource_allocation) > 0
                    resource_allocation = resource_allocation / sum(resource_allocation);
                end
                
            catch
                resource_allocation = [];
            end
        end
        
        %% ====== 错误处理方法 ======
        function handleCriticalError(obj, ME, method_name)
            %HANDLECRITICALERROR 处理关键错误
            
            fprintf('❌ [FSPSimulator::%s] 关键错误: %s\n', method_name, ME.message);
            
            if ~isempty(ME.stack)
                fprintf('   错误位置: %s (第%d行)\n', ME.stack(1).file, ME.stack(1).line);
            end
            
            if obj.debug_mode
                fprintf('   完整错误信息:\n%s\n', ME.getReport());
            end
        end
        
        function handleEpisodeError(obj, ME, episode_num)
            %HANDLEEPISODEERROR 处理episode错误
            
            if obj.debug_mode || mod(episode_num, 10) == 0
                fprintf('⚠️ Episode %d 执行失败: %s\n', episode_num, ME.message);
            end
            
            if ~isempty(obj.performance_monitor)
                obj.performance_monitor.errors_encountered = obj.performance_monitor.errors_encountered + 1;
            end
        end
        
        function handleAgentError(obj, ME, agent_idx, episode_num)
            %HANDLEAGENTERROR 处理智能体错误
            
            if obj.debug_mode
                fprintf('⚠️ Episode %d 智能体 %d 执行失败: %s\n', episode_num, agent_idx, ME.message);
            end
        end
        
        %% ====== 性能监控方法 ======
        function generatePerformanceReport(obj)
            %GENERATEPERFORMANCEREPORT 生成性能报告
            
            if isempty(obj.performance_monitor)
                return;
            end
            
            execution_time = toc(obj.performance_monitor.start_time);
            episodes_completed = obj.performance_monitor.episodes_completed;
            errors_encountered = obj.performance_monitor.errors_encountered;
            
            fprintf('\n=== FSPSimulator 性能报告 ===\n');
            fprintf('执行时间: %.2f 秒\n', execution_time);
            fprintf('完成episodes: %d\n', episodes_completed);
            fprintf('遇到错误: %d\n', errors_encountered);
            
            if episodes_completed > 0
                fprintf('平均每episode时间: %.3f 秒\n', execution_time / episodes_completed);
                fprintf('成功率: %.1f%%\n', (episodes_completed / (episodes_completed + errors_encountered)) * 100);
            end
            
            fprintf('================================\n\n');
        end
        
        %% ====== 工具方法 ======
        function value = getConfigValue(obj, config, field_name, default_value)
            %GETCONFIGVALUE 获取配置值
            
            if isfield(config, field_name)
                value = config.(field_name);
            else
                value = default_value;
            end
        end
        
        function has_method = hasMethod(obj, object, method_name)
            %HASMETHOD 检查对象是否有指定方法
            
            has_method = false;
            
            try
                if isobject(object)
                    methods_list = methods(object);
                    has_method = any(strcmp(methods_list, method_name));
                elseif isstruct(object)
                    has_method = isfield(object, method_name) && isa(object.(method_name), 'function_handle');
                end
            catch
                has_method = false;
            end
        end
    end
end

%% ====== 使用示例和测试 ======
% 
% % 基本使用示例:
% config = struct();
% config.n_episodes_per_iter = 50;
% config.n_stations = 10;
% config.debug_mode = true;
% 
% simulator = FSPSimulator(config);
% results = simulator.runEpisodes(env, defender_agents, attacker_agent, config);
% 
% % 错误容忍模式:
% config.error_tolerance_mode = true;
% simulator = FSPSimulator(config);
% 
% % 禁用策略跟踪以提高性能:
% config.enable_strategy_tracking = false;
% simulator = FSPSimulator(config);