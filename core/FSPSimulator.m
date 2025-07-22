classdef FSPSimulator < handle
    %% FSPSimulator - FSP仿真器（增强版）
    % ================================================================
    % 版本：v3.0 - 支持增强数据记录
    % 新增功能：
    % 1. 策略提取和记录
    % 2. 与TCSEnvironment v4.0的增强数据记录集成
    % 3. 改进的错误处理和调试信息
    % ================================================================
    
    properties (Access = private)
        config          % 仿真配置
        debug_mode      % 调试模式
        strategy_tracking_enabled % 策略跟踪开关
    end
    
    methods (Access = public)
        function obj = FSPSimulator(config)
            %FSPSIMULATOR 构造函数
            
            if nargin < 1
                config = struct();
            end
            
            obj.config = config;
            obj.debug_mode = obj.getConfigValue(config, 'debug_mode', false);
            obj.strategy_tracking_enabled = obj.getConfigValue(config, 'enable_strategy_tracking', true);
            
            if obj.strategy_tracking_enabled
                status_string = '启用';
            else
                status_string = '禁用';
            end
            fprintf('[FSPSimulator v3.0] 初始化完成 - 策略跟踪: %s\n', status_string);
        end
        
        function episode_results = runEpisodes(obj, env, defender_agents, attacker_agent, config)
            %RUNEPISODES 运行episodes（增强版）
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
                % 验证输入
                obj.validateInputs(env, defender_agents, attacker_agent, config);
                
                % 初始化结果结构
                episode_results = obj.initializeResults(defender_agents, config);
                
                % 运行episodes
                episode_results = obj.executeEpisodes(env, defender_agents, attacker_agent, config, episode_results);
                
                % 计算最终统计
                episode_results = obj.computeFinalStatistics(episode_results, config);
                
                if obj.debug_mode
                    fprintf('[FSPSimulator] Episodes运行完成\n');
                end
                
            catch ME
                fprintf('❌ FSPSimulator运行失败: %s\n', ME.message);
                if ~isempty(ME.stack)
                    fprintf('错误位置: %s (第%d行)\n', ME.stack(1).file, ME.stack(1).line);
                end
                rethrow(ME);
            end
        end
        
        function strategies = extractStrategies(obj, agents)
            %EXTRACTSTRATEGIES 提取智能体策略
            % 输入:
            %   agents - 智能体数组
            % 输出:
            %   strategies - 策略数组
            
            strategies = cell(length(agents), 1);
            
            for i = 1:length(agents)
                agent = agents{i};
                
                try
                    % 尝试多种方法获取策略
                    if obj.hasMethod(agent, 'getStrategy')
                        strategies{i} = agent.getStrategy();
                    elseif obj.hasMethod(agent, 'getPolicy')
                        strategies{i} = agent.getPolicy();
                    elseif obj.hasMethod(agent, 'getCurrentStrategy')
                        strategies{i} = agent.getCurrentStrategy();
                    elseif isprop(agent, 'strategy') || isfield(agent, 'strategy')
                        strategies{i} = agent.strategy;
                    elseif isprop(agent, 'policy') || isfield(agent, 'policy')
                        strategies{i} = agent.policy;
                    else
                        % 默认均匀策略
                        if isprop(agent, 'n_actions') || isfield(agent, 'n_actions')
                            n_actions = agent.n_actions;
                        else
                            n_actions = obj.config.n_stations;
                        end
                        strategies{i} = ones(1, n_actions) / n_actions;
                        
                        if obj.debug_mode && mod(randi(100), 50) == 0 % 偶尔提示
                            fprintf('⚠️ 智能体%d无法提取策略，使用默认均匀策略\n', i);
                        end
                    end
                    
                    % 验证和归一化策略
                    strategies{i} = obj.validateAndNormalizeStrategy(strategies{i});
                    
                catch ME
                    if obj.debug_mode
                        fprintf('⚠️ 提取智能体%d策略失败: %s，使用默认策略\n', i, ME.message);
                    end
                    strategies{i} = ones(1, obj.config.n_stations) / obj.config.n_stations;
                end
            end
        end
    end
    
    methods (Access = private)
        function validateInputs(obj, env, defender_agents, attacker_agent, config)
            %VALIDATEINPUTS 验证输入参数
            
            if isempty(env) || ~isa(env, 'TCSEnvironment')
                error('FSPSimulator:InvalidInput', '需要有效的TCSEnvironment对象');
            end
            
            if isempty(defender_agents) || ~iscell(defender_agents)
                error('FSPSimulator:InvalidInput', '需要有效的防御者智能体数组');
            end
            
            if isempty(attacker_agent)
                error('FSPSimulator:InvalidInput', '需要有效的攻击者智能体');
            end
            
            if ~isfield(config, 'n_episodes_per_iter') || config.n_episodes_per_iter <= 0
                error('FSPSimulator:InvalidInput', '需要有效的n_episodes_per_iter参数');
            end
        end
        
        function episode_results = initializeResults(obj, defender_agents, config)
            %INITIALIZERESULTS 初始化结果结构
            
            n_agents = length(defender_agents);
            n_episodes = config.n_episodes_per_iter;
            
            episode_results = struct();
            episode_results.avg_radi = zeros(1, n_agents);
            episode_results.avg_efficiency = zeros(1, n_agents);
            episode_results.avg_balance = zeros(1, n_agents);
            episode_results.avg_defender_reward = zeros(1, n_agents);
            episode_results.avg_attacker_reward = 0;
            episode_results.attack_info = cell(n_episodes, 1);
            episode_results.avg_resource_allocation = zeros(n_agents, config.n_stations);
            
            % 新增：增强指标初始化
            episode_results.avg_nash_convergence = zeros(1, n_agents);
            episode_results.avg_attack_coverage = zeros(1, n_agents);
            episode_results.avg_defense_effectiveness = zeros(1, n_agents);
            episode_results.strategy_consistency = zeros(1, n_agents);
            
            % 策略记录
            if obj.strategy_tracking_enabled
                episode_results.attacker_strategy = [];
                episode_results.defender_strategies = cell(n_agents, 1);
                episode_results.strategy_evolution = struct();
            end
        end
        
        function episode_results = executeEpisodes(obj, env, defender_agents, attacker_agent, config, episode_results)
            %EXECUTEEPISODES 执行episodes
            
            n_agents = length(defender_agents);
            n_episodes = config.n_episodes_per_iter;
            n_steps_per_episode = obj.getConfigValue(config, 'n_steps_per_episode', 50);
            
            % 累积变量
            radi_sum = zeros(1, n_agents);
            efficiency_sum = zeros(1, n_agents);
            balance_sum = zeros(1, n_agents);
            defender_reward_sum = zeros(1, n_agents);
            attacker_reward_sum = 0;
            resource_allocation_sum = zeros(n_agents, config.n_stations);
            
            % 新增：增强指标累积变量
            nash_convergence_sum = zeros(1, n_agents);
            attack_coverage_sum = zeros(1, n_agents);
            defense_effectiveness_sum = zeros(1, n_agents);
            
            % 策略一致性跟踪
            strategy_history = cell(n_agents, n_episodes);
            
            for ep = 1:n_episodes
                try
                    % 重置环境
                    current_state = env.reset();
                    
                    if obj.debug_mode && mod(ep, max(1, floor(n_episodes/5))) == 0
                        fprintf('[FSPSimulator] Episode %d/%d\n', ep, n_episodes);
                    end
                    
                    % 存储每个智能体在这个episode中的结果
                    episode_radi = zeros(1, n_agents);
                    episode_efficiency = zeros(1, n_agents);
                    episode_balance = zeros(1, n_agents);
                    episode_defender_rewards = zeros(1, n_agents);
                    episode_resource_allocation = zeros(n_agents, config.n_stations);
                    episode_nash_convergence = zeros(1, n_agents);
                    episode_attack_coverage = zeros(1, n_agents);
                    episode_defense_effectiveness = zeros(1, n_agents);
                    
                    % 每个防御者执行episodes
                    for agent_idx = 1:n_agents
                        % === 策略提取和记录 ===
                        if obj.strategy_tracking_enabled
                            % 提取当前策略
                            [attack_strategy, defense_strategy] = obj.extractCurrentStrategies(attacker_agent, defender_agents{agent_idx}, config);
                            
                            % 更新环境中的策略记录
                            env.updateStrategies(attack_strategy, defense_strategy);
                            
                            % 记录策略演化
                            strategy_history{agent_idx, ep} = defense_strategy;
                        end
                        
                        % 执行episode步骤
                        [episode_metrics, final_state] = obj.executeEpisodeSteps(env, defender_agents{agent_idx}, attacker_agent, current_state, n_steps_per_episode);
                        
                        % 提取episode结果
                        episode_radi(agent_idx) = episode_metrics.avg_radi;
                        episode_efficiency(agent_idx) = episode_metrics.avg_efficiency;
                        episode_balance(agent_idx) = episode_metrics.avg_balance;
                        episode_defender_rewards(agent_idx) = episode_metrics.total_defender_reward;
                        episode_resource_allocation(agent_idx, :) = episode_metrics.avg_resource_allocation;
                        
                        % 新增：提取增强指标
                        episode_nash_convergence(agent_idx) = episode_metrics.avg_nash_convergence;
                        episode_attack_coverage(agent_idx) = episode_metrics.avg_attack_coverage;
                        episode_defense_effectiveness(agent_idx) = episode_metrics.avg_defense_effectiveness;
                    end
                    
                    % 累积结果
                    radi_sum = radi_sum + episode_radi;
                    efficiency_sum = efficiency_sum + episode_efficiency;
                    balance_sum = balance_sum + episode_balance;
                    defender_reward_sum = defender_reward_sum + episode_defender_rewards;
                    resource_allocation_sum = resource_allocation_sum + episode_resource_allocation;
                    
                    % 新增：累积增强指标
                    nash_convergence_sum = nash_convergence_sum + episode_nash_convergence;
                    attack_coverage_sum = attack_coverage_sum + episode_attack_coverage;
                    defense_effectiveness_sum = defense_effectiveness_sum + episode_defense_effectiveness;
                    
                    % 记录episode信息
                    episode_results.attack_info{ep} = struct(...
                        'episode', ep, ...
                        'radi', episode_radi, ...
                        'nash_convergence', episode_nash_convergence, ...
                        'attack_coverage', episode_attack_coverage, ...
                        'defense_effectiveness', episode_defense_effectiveness);
                    
                catch ME
                    fprintf('⚠️ Episode %d 执行失败: %s\n', ep, ME.message);
                    if obj.debug_mode
                        fprintf('错误位置: %s (第%d行)\n', ME.stack(1).file, ME.stack(1).line);
                    end
                    % 继续执行下一个episode
                end
            end
            
            % 计算平均值
            episode_results.avg_radi = radi_sum / n_episodes;
            episode_results.avg_efficiency = efficiency_sum / n_episodes;
            episode_results.avg_balance = balance_sum / n_episodes;
            episode_results.avg_defender_reward = defender_reward_sum / n_episodes;
            episode_results.avg_attacker_reward = attacker_reward_sum / n_episodes;
            episode_results.avg_resource_allocation = resource_allocation_sum / n_episodes;
            
            % 新增：计算增强指标平均值
            episode_results.avg_nash_convergence = nash_convergence_sum / n_episodes;
            episode_results.avg_attack_coverage = attack_coverage_sum / n_episodes;
            episode_results.avg_defense_effectiveness = defense_effectiveness_sum / n_episodes;
            
            % 计算策略一致性
            if obj.strategy_tracking_enabled
                episode_results.strategy_consistency = obj.calculateStrategyConsistency(strategy_history);
                
                % 记录最终策略
                final_strategies = obj.extractStrategies([attacker_agent, defender_agents]);
                episode_results.attacker_strategy = final_strategies{1};
                episode_results.defender_strategies = final_strategies(2:end);
            end
        end
        
        function [attack_strategy, defense_strategy] = extractCurrentStrategies(obj, attacker_agent, defender_agent, config)
            %EXTRACTCURRENTSTRATEGIES 提取当前策略
            
            % 获取攻击者策略
            try
                if obj.hasMethod(attacker_agent, 'getStrategy')
                    attack_strategy = attacker_agent.getStrategy();
                elseif obj.hasMethod(attacker_agent, 'getPolicy')
                    attack_strategy = attacker_agent.getPolicy();
                elseif isprop(attacker_agent, 'strategy') || isfield(attacker_agent, 'strategy')
                    attack_strategy = attacker_agent.strategy;
                else
                    attack_strategy = ones(1, config.n_stations) / config.n_stations;
                end
                attack_strategy = obj.validateAndNormalizeStrategy(attack_strategy);
            catch
                attack_strategy = ones(1, config.n_stations) / config.n_stations;
            end
            
            % 获取防御者策略
            try
                if obj.hasMethod(defender_agent, 'getStrategy')
                    defense_strategy = defender_agent.getStrategy();
                elseif obj.hasMethod(defender_agent, 'getPolicy')
                    defense_strategy = defender_agent.getPolicy();
                elseif isprop(defender_agent, 'strategy') || isfield(defender_agent, 'strategy')
                    defense_strategy = defender_agent.strategy;
                else
                    defense_strategy = ones(1, config.n_stations) / config.n_stations;
                end
                defense_strategy = obj.validateAndNormalizeStrategy(defense_strategy);
            catch
                defense_strategy = ones(1, config.n_stations) / config.n_stations;
            end
        end
        
        function [episode_metrics, final_state] = executeEpisodeSteps(obj, env, defender_agent, attacker_agent, initial_state, n_steps)
            %EXECUTEEPISODESTEPS 执行episode步骤
            
            % 初始化指标
            total_radi = 0;
            total_efficiency = 0;
            total_balance = 0;
            total_defender_reward = 0;
            total_attacker_reward = 0;
            total_resource_allocation = zeros(1, size(initial_state, 2));
            
            % 新增：增强指标
            total_nash_convergence = 0;
            total_attack_coverage = 0;
            total_defense_effectiveness = 0;
            valid_steps = 0;
            
            current_state = initial_state;
            
            for step = 1:n_steps
                try
                    % 智能体选择动作
                    defender_action = defender_agent.selectAction(current_state);
                    attacker_action = attacker_agent.selectAction(current_state);
                    
                    % 解析动作
                    defender_deployment = env.parseDefenderAction(defender_action);
                    attacker_target = env.parseAttackerAction(attacker_action);
                    
                    % 执行环境步骤
                    [next_state, reward_def, reward_att, info] = env.step(defender_deployment, attacker_target);
                    
                    % 累积基础指标
                    total_radi = total_radi + info.radi_score;
                    total_defender_reward = total_defender_reward + reward_def;
                    total_attacker_reward = total_attacker_reward + reward_att;
                    
                    % 计算效率和平衡度
                    if sum(defender_deployment) > 0
                        resource_allocation = defender_deployment / sum(defender_deployment);
                        total_resource_allocation = total_resource_allocation + resource_allocation;
                        
                        efficiency = 1 - info.radi_score; % 简化效率计算
                        balance = 1 - std(resource_allocation); % 部署平衡度
                        
                        total_efficiency = total_efficiency + efficiency;
                        total_balance = total_balance + balance;
                    end
                    
                    % 新增：累积增强指标
                    if isfield(info, 'current_nash_convergence')
                        total_nash_convergence = total_nash_convergence + info.current_nash_convergence;
                    end
                    if isfield(info, 'current_attack_coverage')
                        total_attack_coverage = total_attack_coverage + info.current_attack_coverage;
                    end
                    if isfield(info, 'current_defense_effectiveness')
                        total_defense_effectiveness = total_defense_effectiveness + info.current_defense_effectiveness;
                    end
                    
                    valid_steps = valid_steps + 1;
                    current_state = next_state;
                    
                catch ME
                    if obj.debug_mode
                        fprintf('⚠️ 步骤%d执行失败: %s\n', step, ME.message);
                    end
                    % 继续执行下一步
                end
            end
            
            % 计算平均指标
            if valid_steps > 0
                episode_metrics = struct();
                episode_metrics.avg_radi = total_radi / valid_steps;
                episode_metrics.avg_efficiency = total_efficiency / valid_steps;
                episode_metrics.avg_balance = total_balance / valid_steps;
                episode_metrics.total_defender_reward = total_defender_reward;
                episode_metrics.total_attacker_reward = total_attacker_reward;
                episode_metrics.avg_resource_allocation = total_resource_allocation / valid_steps;
                
                % 新增：增强指标平均值
                episode_metrics.avg_nash_convergence = total_nash_convergence / valid_steps;
                episode_metrics.avg_attack_coverage = total_attack_coverage / valid_steps;
                episode_metrics.avg_defense_effectiveness = total_defense_effectiveness / valid_steps;
                episode_metrics.valid_steps = valid_steps;
            else
                % 如果没有有效步骤，返回默认值
                episode_metrics = obj.getDefaultEpisodeMetrics();
            end
            
            final_state = current_state;
        end
        
        function consistency = calculateStrategyConsistency(obj, strategy_history)
            %CALCULATESTRATEGYCONSISTENCY 计算策略一致性
            
            [n_agents, n_episodes] = size(strategy_history);
            consistency = zeros(1, n_agents);
            
            for agent_idx = 1:n_agents
                agent_strategies = strategy_history(agent_idx, :);
                
                % 过滤掉空策略
                valid_strategies = {};
                for ep = 1:n_episodes
                    if ~isempty(agent_strategies{ep})
                        valid_strategies{end+1} = agent_strategies{ep};
                    end
                end
                
                if length(valid_strategies) > 1
                    % 计算策略之间的相似度
                    similarities = [];
                    for i = 1:length(valid_strategies)-1
                        for j = i+1:length(valid_strategies)
                            sim = 1 - norm(valid_strategies{i} - valid_strategies{j}, 2);
                            similarities(end+1) = max(0, sim);
                        end
                    end
                    consistency(agent_idx) = mean(similarities);
                else
                    consistency(agent_idx) = 1.0; % 单一策略视为完全一致
                end
            end
        end
        
        function episode_results = computeFinalStatistics(obj, episode_results, config)
            %COMPUTEFINALSTATISTICS 计算最终统计
            
            % 添加汇总统计
            episode_results.overall_performance = struct();
            episode_results.overall_performance.best_radi_agent = find(episode_results.avg_radi == min(episode_results.avg_radi), 1);
            episode_results.overall_performance.best_coverage_agent = find(episode_results.avg_attack_coverage == max(episode_results.avg_attack_coverage), 1);
            episode_results.overall_performance.most_consistent_agent = find(episode_results.strategy_consistency == max(episode_results.strategy_consistency), 1);
            
            % 计算综合评分
            % RADI越小越好，覆盖率越大越好，一致性越高越好
            normalized_radi = 1 - (episode_results.avg_radi / max(episode_results.avg_radi));
            normalized_coverage = episode_results.avg_attack_coverage;
            normalized_consistency = episode_results.strategy_consistency;
            
            episode_results.overall_performance.composite_scores = ...
                0.4 * normalized_radi + 0.3 * normalized_coverage + 0.3 * normalized_consistency;
            
            [~, best_overall_idx] = max(episode_results.overall_performance.composite_scores);
            episode_results.overall_performance.best_overall_agent = best_overall_idx;
            
            if obj.debug_mode
                fprintf('[FSPSimulator] 最佳综合性能智能体: %d (评分: %.3f)\n', ...
                        best_overall_idx, episode_results.overall_performance.composite_scores(best_overall_idx));
            end
        end
        
        function default_metrics = getDefaultEpisodeMetrics(obj)
            %GETDEFAULTEPISODEMETRICS 获取默认episode指标
            
            default_metrics = struct();
            default_metrics.avg_radi = 1.0;
            default_metrics.avg_efficiency = 0.0;
            default_metrics.avg_balance = 0.0;
            default_metrics.total_defender_reward = 0.0;
            default_metrics.total_attacker_reward = 0.0;
            default_metrics.avg_resource_allocation = [];
            default_metrics.avg_nash_convergence = 1.0;
            default_metrics.avg_attack_coverage = 0.5;
            default_metrics.avg_defense_effectiveness = 0.5;
            default_metrics.valid_steps = 0;
        end
        
        function strategy = validateAndNormalizeStrategy(obj, strategy)
            %VALIDATEANDNORMALIZESTRATEGY 验证和归一化策略
            
            if isempty(strategy)
                strategy = ones(1, obj.config.n_stations) / obj.config.n_stations;
                return;
            end
            
            % 确保为行向量
            strategy = strategy(:)';
            
            % 检查NaN和Inf
            if any(isnan(strategy)) || any(isinf(strategy))
                strategy = ones(1, length(strategy)) / length(strategy);
                return;
            end
            
            % 检查负值
            if any(strategy < 0)
                strategy = abs(strategy);
            end
            
            % 归一化
            if sum(strategy) > 0
                strategy = strategy / sum(strategy);
            else
                strategy = ones(1, length(strategy)) / length(strategy);
            end
        end
        
        function result = hasMethod(obj, agent, method_name)
            %HASMETHOD 检查智能体是否有指定方法
            
            try
                if isobject(agent)
                    method_list = methods(agent);
                    result = any(strcmp(method_list, method_name));
                else
                    result = false;
                end
            catch
                result = false;
            end
        end
        
        function value = getConfigValue(obj, config, field, default_value)
            %GETCONFIGVALUE 安全获取配置值
            
            if isfield(config, field)
                value = config.(field);
            else
                value = default_value;
            end
        end
    end
end