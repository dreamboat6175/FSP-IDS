%% TCSEnvironment.m - 基于攻击者策略的防守优化环境
% =========================================================================
% 描述: 实现基于攻击者策略计算最优防守策略的TCS环境
% 防守者奖励基于资源分配偏差指数(RADI)
% 攻击者目标是最大化成功攻击的基站数量
% =========================================================================
classdef TCSEnvironment < handle
    
    properties
        % 基本参数
        n_stations              % 基站数量
        n_components            % 每个基站的组件数量
        total_components        % 总组件数
        n_resource_types        % 防守资源类型数
        n_attack_types          % 攻击类型数
        
        % 状态和动作空间
        state_dim               % 状态维度
        action_dim_defender     % 防守者动作维度
        action_dim_attacker     % 攻击者动作维度
        current_state           % 当前状态
        
        % 策略相关
        attacker_strategy       % 攻击者当前策略
        defender_strategy       % 防守者当前策略
        optimal_defender_strategy % 基于攻击策略的最优防守策略
        
        % 资源和成本
        total_resources         % 总防守资源
        defense_costs           % 防守成本矩阵
        attack_costs            % 攻击成本
        
        % 组件属性
        component_importance    % 组件重要性
        component_station_map   % 组件到基站的映射
        station_values          % 基站价值
        
        % 攻击成功概率模型
        base_attack_success_prob % 基础攻击成功概率
        defense_effectiveness    % 防守效果矩阵
        
        % RADI相关
        radi_score              % 当前RADI分数
        radi_history            % RADI历史记录
        
        % 博弈结果
        attack_success_count    % 攻击成功数量
        attack_success_rate_history % History of attack success rates
        defense_history         % 防守历史
        attack_target_history   % History of which stations were attacked
        
        % 时间步
        time_step               % 当前时间步
        
        % 优化参数
        optimization_method     % 优化方法 ('analytical' 或 'numerical')
        epsilon                 % 小常数，避免除零
        
        % ========== FIXED: Added Compatibility Properties ==========
        % Properties required by the new compatibility methods
        optimal_attacker_strategy % Optimal attacker strategy
        radi_defender             % RADI score for the defender
        radi_attacker             % RADI score for the attacker
        attacker_actions          % Attacker actions for compatibility methods
        defender_actions          % Defender actions for compatibility methods
    end
    
    methods
        function obj = TCSEnvironment(config)
            % 构造函数
            obj.n_stations = config.n_stations;
            obj.n_components = config.n_components_per_station;
            obj.total_components = sum(obj.n_components);
            
            % 设置资源和攻击类型数量（使用默认值或从配置读取）
            if isfield(config, 'n_resource_types')
                obj.n_resource_types = config.n_resource_types;
            else
                % 默认值基于resource_types数组长度
                if isfield(config, 'resource_types')
                    obj.n_resource_types = length(config.resource_types);
                else
                    obj.n_resource_types = 5;  % 默认5种资源类型
                end
            end
            
            if isfield(config, 'n_attack_types')
                obj.n_attack_types = config.n_attack_types;
            else
                % 默认值基于attack_types数组长度
                if isfield(config, 'attack_types')
                    obj.n_attack_types = length(config.attack_types);
                else
                    obj.n_attack_types = 6;  % 默认6种攻击类型
                end
            end
            
            % 设置总资源
            if isfield(config, 'total_resources')
                obj.total_resources = config.total_resources;
            else
                obj.total_resources = 100;  % 默认总资源
            end
            
            obj.epsilon = 1e-6; % FIXED: Using a smaller epsilon
            obj.optimization_method = 'analytical';
            
            % 初始化各个组件
            obj.initializeComponents();
            obj.initializeCosts();
            obj.initializeAttackModel();
            obj.calculateSpaceDimensions();
            
            % 初始化策略
            obj.attacker_strategy = ones(1, obj.n_stations) / obj.n_stations;
            obj.defender_strategy = ones(1, obj.n_stations) / obj.n_stations;
            obj.optimal_defender_strategy = obj.defender_strategy;
            
            % 初始化历史记录
            obj.radi_history = [];
            obj.attack_success_rate_history = [];
            obj.attack_target_history = [];
            obj.defense_history = [];
            obj.time_step = 0;
        end
        
        function initializeComponents(obj)
            % 初始化组件和基站价值
            obj.component_importance = zeros(1, obj.total_components);
            obj.component_station_map = zeros(1, obj.total_components);
            obj.station_values = zeros(1, obj.n_stations);
            
            idx = 1;
            for s = 1:obj.n_stations
                % 基站价值递减
                base_value = 1 - (s-1) * 0.15 / obj.n_stations;
                obj.station_values(s) = base_value;
                
                for c = 1:obj.n_components(s)
                    % 组件重要性
                    if c <= 2  % 核心组件
                        obj.component_importance(idx) = 0.8 + 0.2 * rand();
                    else  % 次要组件
                        obj.component_importance(idx) = 0.3 + 0.4 * rand();
                    end
                    obj.component_station_map(idx) = s;
                    idx = idx + 1;
                end
            end
            
            % 归一化
            obj.station_values = obj.station_values / sum(obj.station_values);
            obj.component_importance = obj.component_importance / max(obj.component_importance);
        end
        
        function initializeCosts(obj)
            % 初始化成本矩阵
            obj.defense_costs = zeros(obj.n_stations, obj.n_resource_types);
            for i = 1:obj.n_stations
                for j = 1:obj.n_resource_types
                    % 成本随防守等级增加
                    base_cost = 10 + j * 5;
                    obj.defense_costs(i, j) = base_cost * (0.8 + 0.4 * obj.station_values(i));
                end
            end
            
            % 攻击成本（简化模型）
            obj.attack_costs = 5 * ones(1, obj.n_stations);
        end
        
        function initializeAttackModel(obj)
            % 初始化攻击成功概率模型
            obj.base_attack_success_prob = 0.8;  % 无防守时的成功率
            
            % 防守效果矩阵 (防守类型 x 攻击类型)
            obj.defense_effectiveness = zeros(obj.n_resource_types, obj.n_attack_types);
            for i = 1:obj.n_resource_types
                for j = 1:obj.n_attack_types
                    % 对应防守类型效果最好
                    if i == j
                        obj.defense_effectiveness(i, j) = 0.8;
                    else
                        obj.defense_effectiveness(i, j) = 0.3 + 0.2 * rand();
                    end
                end
            end
        end
        
        function calculateSpaceDimensions(obj)
            % 计算状态和动作空间维度
            % 状态包括：各站点状态、资源分配、历史信息
            obj.state_dim = obj.n_stations * 5 + 10;
            
            % 动作空间：每个站点的策略选择
            obj.action_dim_defender = obj.n_stations;
            obj.action_dim_attacker = obj.n_stations;
        end
        
        function state = reset(obj)
            % 重置环境
            obj.time_step = 0;
            obj.attack_success_rate_history = [];
            obj.attack_target_history = [];
            obj.defense_history = [];
            obj.radi_history = [];
            obj.attack_success_count = 0;
            
            % 重置策略
            obj.attacker_strategy = ones(1, obj.n_stations) / obj.n_stations;
            obj.defender_strategy = ones(1, obj.n_stations) / obj.n_stations;
            obj.optimal_defender_strategy = obj.defender_strategy;
            obj.radi_score = 0;
            
            % 生成初始状态
            state = obj.generateState();
            obj.current_state = state;
        end
        
        function state = generateState(obj)
            % 生成状态向量
            state = zeros(1, obj.state_dim);
            
            % 基站状态信息
            idx = 1;
            for i = 1:obj.n_stations
                state(idx) = obj.station_values(i);  % 价值
                state(idx+1) = obj.defender_strategy(i);  % 防守分配
                state(idx+2) = obj.attacker_strategy(i);  % 攻击概率
                state(idx+3) = obj.getStationVulnerability(i);  % 脆弱性
                state(idx+4) = obj.getRecentAttackRate(i);  % 近期攻击率
                idx = idx + 5;
            end
            
            % 全局信息
            state(idx) = obj.radi_score;  % RADI分数
            state(idx+1) = obj.time_step / 1000;  % 归一化时间
            state(idx+2) = sum(obj.defender_strategy > 0.1) / obj.n_stations;  % 防守覆盖率
            state(idx+3) = obj.getAttackConcentration();  % 攻击集中度
            
            % 历史信息
            if ~isempty(obj.attack_success_rate_history)
                recent_success_rate = mean(obj.attack_success_rate_history(max(1,end-9):end));
                state(idx+4) = recent_success_rate;
            end
        end
        
        function [next_state, reward_def, reward_att, info] = step(obj, defender_action, attacker_action)
            % 执行一步环境交互
            
            % 解析动作
            defender_allocation = obj.parseDefenderAction(defender_action);
            attacker_targets = obj.parseAttackerAction(attacker_action);
            
            % 更新策略估计
            obj.updateStrategies(defender_allocation, attacker_targets);
            
            % 计算基于攻击策略的最优防守策略
            obj.optimal_defender_strategy = obj.computeOptimalDefenseStrategy(obj.attacker_strategy);
            
            % 计算RADI（资源分配偏差指数）
            obj.radi_score = obj.calculateRADIScore(obj.defender_strategy, obj.optimal_defender_strategy);
            
            % 执行攻防对抗
            [attack_results, defense_results] = obj.executeAttackDefense(attacker_targets, defender_allocation);
            
            % 计算奖励
            reward_def = obj.calculateDefenderReward(defense_results, obj.radi_score);
            reward_att = obj.calculateAttackerReward(attack_results);
            
            % 更新状态
            obj.updateHistory(attack_results, defense_results, attacker_targets);
            next_state = obj.generateState();
            obj.current_state = next_state;
            
            % 准备信息
            info = obj.prepareInfo(attack_results, defense_results, defender_allocation);
            
            obj.time_step = obj.time_step + 1;
        end
        
        function optimal_defense = computeOptimalDefenseStrategy(obj, attack_strategy)
            % 基于攻击策略计算最优防守策略
            % 使用解析解或数值优化
            
            if strcmp(obj.optimization_method, 'analytical')
                % 解析解：防守资源与攻击强度和基站价值成正比
                weighted_threats = attack_strategy .* obj.station_values;
                
                % 考虑防守成本效率
                avg_costs = mean(obj.defense_costs, 2)';
                defense_efficiency = weighted_threats ./ (avg_costs + obj.epsilon);
                
                % 归一化得到最优策略
                optimal_defense = defense_efficiency / sum(defense_efficiency);
                
            else
                % 数值优化方法
                optimal_defense = obj.numericalOptimization(attack_strategy);
            end
            
            % 确保是有效的概率分布
            optimal_defense = max(0, optimal_defense);
            optimal_defense = optimal_defense / sum(optimal_defense);
        end
        
        function radi = calculateRADIScore(obj, current_strategy, optimal_strategy)
            % 计算资源分配偏差指数
            % RADI = sum((current - optimal)^2) / sum(optimal^2)
            
            % 确保维度匹配
            if length(current_strategy) ~= length(optimal_strategy)
                error('策略维度不匹配');
            end
            
            % 计算偏差
            deviation = current_strategy - optimal_strategy;
            
            % 计算RADI
            numerator = sum(deviation.^2);
            denominator = sum(optimal_strategy.^2) + obj.epsilon;
            
            radi = numerator / denominator;
            
            % 记录历史
            obj.radi_history(end+1) = radi;
        end
        
        function [attack_results, defense_results] = executeAttackDefense(obj, attacker_targets, defender_allocation)
            % 执行攻防对抗
            
            attack_results = struct();
            defense_results = struct();
            
            % 初始化结果
            attack_results.success_count = 0;
            attack_results.success_stations = [];
            attack_results.total_attacks = sum(attacker_targets > 0);
            
            defense_results.defended_count = 0;
            defense_results.defended_stations = [];
            defense_results.resource_used = sum(defender_allocation);
            
            % 对每个基站进行攻防判定
            for i = 1:obj.n_stations
                if attacker_targets(i) > 0  % 该基站被攻击
                    % 计算攻击成功概率
                    success_prob = obj.calculateAttackSuccessProbability(i, ...
                        attacker_targets(i), defender_allocation(i));
                    
                    % 随机判定
                    if rand() < success_prob
                        attack_results.success_count = attack_results.success_count + 1;
                        attack_results.success_stations(end+1) = i;
                    else
                        defense_results.defended_count = defense_results.defended_count + 1;
                        defense_results.defended_stations(end+1) = i;
                    end
                end
            end
            
            % 计算成功率
            if attack_results.total_attacks > 0
                attack_results.success_rate = attack_results.success_count / attack_results.total_attacks;
            else
                attack_results.success_rate = 0;
            end
            
            obj.attack_success_count = attack_results.success_count;
        end
        
        function success_prob = calculateAttackSuccessProbability(obj, station_idx, attack_strength, defense_strength)
            % 计算单个基站的攻击成功概率
            
            % 基础成功概率
            base_prob = obj.base_attack_success_prob;
            
            % 防守效果
            if defense_strength > 0
                % 简化模型：防守强度越大，成功率越低
                defense_factor = defense_strength / (attack_strength + defense_strength + obj.epsilon);
                success_prob = base_prob * (1 - defense_factor * 0.9);  % 最多降低90%
            else
                success_prob = base_prob;
            end
            
            % 考虑基站脆弱性
            vulnerability = obj.getStationVulnerability(station_idx);
            success_prob = success_prob * (0.5 + 0.5 * vulnerability);
            
            % 确保在[0,1]范围内
            success_prob = max(0, min(1, success_prob));
        end
        
        function reward = calculateDefenderReward(obj, defense_results, radi_score)
            % 计算防守者奖励（基于RADI）
            
            % RADI惩罚（RADI越小越好）
            radi_penalty = -100 * radi_score;
            
            % 防守成功奖励
            defense_bonus = 20 * defense_results.defended_count;
            
            % 资源效率奖励
            if defense_results.resource_used > 0
                efficiency = defense_results.defended_count / defense_results.resource_used;
                efficiency_bonus = 10 * efficiency;
            else
                efficiency_bonus = 0;
            end
            
            % 组合奖励
            reward = radi_penalty + defense_bonus + efficiency_bonus;
            
            % 使用tanh归一化
            reward = tanh(reward / 50);
        end
        
        function reward = calculateAttackerReward(obj, attack_results)
            % 计算攻击者奖励（最大化攻击成功基站数）
            
            % 攻击成功奖励
            success_reward = 50 * attack_results.success_count;
            
            % 攻击成功率奖励
            if attack_results.total_attacks > 0
                rate_bonus = 20 * attack_results.success_rate;
            else
                rate_bonus = 0;
            end
            
            % 攻击高价值目标的额外奖励
            value_bonus = 0;
            for station = attack_results.success_stations
                value_bonus = value_bonus + 30 * obj.station_values(station);
            end
            
            % 组合奖励
            reward = success_reward + rate_bonus + value_bonus;
            
            % 归一化
            reward = tanh(reward / 50);
        end
        
        function updateStrategies(obj, defender_allocation, attacker_targets)
            % 更新策略估计（使用指数移动平均）
            alpha = 0.3;  % 学习率
            
            % 更新防守策略
            current_defense = defender_allocation / (sum(defender_allocation) + obj.epsilon);
            obj.defender_strategy = alpha * current_defense + (1-alpha) * obj.defender_strategy;
            obj.defender_strategy = obj.defender_strategy / sum(obj.defender_strategy);
            
            % 更新攻击策略
            current_attack = zeros(1, obj.n_stations);
            for i = 1:obj.n_stations
                if attacker_targets(i) > 0
                    current_attack(i) = 1;
                end
            end
            current_attack = current_attack / (sum(current_attack) + obj.epsilon);
            obj.attacker_strategy = alpha * current_attack + (1-alpha) * obj.attacker_strategy;
            obj.attacker_strategy = obj.attacker_strategy / sum(obj.attacker_strategy);
        end
        
        function updateHistory(obj, attack_results, defense_results, attacker_targets)
            % 更新历史记录
            obj.attack_success_rate_history(end+1) = attack_results.success_rate;
            
            % Store which stations were attacked (as a logical row vector)
            if isempty(obj.attack_target_history)
                obj.attack_target_history = attacker_targets > 0;
            else
                obj.attack_target_history(end+1, :) = attacker_targets > 0;
            end
    
            obj.defense_history(end+1) = defense_results.defended_count / (sum(obj.attacker_strategy > 0) + obj.epsilon);
        end
        
        function info = prepareInfo(obj, attack_results, defense_results, defender_allocation)
            % 准备返回信息
            info = struct();
            
            % 基本信息
            info.time_step = obj.time_step;
            info.radi_score = obj.radi_score;
            info.attack_success_count = attack_results.success_count;
            info.defense_success_count = defense_results.defended_count;
            
            % 策略信息
            info.current_strategies.attacker = obj.attacker_strategy;
            info.current_strategies.defender = obj.defender_strategy;
            info.optimal_defender_strategy = obj.optimal_defender_strategy;
            
            % 资源分配
            info.resource_allocation = defender_allocation / (sum(defender_allocation) + obj.epsilon);
            
            % 性能指标
            info.attack_success_rate = attack_results.success_rate;
            if ~isempty(obj.radi_history)
                info.avg_radi = mean(obj.radi_history(max(1,end-9):end));
            else
                info.avg_radi = 0;
            end
            
            % 详细结果
            info.attack_results = attack_results;
            info.defense_results = defense_results;
        end
        
        % 辅助方法
        function defender_allocation = parseDefenderAction(obj, action)
            % 解析防守动作为资源分配
            if isa(action, 'numeric') && length(action) == obj.n_stations
                % 直接是分配向量
                defender_allocation = max(0, action);
            else
                % 动作索引转换为分配
                defender_allocation = zeros(1, obj.n_stations);
                if action >= 1 && action <= obj.n_stations
                    defender_allocation(action) = 1;
                end
            end
            
            % 归一化到资源约束
            total = sum(defender_allocation);
            if total > 0
                defender_allocation = defender_allocation * obj.total_resources / total;
            end
        end
        
        function attacker_targets = parseAttackerAction(obj, action)
            % 解析攻击动作
            if isa(action, 'numeric') && length(action) == obj.n_stations
                % 攻击强度向量
                attacker_targets = max(0, action);
            else
                % 单个目标
                attacker_targets = zeros(1, obj.n_stations);
                if action >= 1 && action <= obj.n_stations
                    attacker_targets(action) = 1;
                end
            end
        end
        
        function vulnerability = getStationVulnerability(obj, station_idx)
            % 获取基站脆弱性（简化模型）
            % 基于历史被攻击频率
            if ~isempty(obj.attack_target_history)
                num_time_steps = size(obj.attack_target_history, 1);
                attack_frequency = sum(obj.attack_target_history(:, station_idx)) / num_time_steps;
                % Vulnerability increases with attack frequency
                vulnerability = 0.5 + 0.5 * attack_frequency;
            else
                vulnerability = 0.5; % Default vulnerability
            end
        end
        
        function rate = getRecentAttackRate(obj, station_idx)
            % 获取近期攻击率
            rate = obj.attacker_strategy(station_idx);
        end
        
        function concentration = getAttackConcentration(obj)
            % 计算攻击集中度（熵的负数）
            p = obj.attacker_strategy + obj.epsilon;
            concentration = -sum(p .* log(p));
            concentration = concentration / log(obj.n_stations);  % 归一化
        end
        
        function optimal_defense = numericalOptimization(obj, attack_strategy)
            % 数值优化方法（备用）
            % 最小化期望损失
            
            % 目标函数：期望损失
            objective = @(x) obj.expectedLoss(x, attack_strategy);
            
            % 约束条件
            A = [];
            b = [];
            Aeq = ones(1, obj.n_stations);  % sum = 1
            beq = 1;
            lb = zeros(1, obj.n_stations);
            ub = ones(1, obj.n_stations);
            
            % 初始猜测
            x0 = ones(1, obj.n_stations) / obj.n_stations;
            
            % 优化选项
            options = optimoptions('fmincon', 'Display', 'off', 'Algorithm', 'sqp'); % FIXED: specified algorithm
            
            % 求解
            [optimal_defense, ~] = fmincon(objective, x0, A, b, Aeq, beq, lb, ub, [], options);
        end
        
        function loss = expectedLoss(obj, defense_strategy, attack_strategy)
            % 计算期望损失
            loss = 0;
            for i = 1:obj.n_stations
                success_prob = obj.calculateAttackSuccessProbability(i, ...
                    attack_strategy(i), defense_strategy(i) * obj.total_resources);
                loss = loss + success_prob * obj.station_values(i) * attack_strategy(i);
            end
        end
        
        function plotStrategies(obj)
            % 可视化当前策略
            figure('Name', 'TCS环境策略分析');
            
            subplot(2,2,1);
            bar(1:obj.n_stations, [obj.attacker_strategy; obj.defender_strategy; obj.optimal_defender_strategy]');
            xlabel('基站编号');
            ylabel('策略概率');
            title('策略对比');
            legend('攻击策略', '当前防守', '最优防守');
            grid on;
            
            subplot(2,2,2);
            plot(obj.radi_history, 'LineWidth', 2);
            xlabel('时间步');
            ylabel('RADI分数');
            title('RADI演化');
            grid on;
            
            subplot(2,2,3);
            if ~isempty(obj.attack_success_rate_history)
                plot(obj.attack_success_rate_history, 'r-', 'LineWidth', 2);
                hold on;
                plot(obj.defense_history, 'b-', 'LineWidth', 2);
                xlabel('时间步');
                ylabel('成功率');
                title('攻防成功率');
                legend('攻击成功率', '防守成功率');
                grid on;
                ylim([0 1]); % FIXED: Set y-axis limit for rates
            end
            
            subplot(2,2,4);
            bar(1:obj.n_stations, obj.station_values);
            xlabel('基站编号');
            ylabel('价值');
            title('基站价值分布');
            grid on;
        end
    
    % ========== 用户新增兼容方法 ========== 
    function state = generateInitialState(obj)
        % 生成初始状态（兼容旧版本）
        state = obj.generateState();
    end
    
    function allocation = calculateResourceAllocationFromActions(obj, defender_actions)
        % 从动作计算资源分配（兼容旧版本）
        if isa(defender_actions, 'numeric') && length(defender_actions) == obj.n_stations
            allocation = abs(defender_actions);
            allocation = allocation / (sum(allocation) + obj.epsilon);
        else
            % 单个动作索引
            allocation = zeros(1, obj.n_resource_types);
            if defender_actions >= 1 && defender_actions <= obj.n_stations
                % 简化映射：将站点索引映射到资源类型
                resource_idx = mod(defender_actions - 1, obj.n_resource_types) + 1;
                allocation(resource_idx) = 1;
            end
            allocation = allocation / (sum(allocation) + obj.epsilon);
        end
        % FIXED: Ensure the output dimension matches n_resource_types instead of being hardcoded
        if length(allocation) < obj.n_resource_types
            allocation = [allocation, zeros(1, obj.n_resource_types - length(allocation))];
        elseif length(allocation) > obj.n_resource_types
            allocation = allocation(1:obj.n_resource_types);
        end
    end
    
    function updateStrategiesBasedOnRADI(obj)
        % 基于RADI更新策略（兼容方法）
        % 使用学习率逐步调整策略向最优策略靠近
        learning_rate = 0.1;
        % 更新防守策略
        obj.defender_strategy = (1 - learning_rate) * obj.defender_strategy + ...
                                learning_rate * obj.optimal_defender_strategy;
        % 归一化
        obj.defender_strategy = obj.defender_strategy / sum(obj.defender_strategy);
    end
    
    function result = executeGameAndEvaluate(obj)
        % 执行博弈并评估（兼容旧版本接口）
        % FIXED: This method uses attacker_actions and defender_actions, which are now class properties.
        % Ensure they are set before calling this method.
        [attack_results, defense_results] = obj.executeAttackDefense(...
            obj.attacker_actions, obj.defender_actions);
        % 构造兼容的结果结构
        result.attack_success = attack_results.success_count;
        result.defense_effectiveness = defense_results.defended_count / (attack_results.total_attacks + obj.epsilon);
        result.total_attacks = attack_results.total_attacks;
        result.total_defenses = sum(obj.defender_actions > 0);
        result.resource_efficiency = defense_results.resource_used / obj.total_resources;
    end
    
    function computeOptimalStrategies(obj)
        % 计算最优策略（使用新的优化方法）
        % 更新攻击者策略评估
        station_values = obj.station_values;
        % 简单的威胁评估
        threat_levels = obj.attacker_strategy * 0.5 + 0.5;
        % 攻击者最优策略
        attack_utilities = station_values .* threat_levels;
        obj.optimal_attacker_strategy = attack_utilities / sum(attack_utilities);
        % 使用新方法计算防守者最优策略
        obj.optimal_defender_strategy = obj.computeOptimalDefenseStrategy(obj.attacker_strategy);
    end
    
    % FIXED: Renamed the conflicting method from 'calculateRADI' to 'calculateAllRADI'
    function calculateAllRADI(obj)
        % 计算RADI（兼容方法）
        obj.radi_defender = obj.calculateRADIScore(obj.defender_strategy, obj.optimal_defender_strategy);
        
        % Ensure optimal_attacker_strategy is computed before use
        if isempty(obj.optimal_attacker_strategy)
            obj.computeOptimalStrategies();
        end
        obj.radi_attacker = obj.calculateRADIScore(obj.attacker_strategy, obj.optimal_attacker_strategy);
    end
    
    % FIXED: Removed the problematic get.resource_effectiveness method.
    % The property 'defense_effectiveness' is correctly initialized in the constructor.
    
    function value = get.defense_costs(obj)
        % 确保defense_costs存在
        if isempty(obj.defense_costs)
            obj.initializeCosts();
        end
        value = obj.defense_costs;
    end
    
    function threat_level = assessStationThreatLevel(obj, station)
        % 评估单个站点的威胁级别
        % 基于历史攻击频率和当前攻击策略
        historical_weight = 0.3;
        current_weight = 0.7;
        % 当前威胁
        current_threat = obj.attacker_strategy(station);
        % 历史威胁
        if ~isempty(obj.attack_target_history)
            % Calculate the historical attack frequency for this specific station
            num_time_steps = size(obj.attack_target_history, 1);
            historical_threat = sum(obj.attack_target_history(:, station)) / num_time_steps;
        else
            % No history, so threat is based on initial uniform strategy
            historical_threat = 1 / obj.n_stations;
        end
        threat_level = historical_weight * historical_threat + current_weight * current_threat;
    end
    
    function [next_state, reward_def, done, info] = stepFSPSimulatorCompat(obj, defender_action, attacker_action)
        % 确保与FSPSimulator兼容的step方法
        % 调用新的step方法
        [next_state, reward_def, reward_att, info] = obj.step(defender_action, attacker_action);
        % 添加done标志（FSP中通常不使用）
        done = false;
        % 确保info包含所需字段
        if ~isfield(info, 'resource_allocation')
            info.resource_allocation = obj.calculateResourceAllocationFromActions(defender_action);
        end
        % 添加其他可能需要的字段
        info.attack_reward = reward_att;
    end
    end
end
