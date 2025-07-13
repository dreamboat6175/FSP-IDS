% %% CyberBattleTCSEnvironment_GameTheory.m - 基于RADI的博弈论TCS环境
% =========================================================================
% 描述: 基于资源分配偏差指数(RADI)的列控系统博弈环境
% =========================================================================

classdef CyberBattleTCSEnvironment < handle
    
    properties
        % 系统架构
        n_stations
        n_components
        total_components
        component_importance
        component_station_map
        
        % 网络拓扑
        network_topology
        node_vulnerabilities
        node_credentials
        
        % 攻击参数
        attack_types
        attack_severity
        attack_detection_difficulty
        n_attack_types
        attack_kill_chain
        
        % 资源参数
        resource_types
        resource_effectiveness
        n_resource_types
        total_resources
        defense_costs  % 每种防御措施的成本 C(dj)
        
        % 博弈论相关参数
        attacker_strategy    % P_attacker = (pa1, pa2, ..., pan)
        defender_strategy    % P_defender = (pd1, pd2, ..., pdn)
        optimal_attacker_strategy
        optimal_defender_strategy
        radi_defender        % 资源分配偏差指数
        radi_attacker
        
        % 站点级决策向量
        attacker_actions     % A_attacker = (a1, a2, ..., an)
        defender_actions     % A_defender = (d1, d2, ..., dn)
        
        % 状态和动作空间
        state_dim
        action_dim_defender
        action_dim_attacker
        
        % 环境状态
        current_state
        attack_history
        defense_history
        time_step
        
        % 奖励函数权重
        reward_weights
        
        % 策略历史和学习参数
        strategy_history
        learning_rate
        strategy_momentum
        radi_history
        
        % 博弈结果
        game_outcomes
        resource_utilization
    end
    
    methods
        function obj = CyberBattleTCSEnvironment(config)
            % 构造函数
            obj.n_stations = config.n_stations;
            obj.n_components = config.n_components_per_station;
            obj.total_components = sum(obj.n_components);
            
            obj.initializeComponents();
            obj.initializeNetworkTopology();
            obj.initializeGameTheoryParameters(config);
            
            % 初始化攻击类型
            if iscell(config.attack_types)
                obj.attack_types = [{'no_attack'}, config.attack_types(:)'];
            elseif ischar(config.attack_types)
                obj.attack_types = {'no_attack', config.attack_types};
            elseif isstring(config.attack_types)
                obj.attack_types = [{'no_attack'}, cellstr(config.attack_types)];
            else
                obj.attack_types = [{'no_attack'}, cellstr(config.attack_types)];
            end
            obj.attack_severity = [0, config.attack_severity(:)'];
            obj.attack_detection_difficulty = [0, config.attack_detection_difficulty(:)'];
            obj.n_attack_types = length(obj.attack_types);
            
            obj.initializeKillChain();
            
            obj.resource_types = config.resource_types;
            obj.resource_effectiveness = config.resource_effectiveness;
            obj.n_resource_types = length(obj.resource_types);
            obj.total_resources = config.total_resources;
            
            obj.initializeRewardWeights(config);
            obj.calculateSpaceDimensions();
            
            % 初始化学习参数
            obj.learning_rate = 0.1;
            obj.strategy_momentum = 0.9;
            
            obj.reset();
        end
        
        function initializeGameTheoryParameters(obj, config)
            % 初始化博弈论相关参数
            
            % 初始化防御成本矩阵 - 每个站点的每种防御措施的成本
            obj.defense_costs = zeros(obj.n_stations, obj.n_resource_types);
            for i = 1:obj.n_stations
                for j = 1:obj.n_resource_types
                    % 基础成本 + 站点重要性调整
                    base_cost = 10 + j * 5;  % 不同防御措施有不同基础成本
                    importance_factor = 0.5 + 0.5 * rand(); % 将在initializeComponents后更新
                    obj.defense_costs(i, j) = base_cost * importance_factor;
                end
            end
            
            % 初始化策略向量 - 攻击者只攻击一个主站
            obj.attacker_strategy = zeros(1, obj.n_stations);
            initial_focus_station = randi(obj.n_stations);
            obj.attacker_strategy(initial_focus_station) = 1.0;
            obj.attacker_strategy = obj.attacker_strategy / sum(obj.attacker_strategy);
            obj.defender_strategy = ones(1, obj.n_stations) / obj.n_stations;
            
            % 初始化最优策略
            obj.optimal_attacker_strategy = obj.attacker_strategy;
            obj.optimal_defender_strategy = obj.defender_strategy;
            
            % 初始化决策向量
            obj.attacker_actions = zeros(1, obj.n_stations);
            obj.defender_actions = zeros(1, obj.n_stations);
            
            % 初始化RADI指数
            obj.radi_defender = 0;
            obj.radi_attacker = 0;
            
            % 初始化历史记录
            obj.strategy_history = struct();
            obj.strategy_history.attacker = [];
            obj.strategy_history.defender = [];
            obj.radi_history = struct();
            obj.radi_history.defender = [];
            obj.radi_history.attacker = [];
        end
        
        function initializeComponents(obj)
            obj.component_importance = zeros(1, obj.total_components);
            obj.component_station_map = zeros(1, obj.total_components);
            idx = 1;
            for s = 1:obj.n_stations
                for c = 1:obj.n_components(s)
                    if c <= 2
                        obj.component_importance(idx) = 0.8 + 0.2 * rand();
                    elseif c <= 4
                        obj.component_importance(idx) = 0.5 + 0.3 * rand();
                    else
                        obj.component_importance(idx) = 0.2 + 0.3 * rand();
                    end
                    obj.component_station_map(idx) = s;
                    idx = idx + 1;
                end
            end
            obj.component_importance = obj.component_importance / max(obj.component_importance);
            
            % 更新防御成本基于实际组件重要性
            for i = 1:obj.n_stations
                station_components = find(obj.component_station_map == i);
                importance_factor = mean(obj.component_importance(station_components));
                for j = 1:obj.n_resource_types
                    base_cost = 10 + j * 5;
                    obj.defense_costs(i, j) = base_cost * (0.5 + 0.5 * importance_factor);
                end
            end
        end
        
        function initializeNetworkTopology(obj)
            obj.network_topology = zeros(obj.total_components);
            idx = 1;
            for s = 1:obj.n_stations
                station_components = idx:(idx + obj.n_components(s) - 1);
                for i = station_components
                    for j = station_components
                        if i ~= j, obj.network_topology(i, j) = 1; end
                    end
                end
                idx = idx + obj.n_components(s);
            end
            
            for s1 = 1:obj.n_stations
                for s2 = (s1+1):obj.n_stations
                    comp1 = sum(obj.n_components(1:s1-1)) + 1;
                    comp2 = sum(obj.n_components(1:s2-1)) + 1;
                    obj.network_topology(comp1, comp2) = 1;
                    obj.network_topology(comp2, comp1) = 1;
                end
            end
            obj.node_vulnerabilities = rand(obj.total_components, 3) * 0.5;
            obj.node_credentials = randi([0, 1], obj.total_components, 2);
        end
        
        function initializeKillChain(obj)
            obj.attack_kill_chain = {'reconnaissance', 'weaponization', 'delivery', 'exploitation', 'installation', 'command_control', 'actions'};
        end
        
        function initializeRewardWeights(obj, config)
            if isfield(config, 'reward_weights')
                obj.reward_weights = config.reward_weights;
            else
                obj.reward_weights.w_radi = 0.6;        % RADI权重
                obj.reward_weights.w_cost = 0.2;        % 成本权重
                obj.reward_weights.w_effectiveness = 0.2; % 效果权重
                obj.reward_weights.radi_penalty = 100;   % RADI惩罚系数
                obj.reward_weights.strategy_bonus = 50;  % 策略改进奖励
            end
        end
        
        function calculateSpaceDimensions(obj)
            % 基于站点的动作空间
            obj.state_dim = obj.n_stations * 8 + obj.total_components * 2; % 增加状态维度
            % 防御者动作空间：每个站点的防御策略选择
            obj.action_dim_defender = obj.n_stations * obj.n_resource_types;
            % 攻击者动作空间：每个站点的攻击策略选择
            obj.action_dim_attacker = obj.n_stations * obj.n_attack_types;
        end
        
        function state = reset(obj)
            obj.time_step = 0;
            obj.attack_history = [];
            obj.defense_history = [];
            
            % 重置策略 - 攻击者只攻击一个主站
            obj.attacker_strategy = zeros(1, obj.n_stations);
            initial_focus_station = randi(obj.n_stations);
            obj.attacker_strategy(initial_focus_station) = 1.0;
            obj.attacker_strategy = obj.attacker_strategy / sum(obj.attacker_strategy);
            obj.defender_strategy = ones(1, obj.n_stations) / obj.n_stations;
            
            % 重置RADI
            obj.radi_defender = 0;
            obj.radi_attacker = 0;
            
            % 重置博弈结果
            obj.game_outcomes = [];
            obj.resource_utilization = [];
            
            obj.current_state = obj.generateInitialState();
            state = obj.current_state;
        end
        
        function state = generateInitialState(obj)
            state = zeros(1, obj.state_dim);
            
            % 站点状态特征 (前 n_stations * 8 维)
            station_features = zeros(1, obj.n_stations * 8);
            for i = 1:obj.n_stations
                base_idx = (i-1) * 8;
                % 站点重要性
                station_components = find(obj.component_station_map == i);
                station_features(base_idx + 1) = mean(obj.component_importance(station_components));
                % 当前策略概率
                station_features(base_idx + 2) = obj.attacker_strategy(i);
                station_features(base_idx + 3) = obj.defender_strategy(i);
                % 威胁级别
                station_features(base_idx + 4) = obj.assessStationThreatLevel(i);
                % 防御成本
                station_features(base_idx + 5) = mean(obj.defense_costs(i, :));
                % 网络连接性
                station_features(base_idx + 6) = sum(sum(obj.network_topology(station_components, :))) / length(station_components);
                % 随机噪声
                station_features(base_idx + 7:base_idx + 8) = rand(1, 2) * 0.1;
            end
            
            % 组件状态特征
            component_features = rand(1, obj.total_components * 2) * 0.5;
            
            state = [station_features, component_features];
            state = state(1:obj.state_dim); % 确保维度正确
        end
        
        function threat_level = assessStationThreatLevel(obj, station)
            % 评估单个站点的威胁级别
            station_components = find(obj.component_station_map == station);
            
            if isempty(station_components)
                threat_level = 0;
                return;
            end
            
            % 基于漏洞评估威胁
            vulnerability_score = mean(mean(obj.node_vulnerabilities(station_components, :)));
            
            % 基于网络连接性评估威胁
            connectivity_score = sum(sum(obj.network_topology(station_components, :))) / length(station_components) / obj.total_components;
            
            % 基于重要性评估威胁
            importance_score = mean(obj.component_importance(station_components));
            
            threat_level = (vulnerability_score + connectivity_score + importance_score) / 3;
        end
        
        function [next_state, reward_def, reward_att, info] = step(obj, defender_action, attacker_action)
            % 解析动作为站点级决策向量
            obj.defender_actions = obj.parseDefenderAction(defender_action);
            obj.attacker_actions = obj.parseAttackerAction(attacker_action);
            
            % 检查并执行资源约束
            [obj.defender_actions, resource_violation] = obj.enforceResourceConstraints(obj.defender_actions);
            
            % 计算当前最优策略
            obj.computeOptimalStrategies();
            
            % 计算RADI指数
            obj.calculateRADI();
            
            % 基于RADI更新策略
            obj.updateStrategiesBasedOnRADI();
            
            % 执行博弈并评估效果
            game_result = obj.executeGameAndEvaluate();
            
            % 计算基于RADI的奖励
            reward_def = obj.calculateRADIBasedReward('defender', game_result, resource_violation);
            reward_att = obj.calculateRADIBasedReward('attacker', game_result, resource_violation);
            
            % 更新状态
            obj.updateState(game_result);
            next_state = obj.current_state;
            
            % 记录历史
            obj.recordHistory();
            
            % 准备信息
            info.defender_actions = obj.defender_actions;
            info.attacker_actions = obj.attacker_actions;
            info.radi_defender = obj.radi_defender;
            info.radi_attacker = obj.radi_attacker;
            info.optimal_strategies.attacker = obj.optimal_attacker_strategy;
            info.optimal_strategies.defender = obj.optimal_defender_strategy;
            info.current_strategies.attacker = obj.attacker_strategy;
            info.current_strategies.defender = obj.defender_strategy;
            info.resource_violation = resource_violation;
            info.game_result = game_result;
            
            obj.time_step = obj.time_step + 1;
        end
        
        function defender_actions = parseDefenderAction(obj, action)
            % 将动作索引转换为站点级防御决策向量
            defender_actions = zeros(1, obj.n_stations);
            
            % 确保动作在有效范围内
            action = max(1, min(action, obj.action_dim_defender));
            
            % 计算选择的站点和防御类型
            station = ceil(action / obj.n_resource_types);
            defense_type = mod(action - 1, obj.n_resource_types) + 1;
            
            % 确保索引有效
            station = max(1, min(station, obj.n_stations));
            defense_type = max(1, min(defense_type, obj.n_resource_types));
            
            defender_actions(station) = defense_type;
        end
        
        function attacker_actions = parseAttackerAction(obj, action)
            % 将动作索引转换为站点级攻击决策向量
            attacker_actions = zeros(1, obj.n_stations);
            
            % 确保动作在有效范围内
            action = max(1, min(action, obj.action_dim_attacker));
            
            % 计算选择的站点和攻击类型
            station = ceil(action / obj.n_attack_types);
            attack_type = mod(action - 1, obj.n_attack_types) + 1;
            
            % 确保索引有效
            station = max(1, min(station, obj.n_stations));
            attack_type = max(1, min(attack_type, obj.n_attack_types));
            
            attacker_actions(station) = attack_type;
        end
        
        function [constrained_actions, violation] = enforceResourceConstraints(obj, defender_actions)
            % 实施资源约束：∑C(dj) ≤ S_total
            constrained_actions = defender_actions;
            violation = false;
            
            % 计算总成本
            total_cost = 0;
            for station = 1:obj.n_stations
                if defender_actions(station) > 0
                    total_cost = total_cost + obj.defense_costs(station, defender_actions(station));
                end
            end
            
            % 检查是否违反资源约束
            if total_cost > obj.total_resources
                violation = true;
                % 基于最优策略调整资源分配
                station_priorities = obj.optimal_defender_strategy;
                [~, priority_order] = sort(station_priorities, 'descend');
                
                current_cost = 0;
                constrained_actions = zeros(1, obj.n_stations);
                
                % 按优先级分配资源
                for i = 1:length(priority_order)
                    station = priority_order(i);
                    if defender_actions(station) > 0
                        new_cost = current_cost + obj.defense_costs(station, defender_actions(station));
                        if new_cost <= obj.total_resources
                            constrained_actions(station) = defender_actions(station);
                            current_cost = new_cost;
                        end
                    end
                end
            end
        end
        
        function computeOptimalStrategies(obj)
            % 计算纳什均衡策略
            
            % 计算站点价值和威胁评估
            station_values = obj.getStationValues();
            threat_levels = obj.assessThreatLevels();
            
            % 攻击者的最优策略：基于价值和脆弱性
            attack_utilities = station_values .* threat_levels;
            obj.optimal_attacker_strategy = attack_utilities / sum(attack_utilities);
            
            % 防御者的最优策略：基于攻击威胁和资源效率
            defense_utilities = obj.calculateDefenseUtilities(station_values, threat_levels);
            obj.optimal_defender_strategy = defense_utilities / sum(defense_utilities);
        end
        
        function station_values = getStationValues(obj)
            % 计算每个站点的价值
            station_values = zeros(1, obj.n_stations);
            for station = 1:obj.n_stations
                station_components = find(obj.component_station_map == station);
                if ~isempty(station_components)
                    station_values(station) = mean(obj.component_importance(station_components));
                end
            end
        end
        
        function threat_levels = assessThreatLevels(obj)
            % 评估每个站点的威胁级别
            threat_levels = zeros(1, obj.n_stations);
            for station = 1:obj.n_stations
                threat_levels(station) = obj.assessStationThreatLevel(station);
            end
        end
        
        function defense_utilities = calculateDefenseUtilities(obj, station_values, threat_levels)
            % 计算防御效用
            defense_utilities = zeros(1, obj.n_stations);
            
            for station = 1:obj.n_stations
                % 基础效用：价值 × 威胁
                base_utility = station_values(station) * threat_levels(station);
                
                % 成本效率调整
                min_cost = min(obj.defense_costs(station, :));
                cost_efficiency = obj.total_resources / (min_cost * obj.n_stations);
                
                defense_utilities(station) = base_utility * min(1, cost_efficiency);
            end
        end
        
        function calculateRADI(obj)
            % 计算资源分配偏差指数
            % RADI = ∑||p_j - d_j|| / ||p_j||
            
            % 防御者RADI
            obj.radi_defender = 0;
            valid_stations = 0;
            for station = 1:obj.n_stations
                if obj.optimal_defender_strategy(station) > 1e-6  % 避免除零
                    actual_allocation = obj.defender_strategy(station);
                    optimal_allocation = obj.optimal_defender_strategy(station);
                    deviation = abs(actual_allocation - optimal_allocation);
                    obj.radi_defender = obj.radi_defender + deviation / optimal_allocation;
                    valid_stations = valid_stations + 1;
                end
            end
            if valid_stations > 0
                obj.radi_defender = obj.radi_defender / valid_stations;
            end
            
            % 攻击者RADI
            obj.radi_attacker = 0;
            valid_stations = 0;
            for station = 1:obj.n_stations
                if obj.optimal_attacker_strategy(station) > 1e-6  % 避免除零
                    actual_allocation = obj.attacker_strategy(station);
                    optimal_allocation = obj.optimal_attacker_strategy(station);
                    deviation = abs(actual_allocation - optimal_allocation);
                    obj.radi_attacker = obj.radi_attacker + deviation / optimal_allocation;
                    valid_stations = valid_stations + 1;
                end
            end
            if valid_stations > 0
                obj.radi_attacker = obj.radi_attacker / valid_stations;
            end
        end
        
        function updateStrategiesBasedOnRADI(obj)
            % 基于RADI更新策略
            
            % 防御者策略更新：向最优策略移动
            strategy_diff_defender = obj.optimal_defender_strategy - obj.defender_strategy;
            adaptation_rate_defender = obj.learning_rate * (1 + obj.radi_defender); % RADI越大，调整越快
            
            obj.defender_strategy = obj.defender_strategy + ...
                adaptation_rate_defender * strategy_diff_defender;
            
            % 添加动量项
            if size(obj.strategy_history.defender, 1) > 1
                momentum_defender = obj.strategy_history.defender(end, :) - ...
                                  obj.strategy_history.defender(end-1, :);
                obj.defender_strategy = obj.defender_strategy + ...
                    obj.strategy_momentum * momentum_defender;
            end
            
            % 攻击者策略更新：向最优策略移动
            strategy_diff_attacker = obj.optimal_attacker_strategy - obj.attacker_strategy;
            adaptation_rate_attacker = obj.learning_rate * (1 + obj.radi_attacker);
            
            obj.attacker_strategy = obj.attacker_strategy + ...
                adaptation_rate_attacker * strategy_diff_attacker;
            
            % 添加动量项
            if size(obj.strategy_history.attacker, 1) > 1
                momentum_attacker = obj.strategy_history.attacker(end, :) - ...
                                   obj.strategy_history.attacker(end-1, :);
                obj.attacker_strategy = obj.attacker_strategy + ...
                    obj.strategy_momentum * momentum_attacker;
            end
            
            % 归一化策略概率
            obj.defender_strategy = max(0, obj.defender_strategy);
            obj.defender_strategy = obj.defender_strategy / sum(obj.defender_strategy);
            
            obj.attacker_strategy = max(0, obj.attacker_strategy);
            obj.attacker_strategy = obj.attacker_strategy / sum(obj.attacker_strategy);
            
            % 自适应学习率
            obj.adaptLearningRate();
        end
        
        function adaptLearningRate(obj)
            % 自适应调整学习率
            if length(obj.radi_history.defender) > 5
                recent_radi = obj.radi_history.defender(end-4:end);
                if std(recent_radi) < 0.01  % RADI趋于稳定
                    obj.learning_rate = max(0.01, obj.learning_rate * 0.95); % 降低学习率
                elseif mean(recent_radi) > 0.5  % RADI过高
                    obj.learning_rate = min(0.3, obj.learning_rate * 1.05); % 提高学习率
                end
            end
        end
        
        function game_result = executeGameAndEvaluate(obj)
            % 执行博弈并评估效果
            game_result = struct();
            
            % 计算攻击成功率和防御效果
            attack_success = 0;
            defense_effectiveness = 0;
            resource_efficiency = 0;
            
            for station = 1:obj.n_stations
                attack_type = obj.attacker_actions(station);
                defense_type = obj.defender_actions(station);
                
                if attack_type > 1  % 有攻击
                    % 计算攻击成功概率
                    attack_strength = obj.attack_severity(attack_type);
                    
                    if defense_type > 0  % 有防御
                        defense_strength = obj.resource_effectiveness(defense_type);
                        success_prob = attack_strength * (1 - defense_strength);
                    else
                        success_prob = attack_strength;
                    end
                    
                    if rand() < success_prob
                        attack_success = attack_success + 1;
                    else
                        defense_effectiveness = defense_effectiveness + 1;
                    end
                end
            end
            
            % 计算资源利用效率
            total_cost = obj.calculateTotalDefenseCost();
            resource_efficiency = 1 - (total_cost / obj.total_resources);
            
            game_result.attack_success = attack_success;
            game_result.defense_effectiveness = defense_effectiveness;
            game_result.resource_efficiency = resource_efficiency;
            game_result.total_attacks = sum(obj.attacker_actions > 1);
            game_result.total_defenses = sum(obj.defender_actions > 0);
        end
        
        function reward = calculateRADIBasedReward(obj, player, game_result, resource_violation)
            % 基于RADI的奖励函数
            
            if strcmp(player, 'defender')
                % 防御者奖励
                
                % 1. RADI奖励 - RADI越小奖励越高
                r_radi = -obj.radi_defender * obj.reward_weights.radi_penalty;
                
                % 2. 防御效果奖励
                r_effectiveness = game_result.defense_effectiveness * obj.reward_weights.strategy_bonus;
                
                % 3. 资源效率奖励
                r_cost = game_result.resource_efficiency * 20;
                if resource_violation
                    r_cost = r_cost - 50;  % 违反约束惩罚
                end
                
                % 4. 策略改进奖励
                if length(obj.radi_history.defender) > 1
                    radi_improvement = obj.radi_history.defender(end-1) - obj.radi_defender;
                    if radi_improvement > 0
                        r_radi = r_radi + radi_improvement * 30; % 奖励RADI改进
                    end
                end
                
                reward = obj.reward_weights.w_radi * r_radi + ...
                        obj.reward_weights.w_effectiveness * r_effectiveness + ...
                        obj.reward_weights.w_cost * r_cost;
                
            else
                % 攻击者奖励
                
                % 1. RADI奖励 - RADI越小奖励越高
                r_radi = -obj.radi_attacker * obj.reward_weights.radi_penalty;
                
                % 2. 攻击成功奖励
                r_attack = game_result.attack_success * obj.reward_weights.strategy_bonus;
                
                % 3. 策略改进奖励
                if length(obj.radi_history.attacker) > 1
                    radi_improvement = obj.radi_history.attacker(end-1) - obj.radi_attacker;
                    if radi_improvement > 0
                        r_radi = r_radi + radi_improvement * 30;
                    end
                end
                
                reward = obj.reward_weights.w_radi * r_radi + ...
                        obj.reward_weights.w_effectiveness * r_attack;
            end
            
            % 归一化奖励
            reward = tanh(reward / 100); % 使用tanh归一化到[-1,1]
        end
        
        function total_cost = calculateTotalDefenseCost(obj)
            % 计算总防御成本
            total_cost = 0;
            for station = 1:obj.n_stations
                if obj.defender_actions(station) > 0
                    total_cost = total_cost + obj.defense_costs(station, obj.defender_actions(station));
                end
            end
        end
        
        function updateState(obj, game_result)
            % 更新环境状态
            obj.attack_history(end+1, :) = [obj.time_step, sum(obj.attacker_actions > 1), obj.radi_attacker];
            obj.defense_history(end+1, :) = [obj.time_step, sum(obj.defender_actions > 0), obj.radi_defender];
            
            % 记录博弈结果
            obj.game_outcomes(end+1, :) = [game_result.attack_success, game_result.defense_effectiveness, game_result.total_attacks, game_result.total_defenses];
            
            % 更新状态向量
            new_state = obj.generateInitialState();
            
            % 添加历史信息影响
            if obj.time_step > 0
                % 基于RADI历史调整状态
                radi_factor = 1 - min(0.5, (obj.radi_defender + obj.radi_attacker) / 2);
                new_state = new_state * radi_factor + obj.current_state * (1 - radi_factor) * 0.1;
            end
            
            obj.current_state = new_state;
        end
        
        function recordHistory(obj)
            % 记录策略和RADI历史
            obj.strategy_history.attacker(end+1, :) = obj.attacker_strategy;
            obj.strategy_history.defender(end+1, :) = obj.defender_strategy;
            obj.radi_history.defender(end+1) = obj.radi_defender;
            obj.radi_history.attacker(end+1) = obj.radi_attacker;
            
            % 限制历史长度
            max_history = 1000;
            if size(obj.strategy_history.attacker, 1) > max_history
                obj.strategy_history.attacker(1, :) = [];
                obj.strategy_history.defender(1, :) = [];
                obj.radi_history.attacker(1) = [];
                obj.radi_history.defender(1) = [];
            end
        end
        
        function metrics = getRADIMetrics(obj)
            % 返回RADI相关指标
            metrics = struct();
            
            % 当前RADI值
            metrics.current_radi_defender = obj.radi_defender;
            metrics.current_radi_attacker = obj.radi_attacker;
            
            % RADI历史统计
            if length(obj.radi_history.defender) > 1
                metrics.avg_radi_defender = mean(obj.radi_history.defender);
                metrics.std_radi_defender = std(obj.radi_history.defender);
                metrics.min_radi_defender = min(obj.radi_history.defender);
                metrics.max_radi_defender = max(obj.radi_history.defender);
                
                metrics.avg_radi_attacker = mean(obj.radi_history.attacker);
                metrics.std_radi_attacker = std(obj.radi_history.attacker);
                metrics.min_radi_attacker = min(obj.radi_history.attacker);
                metrics.max_radi_attacker = max(obj.radi_history.attacker);
                
                % RADI趋势（最近10步的变化）
                recent_steps = min(10, length(obj.radi_history.defender));
                if recent_steps > 1
                    recent_defender = obj.radi_history.defender(end-recent_steps+1:end);
                    recent_attacker = obj.radi_history.attacker(end-recent_steps+1:end);
                    
                    metrics.radi_trend_defender = (recent_defender(end) - recent_defender(1)) / recent_steps;
                    metrics.radi_trend_attacker = (recent_attacker(end) - recent_attacker(1)) / recent_steps;
                else
                    metrics.radi_trend_defender = 0;
                    metrics.radi_trend_attacker = 0;
                end
            else
                metrics.avg_radi_defender = obj.radi_defender;
                metrics.std_radi_defender = 0;
                metrics.min_radi_defender = obj.radi_defender;
                metrics.max_radi_defender = obj.radi_defender;
                
                metrics.avg_radi_attacker = obj.radi_attacker;
                metrics.std_radi_attacker = 0;
                metrics.min_radi_attacker = obj.radi_attacker;
                metrics.max_radi_attacker = obj.radi_attacker;
                
                metrics.radi_trend_defender = 0;
                metrics.radi_trend_attacker = 0;
            end
            
            % 策略收敛性指标
            metrics.strategy_convergence_defender = obj.calculateStrategyConvergence('defender');
            metrics.strategy_convergence_attacker = obj.calculateStrategyConvergence('attacker');
            
            % 博弈效率指标
            if ~isempty(obj.game_outcomes)
                metrics.avg_attack_success_rate = mean(obj.game_outcomes(:, 1) ./ max(1, obj.game_outcomes(:, 3)));
                metrics.avg_defense_success_rate = mean(obj.game_outcomes(:, 2) ./ max(1, obj.game_outcomes(:, 4)));
                metrics.resource_utilization = mean(obj.resource_utilization);
            else
                metrics.avg_attack_success_rate = 0;
                metrics.avg_defense_success_rate = 0;
                metrics.resource_utilization = 0;
            end
        end
        
        function convergence = calculateStrategyConvergence(obj, player)
            % 计算策略收敛性
            if strcmp(player, 'defender')
                strategy_history = obj.strategy_history.defender;
                optimal_strategy = obj.optimal_defender_strategy;
            else
                strategy_history = obj.strategy_history.attacker;
                optimal_strategy = obj.optimal_attacker_strategy;
            end
            
            if size(strategy_history, 1) < 2
                convergence = 0;
                return;
            end
            
            % 计算最近几步策略与最优策略的平均距离
            recent_steps = min(10, size(strategy_history, 1));
            recent_strategies = strategy_history(end-recent_steps+1:end, :);
            
            distances = zeros(recent_steps, 1);
            for i = 1:recent_steps
                distances(i) = norm(recent_strategies(i, :) - optimal_strategy);
            end
            
            % 收敛性 = 1 - 平均距离（归一化）
            convergence = 1 - mean(distances) / sqrt(obj.n_stations);
            convergence = max(0, min(1, convergence));
        end
        
        function analysis = analyzeGameDynamics(obj)
            % 分析博弈动态
            analysis = struct();
            
            if length(obj.radi_history.defender) < 10
                analysis.message = 'Insufficient data for analysis';
                return;
            end
            
            % RADI收敛分析
            recent_radi_def = obj.radi_history.defender(end-9:end);
            recent_radi_att = obj.radi_history.attacker(end-9:end);
            
            analysis.radi_stability_defender = 1 / (1 + std(recent_radi_def));
            analysis.radi_stability_attacker = 1 / (1 + std(recent_radi_att));
            
            % 策略均衡分析
            analysis.nash_distance_defender = norm(obj.defender_strategy - obj.optimal_defender_strategy);
            analysis.nash_distance_attacker = norm(obj.attacker_strategy - obj.optimal_attacker_strategy);
            
            % 学习效率分析
            if length(obj.radi_history.defender) > 20
                early_radi_def = mean(obj.radi_history.defender(1:10));
                late_radi_def = mean(obj.radi_history.defender(end-9:end));
                analysis.learning_efficiency_defender = max(0, (early_radi_def - late_radi_def) / early_radi_def);
                
                early_radi_att = mean(obj.radi_history.attacker(1:10));
                late_radi_att = mean(obj.radi_history.attacker(end-9:end));
                analysis.learning_efficiency_attacker = max(0, (early_radi_att - late_radi_att) / early_radi_att);
            else
                analysis.learning_efficiency_defender = 0;
                analysis.learning_efficiency_attacker = 0;
            end
            
            % 博弈平衡评估
            analysis.game_balance = 1 - abs(obj.radi_defender - obj.radi_attacker) / (obj.radi_defender + obj.radi_attacker + 1e-6);
            
            % 系统稳定性评估
            if ~isempty(obj.game_outcomes)
                outcome_variance = var(obj.game_outcomes(:, 1) - obj.game_outcomes(:, 2));
                analysis.system_stability = 1 / (1 + outcome_variance);
            else
                analysis.system_stability = 0;
            end
        end
        
        function visualizeRADITrends(obj)
            % 可视化RADI趋势（如果在MATLAB环境中运行）
            if length(obj.radi_history.defender) < 2
                fprintf('Insufficient data for visualization\n');
                return;
            end
            
            try
                figure('Name', 'RADI Trends Analysis');
                
                % 子图1：RADI历史趋势
                subplot(2, 2, 1);
                plot(1:length(obj.radi_history.defender), obj.radi_history.defender, 'b-', 'LineWidth', 2);
                hold on;
                plot(1:length(obj.radi_history.attacker), obj.radi_history.attacker, 'r-', 'LineWidth', 2);
                xlabel('Time Step');
                ylabel('RADI Value');
                title('RADI Evolution');
                legend('Defender', 'Attacker');
                grid on;
                
                % 子图2：策略收敛
                subplot(2, 2, 2);
                if size(obj.strategy_history.defender, 1) > 1
                    plot(obj.strategy_history.defender);
                    title('Defender Strategy Evolution');
                    xlabel('Time Step');
                    ylabel('Strategy Probability');
                    legend(arrayfun(@(x) sprintf('Station %d', x), 1:obj.n_stations, 'UniformOutput', false));
                end
                
                % 子图3：策略vs最优策略比较
                subplot(2, 2, 3);
                bar([obj.defender_strategy; obj.optimal_defender_strategy]');
                title('Current vs Optimal Defender Strategy');
                xlabel('Station');
                ylabel('Probability');
                legend('Current', 'Optimal');
                
                % 子图4：博弈结果统计
                subplot(2, 2, 4);
                if ~isempty(obj.game_outcomes)
                    histogram(obj.game_outcomes(:, 1) - obj.game_outcomes(:, 2), 'BinWidth', 1);
                    title('Game Outcome Distribution');
                    xlabel('Attack Success - Defense Success');
                    ylabel('Frequency');
                end
                
            catch ME
                fprintf('Visualization error: %s\n', ME.message);
            end
        end
        
        function saveGameData(obj, filename)
            % 保存博弈数据
            game_data = struct();
            game_data.strategy_history = obj.strategy_history;
            game_data.radi_history = obj.radi_history;
            game_data.game_outcomes = obj.game_outcomes;
            game_data.optimal_strategies.defender = obj.optimal_defender_strategy;
            game_data.optimal_strategies.attacker = obj.optimal_attacker_strategy;
            game_data.defense_costs = obj.defense_costs;
            game_data.station_importance = obj.getStationValues();
            game_data.threat_levels = obj.assessThreatLevels();
            
            if nargin < 2
                filename = sprintf('cyberbattle_game_data_%s.mat', datestr(now, 'yyyymmdd_HHMMSS'));
            end
            
            try
                save(filename, 'game_data');
                fprintf('Game data saved to: %s\n', filename);
            catch ME
                fprintf('Error saving data: %s\n', ME.message);
            end
        end
        
        function loadGameData(obj, filename)
            % 加载博弈数据
            try
                loaded_data = load(filename);
                game_data = loaded_data.game_data;
                
                obj.strategy_history = game_data.strategy_history;
                obj.radi_history = game_data.radi_history;
                obj.game_outcomes = game_data.game_outcomes;
                obj.optimal_defender_strategy = game_data.optimal_strategies.defender;
                obj.optimal_attacker_strategy = game_data.optimal_strategies.attacker;
                
                if isfield(game_data, 'defense_costs')
                    obj.defense_costs = game_data.defense_costs;
                end
                
                fprintf('Game data loaded from: %s\n', filename);
            catch ME
                fprintf('Error loading data: %s\n', ME.message);
            end
        end
        
        function summary = getGameSummary(obj)
            % 获取博弈总结
            summary = struct();
            
            % 基本信息
            summary.total_time_steps = obj.time_step;
            summary.n_stations = obj.n_stations;
            summary.total_resources = obj.total_resources;
            
            % RADI指标
            radi_metrics = obj.getRADIMetrics();
            summary.final_radi_defender = radi_metrics.current_radi_defender;
            summary.final_radi_attacker = radi_metrics.current_radi_attacker;
            summary.avg_radi_defender = radi_metrics.avg_radi_defender;
            summary.avg_radi_attacker = radi_metrics.avg_radi_attacker;
            
            % 收敛性
            summary.strategy_convergence_defender = radi_metrics.strategy_convergence_defender;
            summary.strategy_convergence_attacker = radi_metrics.strategy_convergence_attacker;
            
            % 博弈平衡
            game_analysis = obj.analyzeGameDynamics();
            summary.game_balance = game_analysis.game_balance;
            summary.system_stability = game_analysis.system_stability;
            summary.learning_efficiency_defender = game_analysis.learning_efficiency_defender;
            summary.learning_efficiency_attacker = game_analysis.learning_efficiency_attacker;
            
            % 策略信息
            summary.final_defender_strategy = obj.defender_strategy;
            summary.final_attacker_strategy = obj.attacker_strategy;
            summary.optimal_defender_strategy = obj.optimal_defender_strategy;
            summary.optimal_attacker_strategy = obj.optimal_attacker_strategy;
            
            % 性能评估
            if obj.time_step > 0
                summary.performance_score = (summary.strategy_convergence_defender + ...
                                           summary.strategy_convergence_attacker + ...
                                           summary.game_balance + ...
                                           summary.system_stability) / 4;
            else
                summary.performance_score = 0;
            end
        end
    end
end
