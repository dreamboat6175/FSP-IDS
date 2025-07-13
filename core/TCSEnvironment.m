%% TCSEnvironment.m - 基于RADI的博弈论TCS环境（优化版）
% =========================================================================
% 描述: 基于资源分配偏差指数(RADI)的列控系统博弈环境
% 修改内容：
% 1. 攻击者策略初始化为非均匀分布
% 2. 攻击者奖励函数基于攻击成功次数
% 3. 添加历史分析功能自适应调整策略
% =========================================================================

classdef TCSEnvironment < handle
    
    properties
        % 系统架构
        n_stations
        n_components
        total_components
        component_importance
        component_station_map
        component_status  % 新增：组件状态跟踪
        
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
        
        % 新增属性
        attack_success_history   % 攻击成功历史记录
        initial_focus_station    % 初始重点攻击站点
        strategy_change_count    % 策略改变次数
        last_strategy_change     % 上次策略改变时间
    end
    
    methods
        function obj = TCSEnvironment(config)
            % 构造函数 - 按正确顺序初始化
            
            % 第1步：设置基础参数
            obj.n_stations = config.n_stations;
            obj.n_components = config.n_components_per_station;
            obj.total_components = sum(obj.n_components);
            
            % 第2步：设置攻击类型相关参数
            if iscell(config.attack_types)
                obj.attack_types = [{'no_attack'}, config.attack_types(:)'];
            elseif ischar(config.attack_types) || isstring(config.attack_types)
                obj.attack_types = [{'no_attack'}, cellstr(config.attack_types)];
            else
                obj.attack_types = [{'no_attack'}, cellstr(config.attack_types)];
            end
            obj.attack_severity = [0, config.attack_severity(:)'];
            obj.attack_detection_difficulty = [0, config.attack_detection_difficulty(:)'];
            obj.n_attack_types = length(obj.attack_types);
            
            % 第3步：设置资源类型相关参数
            obj.resource_types = config.resource_types;
            obj.resource_effectiveness = config.resource_effectiveness;
            obj.n_resource_types = length(obj.resource_types);
            obj.total_resources = config.total_resources;
            
            % 第4步：验证关键参数
            if obj.n_stations <= 0 || obj.n_resource_types <= 0
                error('TCSEnvironment: Invalid parameters');
            end
            
            % 第5步：初始化其他组件
            obj.initializeRewardWeights(config);
            obj.calculateSpaceDimensions();
            obj.learning_rate = 0.1;
            obj.strategy_momentum = 0.9;
            obj.time_step = 0;
            
            % 第6步：初始化系统组件
            obj.initializeComponents();
            obj.initializeNetworkTopology();
            obj.initializeKillChain();
            
            % 第7步：初始化博弈论参数
            obj.initializeGameTheoryParameters(config);
            
            % 第8步：初始化新增属性
            obj.attack_success_history = [];
            obj.strategy_change_count = 0;
            obj.last_strategy_change = 0;
            
            % 第9步：重置环境
            obj.reset();
        end
        
        function initializeGameTheoryParameters(obj, config)
            % 初始化博弈论相关参数
            
            % 初始化防御成本矩阵
            obj.defense_costs = zeros(obj.n_stations, obj.n_resource_types);
            for i = 1:obj.n_stations
                importance_factor = 0.5 + 0.5 * rand();
                for j = 1:obj.n_resource_types
                    base_cost = 10 + (j-1) * 5;
                    obj.defense_costs(i, j) = base_cost * importance_factor;
                end
            end
            
            % 初始化防御者策略 - 均匀分布
            obj.defender_strategy = ones(1, obj.n_stations) / obj.n_stations;
            
            % 【修改】初始化攻击者策略 - 非均匀分布
            obj.attacker_strategy = 0.05 * ones(1, obj.n_stations);
            obj.initial_focus_station = randi(obj.n_stations);
            obj.attacker_strategy(obj.initial_focus_station) = 0.5;
            obj.attacker_strategy = obj.attacker_strategy / sum(obj.attacker_strategy);
            
            fprintf('攻击者初始策略：重点攻击站点 %d，分配权重 %.2f\n', ...
                obj.initial_focus_station, obj.attacker_strategy(obj.initial_focus_station));
            
            % 初始化最优策略
            obj.optimal_attacker_strategy = obj.attacker_strategy;
            obj.optimal_defender_strategy = obj.defender_strategy;
            
            % 初始化决策向量和RADI
            obj.attacker_actions = zeros(1, obj.n_stations);
            obj.defender_actions = zeros(1, obj.n_stations);
            obj.radi_defender = 0;
            obj.radi_attacker = 0;
            
            % 初始化历史记录
            obj.strategy_history = struct('attacker', [], 'defender', []);
            obj.radi_history = struct('defender', [], 'attacker', []);
            obj.game_outcomes = [];
            obj.resource_utilization = [];
        end
        
        function initializeComponents(obj)
            % 初始化系统组件
            obj.component_importance = rand(1, obj.total_components);
            obj.component_station_map = [];
            obj.component_status = ones(1, obj.total_components);  % 初始状态都是正常
            
            comp_idx = 1;
            for i = 1:obj.n_stations
                for j = 1:obj.n_components(i)
                    obj.component_station_map(comp_idx) = i;
                    comp_idx = comp_idx + 1;
                end
            end
        end
        
        function initializeNetworkTopology(obj)
            % 初始化网络拓扑
            obj.network_topology = rand(obj.total_components, obj.total_components) > 0.8;
            obj.network_topology = obj.network_topology | obj.network_topology';
            obj.network_topology(logical(eye(size(obj.network_topology)))) = 0;
            
            obj.node_vulnerabilities = rand(obj.total_components, 3) * 0.5;
            obj.node_credentials = randi([0, 1], obj.total_components, 2);
        end
        
        function initializeKillChain(obj)
            % 初始化攻击链
            obj.attack_kill_chain = {'reconnaissance', 'weaponization', 'delivery', ...
                                   'exploitation', 'installation', 'command_control', 'actions'};
        end
        
        function initializeRewardWeights(obj, config)
            % 初始化奖励权重
            if isfield(config, 'reward_weights')
                obj.reward_weights = config.reward_weights;
            else
                obj.reward_weights.w_radi = 0.6;
                obj.reward_weights.w_cost = 0.2;
                obj.reward_weights.w_effectiveness = 0.2;
                obj.reward_weights.radi_penalty = 100;
                obj.reward_weights.strategy_bonus = 50;
            end
        end
        
        function calculateSpaceDimensions(obj)
            % 计算状态和动作空间维度
            obj.state_dim = obj.n_stations * 8 + obj.total_components * 2;
            obj.action_dim_defender = obj.n_stations * obj.n_resource_types;
            obj.action_dim_attacker = obj.n_stations * obj.n_attack_types;
        end
        
        function state = reset(obj)
            % 重置环境
            obj.time_step = 0;
            obj.attack_history = [];
            obj.defense_history = [];
            
            % 重置攻击成功历史
            obj.attack_success_history = [];
            obj.strategy_change_count = 0;
            obj.last_strategy_change = 0;
            
            % 重新初始化策略
            obj.defender_strategy = ones(1, obj.n_stations) / obj.n_stations;
            
            % 攻击者策略保持非均匀初始化
            obj.attacker_strategy = 0.05 * ones(1, obj.n_stations);
            obj.initial_focus_station = randi(obj.n_stations);
            obj.attacker_strategy(obj.initial_focus_station) = 0.5;
            obj.attacker_strategy = obj.attacker_strategy / sum(obj.attacker_strategy);
            
            % 重置其他参数
            obj.radi_defender = 0;
            obj.radi_attacker = 0;
            obj.game_outcomes = [];
            obj.resource_utilization = [];
            obj.component_status = ones(1, obj.total_components);
            
            % 清空历史记录
            obj.strategy_history = struct('attacker', [], 'defender', []);
            obj.radi_history = struct('defender', [], 'attacker', []);
            
            % 生成初始状态
            obj.current_state = obj.generateInitialState();
            state = obj.current_state;
        end
        
        function state = generateInitialState(obj)
            % 生成初始状态向量
            state = zeros(1, obj.state_dim);
            
            % 站点状态特征
            station_features = zeros(1, obj.n_stations * 8);
            for i = 1:obj.n_stations
                base_idx = (i-1) * 8;
                station_components = find(obj.component_station_map == i);
                
                % 站点重要性
                station_features(base_idx + 1) = mean(obj.component_importance(station_components));
                % 当前策略概率
                station_features(base_idx + 2) = obj.attacker_strategy(i);
                station_features(base_idx + 3) = obj.defender_strategy(i);
                % 威胁级别
                station_features(base_idx + 4) = obj.assessStationThreatLevel(i);
                % 防御成本
                station_features(base_idx + 5) = mean(obj.defense_costs(i, :));
                % 网络连接性
                connections = sum(sum(obj.network_topology(station_components, :))) / length(station_components);
                station_features(base_idx + 6) = connections;
                % 组件状态
                station_features(base_idx + 7) = mean(obj.component_status(station_components));
                % 历史攻击成功率（如果有历史）
                if ~isempty(obj.attack_success_history)
                    recent_success = mean(obj.attack_success_history(max(1,end-9):end));
                    station_features(base_idx + 8) = recent_success / obj.n_stations;
                else
                    station_features(base_idx + 8) = 0;
                end
            end
            
            % 组件状态特征
            component_features = [obj.component_status, rand(1, obj.total_components) * 0.1];
            
            % 组合特征
            state = [station_features, component_features];
            state = state(1:obj.state_dim);  % 确保维度正确
        end
        
        function threat_level = assessStationThreatLevel(obj, station)
            % 评估站点威胁级别
            station_components = find(obj.component_station_map == station);
            
            if isempty(station_components)
                threat_level = 0;
                return;
            end
            
            % 基于多个因素评估威胁
            vulnerability_score = mean(mean(obj.node_vulnerabilities(station_components, :)));
            connectivity_score = sum(sum(obj.network_topology(station_components, :))) / ...
                               (length(station_components) * obj.total_components);
            importance_score = mean(obj.component_importance(station_components));
            
            % 考虑当前攻击策略
            attack_focus = obj.attacker_strategy(station);
            
            threat_level = 0.3 * vulnerability_score + ...
                          0.2 * connectivity_score + ...
                          0.2 * importance_score + ...
                          0.3 * attack_focus;
        end
        
        function [next_state, reward_def, reward_att, done, info] = step(obj, defender_action_vec, attacker_action_vec)
            % 执行一步环境交互
            
            % 解析动作
            obj.defender_actions = obj.parseDefenderAction(defender_action_vec);
            obj.attacker_actions = obj.parseAttackerAction(attacker_action_vec);
            
            % 执行资源约束
            [obj.defender_actions, resource_violation] = obj.enforceResourceConstraints(obj.defender_actions);
            
            % 计算最优策略
            obj.computeOptimalStrategies();
            
            % 计算RADI指数
            obj.calculateRADI();
            
            % 基于RADI更新策略
            obj.updateStrategiesBasedOnRADI();
            
            % 执行博弈并评估
            game_result = obj.executeGameAndEvaluate();
            
            % 计算奖励
            reward_def = obj.calculateRADIBasedReward('defender', game_result, resource_violation);
            reward_att = obj.calculateRADIBasedReward('attacker', game_result, resource_violation);
            
            % 更新状态
            obj.updateState(game_result);
            next_state = obj.current_state;
            
            % 记录历史
            obj.recordHistory();
            
            % 检查是否结束
            done = obj.time_step >= 1000;
            
            % 准备信息（确保资源分配正确）
            info = obj.prepareInfoDict(game_result, resource_violation, obj.defender_actions);
            
            obj.time_step = obj.time_step + 1;
        end
        
        function defender_actions = parseDefenderAction(obj, action_vec)
            % 解析防御者动作
            defender_actions = zeros(1, obj.n_stations);
            for j = 1:obj.n_stations
                if j <= length(action_vec)
                    a = action_vec(j);
                    if ~isempty(a) && ~isnan(a) && ~isinf(a) && a >= 1
                        defender_actions(j) = min(max(1, round(a)), obj.n_resource_types);
                    else
                        defender_actions(j) = 1;
                    end
                else
                    defender_actions(j) = 1;
                end
            end
        end
        
        function attacker_actions = parseAttackerAction(obj, action_vec)
            % 解析攻击者动作
            attacker_actions = zeros(1, obj.n_stations);
            for j = 1:obj.n_stations
                if j <= length(action_vec)
                    a = action_vec(j);
                    if ~isempty(a) && ~isnan(a) && ~isinf(a) && a >= 1
                        attacker_actions(j) = min(max(1, round(a)), obj.n_attack_types);
                    else
                        attacker_actions(j) = 1;
                    end
                else
                    attacker_actions(j) = 1;
                end
            end
        end
        
        function [constrained_actions, violation] = enforceResourceConstraints(obj, defender_actions)
            % 强制执行资源约束
            constrained_actions = defender_actions;
            violation = false;
            
            % 计算总成本
            total_cost = 0;
            for j = 1:obj.n_stations
                if defender_actions(j) > 0
                    total_cost = total_cost + obj.defense_costs(j, defender_actions(j));
                end
            end
            
            % 检查约束
            if total_cost > obj.total_resources
                violation = true;
                
                % 基于优先级调整
                [~, priority_order] = sort(obj.optimal_defender_strategy, 'descend');
                
                current_cost = 0;
                constrained_actions = zeros(1, obj.n_stations);
                
                for k = 1:length(priority_order)
                    j = priority_order(k);
                    action_idx = defender_actions(j);
                    new_cost = current_cost + obj.defense_costs(j, action_idx);
                    
                    if new_cost <= obj.total_resources
                        constrained_actions(j) = action_idx;
                        current_cost = new_cost;
                    end
                end
            end
        end
        
        function computeOptimalStrategies(obj)
            % 计算纳什均衡策略
            
            % 计算站点价值
            station_values = zeros(1, obj.n_stations);
            for i = 1:obj.n_stations
                station_components = find(obj.component_station_map == i);
                station_values(i) = mean(obj.component_importance(station_components));
            end
            
            % 计算威胁评估
            threat_levels = zeros(1, obj.n_stations);
            for i = 1:obj.n_stations
                threat_levels(i) = obj.assessStationThreatLevel(i);
            end
            
            % 攻击者最优策略：基于价值和脆弱性
            attack_utilities = station_values .* threat_levels;
            obj.optimal_attacker_strategy = attack_utilities / sum(attack_utilities);
            
            % 防御者最优策略：基于威胁和成本效益
            defense_priorities = threat_levels .* station_values;
            avg_costs = mean(obj.defense_costs, 2)';
            defense_utilities = defense_priorities ./ (avg_costs + 1);
            obj.optimal_defender_strategy = defense_utilities / sum(defense_utilities);
        end
        
        function calculateRADI(obj)
            % 计算RADI指数
            obj.radi_defender = obj.calculateRADIScore(obj.optimal_defender_strategy, obj.defender_strategy);
            obj.radi_attacker = obj.calculateRADIScore(obj.optimal_attacker_strategy, obj.attacker_strategy);
        end
        
        function radi = calculateRADIScore(obj, optimal_strategy, current_strategy)
            % 计算RADI分数
            if length(optimal_strategy) ~= length(current_strategy)
                error('Strategy dimensions mismatch');
            end
            
            % 计算策略偏差
            deviation = abs(optimal_strategy - current_strategy);
            radi = sum(deviation .* optimal_strategy) / sum(optimal_strategy);
            
            % 归一化到[0,1]
            radi = min(1, max(0, radi));
        end
        
        function updateStrategiesBasedOnRADI(obj)
            % 基于RADI更新策略
            
            % 防御者策略更新
            strategy_diff_defender = obj.optimal_defender_strategy - obj.defender_strategy;
            adaptation_rate_defender = obj.learning_rate * (1 + obj.radi_defender);
            
            obj.defender_strategy = obj.defender_strategy + ...
                adaptation_rate_defender * strategy_diff_defender;
            
            % 添加动量项
            if size(obj.strategy_history.defender, 1) > 1
                momentum_defender = obj.strategy_history.defender(end, :) - ...
                                  obj.strategy_history.defender(end-1, :);
                obj.defender_strategy = obj.defender_strategy + ...
                    obj.strategy_momentum * momentum_defender;
            end
            
            % 【修改】攻击者策略更新：基于攻击成功进行调整
            if ~isempty(obj.game_outcomes) && size(obj.game_outcomes, 1) > 0
                recent_outcome = obj.game_outcomes(end, :);
                attack_success = recent_outcome(1);
                total_attacks = recent_outcome(3);
                
                if attack_success > 0
                    % 强化成功的攻击站点
                    for station = 1:obj.n_stations
                        if obj.attacker_actions(station) > 1
                            success_boost = 0.1 * (attack_success / max(1, total_attacks));
                            obj.attacker_strategy(station) = obj.attacker_strategy(station) * (1 + success_boost);
                        end
                    end
                else
                    % 没有成功时向最优策略小幅移动
                    strategy_diff_attacker = obj.optimal_attacker_strategy - obj.attacker_strategy;
                    adaptation_rate_attacker = obj.learning_rate * 0.5;
                    obj.attacker_strategy = obj.attacker_strategy + ...
                        adaptation_rate_attacker * strategy_diff_attacker;
                end
            else
                % 初始阶段：向最优策略移动
                strategy_diff_attacker = obj.optimal_attacker_strategy - obj.attacker_strategy;
                adaptation_rate_attacker = obj.learning_rate * (1 + obj.radi_attacker);
                obj.attacker_strategy = obj.attacker_strategy + ...
                    adaptation_rate_attacker * strategy_diff_attacker;
            end
            
            % 添加探索性扰动
            exploration_noise = 0.01 * randn(1, obj.n_stations);
            obj.attacker_strategy = obj.attacker_strategy + exploration_noise;
            
            % 归一化策略
            obj.defender_strategy = max(0, obj.defender_strategy);
            obj.defender_strategy = obj.defender_strategy / sum(obj.defender_strategy);
            
            obj.attacker_strategy = max(0, obj.attacker_strategy);
            obj.attacker_strategy = obj.attacker_strategy / sum(obj.attacker_strategy);
            
            % 自适应学习率
            obj.adaptLearningRate();
            
            % 【新增】调用历史分析函数
            obj.adjustAttackerStrategyBasedOnHistory();
        end
        
        function adaptLearningRate(obj)
            % 自适应调整学习率
            if length(obj.radi_history.defender) > 5
                recent_radi = obj.radi_history.defender(end-4:end);
                if std(recent_radi) < 0.01
                    obj.learning_rate = max(0.01, obj.learning_rate * 0.95);
                elseif mean(recent_radi) > 0.5
                    obj.learning_rate = min(0.3, obj.learning_rate * 1.05);
                end
            end
        end
        
        function adjustAttackerStrategyBasedOnHistory(obj)
            % 基于历史攻击成功记录调整策略
            
            if length(obj.attack_success_history) < 10
                return;  % 历史数据不足
            end
            
            % 分析最近的成功趋势
            recent_window = 10;
            recent_success = obj.attack_success_history(end-recent_window+1:end);
            avg_success = mean(recent_success);
            
            % 如果平均成功率下降，考虑改变策略
            if avg_success < 0.3 && length(obj.attack_success_history) > 20
                earlier_success = obj.attack_success_history(end-2*recent_window+1:end-recent_window);
                earlier_avg = mean(earlier_success);
                
                if avg_success < earlier_avg * 0.7  % 成功率显著下降
                    % 记录策略改变
                    obj.strategy_change_count = obj.strategy_change_count + 1;
                    obj.last_strategy_change = obj.time_step;
                    
                    % 选择新的重点站点
                    [~, current_focus] = max(obj.attacker_strategy);
                    new_focus_station = randi(obj.n_stations);
                    while new_focus_station == current_focus && obj.n_stations > 1
                        new_focus_station = randi(obj.n_stations);
                    end
                    
                    % 渐进式转移重点
                    transfer_rate = 0.2;
                    obj.attacker_strategy = obj.attacker_strategy * (1 - transfer_rate);
                    obj.attacker_strategy(new_focus_station) = ...
                        obj.attacker_strategy(new_focus_station) + transfer_rate;
                    
                    % 归一化
                    obj.attacker_strategy = obj.attacker_strategy / sum(obj.attacker_strategy);
                    
                    fprintf('时间步 %d: 攻击策略调整，转移重点到站点 %d (第%d次调整)\n', ...
                        obj.time_step, new_focus_station, obj.strategy_change_count);
                end
            end
        end
        
        function game_result = executeGameAndEvaluate(obj)
            % 执行博弈并评估效果
            game_result = struct();
            
            % 初始化计数器
            attack_success = 0;
            defense_effectiveness = 0;
            station_attack_results = zeros(1, obj.n_stations);  % 记录每个站点的攻击结果
            
            for station = 1:obj.n_stations
                attack_type = obj.attacker_actions(station);
                defense_type = obj.defender_actions(station);
                
                if attack_type > 1  % 有攻击
                    % 计算攻击成功概率
                    attack_strength = obj.attack_severity(attack_type);
                    
                    if defense_type > 0  % 有防御
                        defense_strength = obj.resource_effectiveness(defense_type);
                        % 考虑组件状态的影响
                        station_components = find(obj.component_station_map == station);
                        component_health = mean(obj.component_status(station_components));
                        success_prob = attack_strength * (1 - defense_strength * component_health);
                    else
                        success_prob = attack_strength;
                    end
                    
                    % 执行攻击
                    if rand() < success_prob
                        attack_success = attack_success + 1;
                        station_attack_results(station) = 1;
                        
                        % 更新组件状态（攻击成功造成损害）
                        station_components = find(obj.component_station_map == station);
                        damage = 0.1 + 0.1 * attack_strength;
                        obj.component_status(station_components) = ...
                            max(0, obj.component_status(station_components) - damage);
                    else
                        defense_effectiveness = defense_effectiveness + 1;
                    end
                end
            end
            
            % 计算资源利用效率
            total_cost = obj.calculateTotalDefenseCost();
            resource_efficiency = 1 - (total_cost / obj.total_resources);
            
            % 记录结果
            game_result.attack_success = attack_success;
            game_result.defense_effectiveness = defense_effectiveness;
            game_result.resource_efficiency = resource_efficiency;
            game_result.total_attacks = sum(obj.attacker_actions > 1);
            game_result.total_defenses = sum(obj.defender_actions > 0);
            game_result.station_attack_results = station_attack_results;
        end
        
        function reward = calculateRADIBasedReward(obj, player, game_result, resource_violation)
            % 基于RADI的奖励函数
            
            if strcmp(player, 'defender')
                % 防御者奖励
                r_radi = -obj.radi_defender * obj.reward_weights.radi_penalty;
                r_effectiveness = game_result.defense_effectiveness * obj.reward_weights.strategy_bonus;
                r_cost = game_result.resource_efficiency * 20;
                
                if resource_violation
                    r_cost = r_cost - 50;
                end
                
                if length(obj.radi_history.defender) > 1
                    radi_improvement = obj.radi_history.defender(end-1) - obj.radi_defender;
                    if radi_improvement > 0
                        r_radi = r_radi + radi_improvement * 30;
                    end
                end
                
                reward = obj.reward_weights.w_radi * r_radi + ...
                        obj.reward_weights.w_effectiveness * r_effectiveness + ...
                        obj.reward_weights.w_cost * r_cost;
                
            else
                % 【修改】攻击者奖励 - 主要基于攻击成功次数
                
                % 1. 攻击成功奖励（权重大幅提高）
                r_attack_success = game_result.attack_success * 100;
                
                % 2. 攻击效率奖励
                if game_result.total_attacks > 0
                    attack_efficiency = game_result.attack_success / game_result.total_attacks;
                    r_efficiency = attack_efficiency * 50;
                else
                    r_efficiency = 0;
                end
                
                % 3. 目标集中度奖励
                strategy_entropy = -sum(obj.attacker_strategy .* log(obj.attacker_strategy + 1e-10));
                max_entropy = log(obj.n_stations);
                r_focus = (1 - strategy_entropy/max_entropy) * 20;
                
                % 4. 连续成功奖励
                if length(obj.attack_success_history) >= 2
                    recent_success = obj.attack_success_history(end-1:end);
                    if all(recent_success > 0)
                        r_consecutive = 30;
                    else
                        r_consecutive = 0;
                    end
                else
                    r_consecutive = 0;
                end
                
                % 5. RADI相关奖励（权重降低）
                r_radi = -obj.radi_attacker * obj.reward_weights.radi_penalty * 0.2;
                
                % 组合奖励
                reward = 0.6 * r_attack_success + ...
                        0.2 * r_efficiency + ...
                        0.1 * r_focus + ...
                        0.05 * r_consecutive + ...
                        0.05 * r_radi;
                
                % 记录攻击成功历史
                obj.attack_success_history(end+1) = game_result.attack_success;
            end
            
            % 归一化奖励
            reward = tanh(reward / 100);
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
            obj.game_outcomes(end+1, :) = [game_result.attack_success, ...
                                          game_result.defense_effectiveness, ...
                                          game_result.total_attacks, ...
                                          game_result.total_defenses];
            
            % 记录资源利用率
            obj.resource_utilization(end+1) = game_result.resource_efficiency;
            
            % 更新状态向量
            new_state = obj.generateInitialState();
            
            % 添加历史信息影响
            if obj.time_step > 0
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
        end
        
        function info = prepareInfoDict(obj, game_result, resource_violation, defender_actions)
            % 准备返回信息字典
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
            info.time_step = obj.time_step;
            info.attack_success_rate = 0;
            
            if ~isempty(obj.attack_success_history)
                recent_window = min(10, length(obj.attack_success_history));
                info.attack_success_rate = mean(obj.attack_success_history(end-recent_window+1:end));
            end
            
            % 计算资源分配（使用defender_actions参数）
            if nargin >= 4 && ~isempty(defender_actions)
                info.resource_allocation = obj.calculateResourceAllocationFromActions(defender_actions);
            else
                info.resource_allocation = obj.calculateResourceAllocationFromActions(obj.defender_actions);
            end
            
            % 验证资源分配
            if abs(sum(info.resource_allocation) - 1) > 1e-6
                warning('资源分配总和不为1: %.6f', sum(info.resource_allocation));
                info.resource_allocation = info.resource_allocation / sum(info.resource_allocation);
            end
        end
        
        function initializeDefenseCosts(obj)
            % 紧急初始化defense_costs（备用方法）
            if obj.n_stations <= 0 || obj.n_resource_types <= 0
                error('Cannot initialize defense_costs: invalid parameters');
            end
            
            obj.defense_costs = zeros(obj.n_stations, obj.n_resource_types);
            
            for i = 1:obj.n_stations
                for j = 1:obj.n_resource_types
                    base_cost = 10 + (j-1) * 5;
                    random_factor = 0.8 + 0.4 * rand();
                    obj.defense_costs(i, j) = base_cost * random_factor;
                end
            end
        end
        
        function station_values = getStationValues(obj)
            % 获取站点价值评估
            station_values = zeros(1, obj.n_stations);
            for i = 1:obj.n_stations
                station_components = find(obj.component_station_map == i);
                if ~isempty(station_components)
                    % 综合考虑重要性和组件状态
                    importance = mean(obj.component_importance(station_components));
                    health = mean(obj.component_status(station_components));
                    connectivity = sum(sum(obj.network_topology(station_components, :))) / ...
                                 (length(station_components) * obj.total_components);
                    
                    station_values(i) = importance * health * (1 + connectivity);
                else
                    station_values(i) = 0.1;  % 最小值
                end
            end
            
            % 归一化
            if sum(station_values) > 0
                station_values = station_values / sum(station_values);
            else
                station_values = ones(1, obj.n_stations) / obj.n_stations;
            end
        end
        
        function threat_levels = assessThreatLevels(obj)
            % 评估所有站点的威胁级别
            threat_levels = zeros(1, obj.n_stations);
            for i = 1:obj.n_stations
                threat_levels(i) = obj.assessStationThreatLevel(i);
            end
        end
        
        function metrics = getRADIMetrics(obj)
            % 获取RADI相关指标
            metrics = struct();
            
            metrics.current_radi_defender = obj.radi_defender;
            metrics.current_radi_attacker = obj.radi_attacker;
            
            if ~isempty(obj.radi_history.defender)
                metrics.avg_radi_defender = mean(obj.radi_history.defender);
                metrics.std_radi_defender = std(obj.radi_history.defender);
                metrics.min_radi_defender = min(obj.radi_history.defender);
                metrics.max_radi_defender = max(obj.radi_history.defender);
                
                % 计算趋势
                if length(obj.radi_history.defender) > 10
                    recent = obj.radi_history.defender(end-9:end);
                    earlier = obj.radi_history.defender(end-19:end-10);
                    metrics.radi_trend_defender = mean(recent) - mean(earlier);
                else
                    metrics.radi_trend_defender = 0;
                end
            else
                metrics.avg_radi_defender = obj.radi_defender;
                metrics.std_radi_defender = 0;
                metrics.min_radi_defender = obj.radi_defender;
                metrics.max_radi_defender = obj.radi_defender;
                metrics.radi_trend_defender = 0;
            end
            
            if ~isempty(obj.radi_history.attacker)
                metrics.avg_radi_attacker = mean(obj.radi_history.attacker);
                metrics.std_radi_attacker = std(obj.radi_history.attacker);
                metrics.min_radi_attacker = min(obj.radi_history.attacker);
                metrics.max_radi_attacker = max(obj.radi_history.attacker);
                
                if length(obj.radi_history.attacker) > 10
                    recent = obj.radi_history.attacker(end-9:end);
                    earlier = obj.radi_history.attacker(end-19:end-10);
                    metrics.radi_trend_attacker = mean(recent) - mean(earlier);
                else
                    metrics.radi_trend_attacker = 0;
                end
            else
                metrics.avg_radi_attacker = obj.radi_attacker;
                metrics.std_radi_attacker = 0;
                metrics.min_radi_attacker = obj.radi_attacker;
                metrics.max_radi_attacker = obj.radi_attacker;
                metrics.radi_trend_attacker = 0;
            end
            
            % 策略收敛性
            metrics.strategy_convergence_defender = obj.calculateStrategyConvergence('defender');
            metrics.strategy_convergence_attacker = obj.calculateStrategyConvergence('attacker');
            
            % 博弈效率
            if ~isempty(obj.game_outcomes)
                metrics.avg_attack_success_rate = mean(obj.game_outcomes(:, 1) ./ max(1, obj.game_outcomes(:, 3)));
                metrics.avg_defense_success_rate = mean(obj.game_outcomes(:, 2) ./ max(1, obj.game_outcomes(:, 4)));
            else
                metrics.avg_attack_success_rate = 0;
                metrics.avg_defense_success_rate = 0;
            end
            
            if ~isempty(obj.resource_utilization)
                metrics.avg_resource_utilization = mean(obj.resource_utilization);
            else
                metrics.avg_resource_utilization = 0;
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
            
            % 计算最近策略与最优策略的距离
            recent_steps = min(10, size(strategy_history, 1));
            recent_strategies = strategy_history(end-recent_steps+1:end, :);
            
            distances = zeros(recent_steps, 1);
            for i = 1:recent_steps
                distances(i) = norm(recent_strategies(i, :) - optimal_strategy);
            end
            
            % 收敛性：距离的稳定性和接近程度
            avg_distance = mean(distances);
            std_distance = std(distances);
            
            % 归一化收敛性分数
            convergence = 1 - (avg_distance + std_distance) / 2;
            convergence = max(0, min(1, convergence));
        end
        
        function analysis = analyzeGameDynamics(obj)
            % 分析博弈动态
            analysis = struct();
            
            if isempty(obj.game_outcomes)
                analysis.game_balance = 0.5;
                analysis.system_stability = 0.5;
                analysis.learning_efficiency_defender = 0;
                analysis.learning_efficiency_attacker = 0;
                return;
            end
            
            % 博弈平衡性
            total_games = size(obj.game_outcomes, 1);
            if total_games > 0
                attack_wins = sum(obj.game_outcomes(:, 1) > obj.game_outcomes(:, 2));
                defense_wins = total_games - attack_wins;
                analysis.game_balance = 1 - abs(attack_wins - defense_wins) / total_games;
            else
                analysis.game_balance = 0.5;
            end
            
            % 系统稳定性
            if length(obj.radi_history.defender) > 20
                recent_radi_def = obj.radi_history.defender(end-19:end);
                recent_radi_att = obj.radi_history.attacker(end-19:end);
                
                stability_def = 1 - std(recent_radi_def) / (mean(recent_radi_def) + 0.01);
                stability_att = 1 - std(recent_radi_att) / (mean(recent_radi_att) + 0.01);
                
                analysis.system_stability = (stability_def + stability_att) / 2;
            else
                analysis.system_stability = 0.5;
            end
            
            % 学习效率
            if length(obj.radi_history.defender) > 10
                initial_radi_def = mean(obj.radi_history.defender(1:10));
                final_radi_def = mean(obj.radi_history.defender(end-9:end));
                analysis.learning_efficiency_defender = max(0, (initial_radi_def - final_radi_def) / initial_radi_def);
                
                initial_radi_att = mean(obj.radi_history.attacker(1:10));
                final_radi_att = mean(obj.radi_history.attacker(end-9:end));
                analysis.learning_efficiency_attacker = max(0, (initial_radi_att - final_radi_att) / initial_radi_att);
            else
                analysis.learning_efficiency_defender = 0;
                analysis.learning_efficiency_attacker = 0;
            end
            
            % 归一化所有指标到[0,1]
            fields = fieldnames(analysis);
            for i = 1:length(fields)
                analysis.(fields{i}) = max(0, min(1, analysis.(fields{i})));
            end
        end
        
        function analyzeAttackStrategyEvolution(obj)
            % 分析和可视化攻击策略的演变
            
            if isempty(obj.strategy_history.attacker)
                fprintf('没有足够的历史数据进行分析\n');
                return;
            end
            
            figure('Name', '攻击策略演变分析', 'Position', [100, 100, 1200, 800]);
            
            % 子图1：策略权重随时间的变化
            subplot(2,2,1);
            plot(obj.strategy_history.attacker, 'LineWidth', 1.5);
            xlabel('时间步');
            ylabel('攻击概率');
            title('各站点攻击概率演变');
            legend(arrayfun(@(x) sprintf('站点 %d', x), 1:obj.n_stations, 'UniformOutput', false), ...
                   'Location', 'best');
            grid on;
            
            % 子图2：攻击成功次数随时间的变化
            subplot(2,2,2);
            if ~isempty(obj.attack_success_history)
                plot(obj.attack_success_history, 'b-', 'LineWidth', 2);
                hold on;
                
                % 添加移动平均线
                if length(obj.attack_success_history) > 10
                    ma = movmean(obj.attack_success_history, 10);
                    plot(ma, 'r--', 'LineWidth', 1.5);
                    legend('实际成功次数', '10步移动平均', 'Location', 'best');
                end
                
                % 标记策略改变点
                if obj.strategy_change_count > 0 && obj.last_strategy_change > 0
                    xline(obj.last_strategy_change, 'g--', 'LineWidth', 1.5);
                end
                
                xlabel('时间步');
                ylabel('攻击成功次数');
                title(sprintf('攻击成功历史 (总改变次数: %d)', obj.strategy_change_count));
                grid on;
            end
            
            % 子图3：策略熵的变化
            subplot(2,2,3);
            entropy_history = zeros(size(obj.strategy_history.attacker, 1), 1);
            for t = 1:size(obj.strategy_history.attacker, 1)
                p = obj.strategy_history.attacker(t, :);
                p = p(p > 0);
                entropy_history(t) = -sum(p .* log(p));
            end
            plot(entropy_history, 'g-', 'LineWidth', 2);
            xlabel('时间步');
            ylabel('策略熵');
            title('攻击策略集中度（熵越低越集中）');
            grid on;
            
            % 子图4：重点攻击站点的识别
            subplot(2,2,4);
            [~, max_prob_stations] = max(obj.strategy_history.attacker, [], 2);
            histogram(max_prob_stations, 0.5:1:(obj.n_stations+0.5));
            xlabel('站点编号');
            ylabel('作为重点目标的次数');
            title('各站点作为重点攻击目标的频率');
            grid on;
            
            % 添加统计信息
            if ~isempty(obj.attack_success_history)
                total_success = sum(obj.attack_success_history);
                avg_success = mean(obj.attack_success_history);
                fprintf('\n=== 攻击策略分析总结 ===\n');
                fprintf('总攻击成功次数: %d\n', total_success);
                fprintf('平均每步成功次数: %.2f\n', avg_success);
                fprintf('策略改变次数: %d\n', obj.strategy_change_count);
                fprintf('最后一次策略改变: 时间步 %d\n', obj.last_strategy_change);
                fprintf('初始重点站点: %d\n', obj.initial_focus_station);
                fprintf('当前重点站点: %d\n', max_prob_stations(end));
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
            
            % 攻击相关统计
            if ~isempty(obj.attack_success_history)
                summary.total_attack_success = sum(obj.attack_success_history);
                summary.avg_attack_success_rate = mean(obj.attack_success_history);
                summary.attack_strategy_changes = obj.strategy_change_count;
            else
                summary.total_attack_success = 0;
                summary.avg_attack_success_rate = 0;
                summary.attack_strategy_changes = 0;
            end
            
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
        
        function saveData(obj, filename)
            % 保存环境数据
            if nargin < 2
                filename = sprintf('tcs_env_data_%s.mat', datestr(now, 'yyyymmdd_HHMMSS'));
            end
            
            save_data = struct();
            save_data.config = struct('n_stations', obj.n_stations, ...
                                    'n_resource_types', obj.n_resource_types, ...
                                    'n_attack_types', obj.n_attack_types);
            save_data.history = struct('strategy', obj.strategy_history, ...
                                     'radi', obj.radi_history, ...
                                     'attack_success', obj.attack_success_history, ...
                                     'game_outcomes', obj.game_outcomes);
            save_data.final_state = struct('defender_strategy', obj.defender_strategy, ...
                                         'attacker_strategy', obj.attacker_strategy, ...
                                         'component_status', obj.component_status);
            save_data.summary = obj.getGameSummary();
            
            save(filename, 'save_data');
            fprintf('数据已保存到: %s\n', filename);
        end
        
        function loadData(obj, filename)
            % 加载环境数据
            try
                load_data = load(filename);
                if isfield(load_data, 'save_data')
                    data = load_data.save_data;
                    
                    % 恢复历史数据
                    if isfield(data, 'history')
                        obj.strategy_history = data.history.strategy;
                        obj.radi_history = data.history.radi;
                        obj.attack_success_history = data.history.attack_success;
                        obj.game_outcomes = data.history.game_outcomes;
                    end
                    
                    % 恢复最终状态
                    if isfield(data, 'final_state')
                        obj.defender_strategy = data.final_state.defender_strategy;
                        obj.attacker_strategy = data.final_state.attacker_strategy;
                        if isfield(data.final_state, 'component_status')
                            obj.component_status = data.final_state.component_status;
                        end
                    end
                    
                    fprintf('数据已从 %s 加载\n', filename);
                else
                    warning('文件格式不正确');
                end
            catch ME
                fprintf('加载数据时出错: %s\n', ME.message);
            end
        end
        
        function allocation = calculateResourceAllocationFromActions(obj, defender_actions)
            % 将防御动作转换为资源分配向量（确保归一化）
            
            % 初始化分配向量
            allocation = zeros(1, obj.n_resource_types);
            
            % 方法1：基于动作计数的简单分配
            total_actions = 0;
            for station = 1:obj.n_stations
                if defender_actions(station) > 0 && defender_actions(station) <= obj.n_resource_types
                    allocation(defender_actions(station)) = allocation(defender_actions(station)) + 1;
                    total_actions = total_actions + 1;
                end
            end
            
            % 如果有动作，则归一化
            if total_actions > 0
                allocation = allocation / total_actions;
            else
                % 没有动作时，平均分配
                allocation = ones(1, obj.n_resource_types) / obj.n_resource_types;
            end
            
            % 确保总和为1
            if abs(sum(allocation) - 1) > 1e-6
                allocation = allocation / sum(allocation);
            end
        end
    end
end